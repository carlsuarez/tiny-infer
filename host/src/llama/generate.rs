//! Text generation for the Llama path: the [`Sampler`], the streaming [`generate`]
//! loop, kernel selection, and the quantization group-size check.

use std::io::{self, Write};
use std::time::Instant;

use llama::memory::{arena_floats, max_proj_d_in};
use llama::{forward, Config, Kernel, ModelWeights, QuantScratch, RunState};
use engine::sample::ProbIndex;
use engine::{Arena, Sampler};

use humansize::{format_size, BINARY};

use crate::error::HostError;
use crate::llama::tokenizer::{Vocab, BOS_ID};

/// Generate text from `prompt` and stream it to stdout.
///
/// Tokenizes the prompt (with BOS), then for each position runs [`forward`] and
/// picks the next token with a [`Sampler`]: while still inside the prompt it
/// replays the prompt tokens; afterwards it samples from the logits (greedy when
/// `temperature == 0`). Stops at `steps` or when the model emits BOS (the sequence
/// delimiter). Each decoded piece is flushed immediately, so the text streams live
/// rather than appearing in bursts; a closed reader (e.g. `| head`) ends the run
/// quietly. A throughput line — prompt (prefill) and generation (decode) rates — is
/// written to stderr.
pub(crate) fn generate(
    config: &Config,
    weights: &ModelWeights,
    vocab: &Vocab,
    prompt: &str,
    steps: Option<usize>,
    mut sampler: Sampler,
    kernel: Kernel,
) -> Result<(), HostError> {
    // `encode` always yields at least the BOS token, so `prompt_tokens[0]` is safe.
    let prompt_tokens = vocab.encode(prompt, true);
    let steps = steps.unwrap_or(config.seq_len).min(config.seq_len);

    // One arena, sized exactly to the budget; the run state is carved once.
    let mut arena_buf = vec![0.0f32; arena_floats(config)];
    let mut arena = Arena::new(&mut arena_buf);
    let mut state =
        RunState::new(&mut arena, config).map_err(|e| HostError::engine("<arena>", e))?;

    // The int8 (W8A8) path needs a per-matmul activation scratch; allocate it only when
    // the weights are quantized so an fp32 run carries none of it.
    let n = max_proj_d_in(config);
    let mut qbuf = matches!(weights, ModelWeights::Q8(_))
        .then(|| (vec![0i8; n], vec![0.0f32; n], vec![0.0f32; n]));
    let mut scratch = match qbuf.as_mut() {
        Some((xq, sc, gs)) => Some(
            QuantScratch::new(xq, sc, gs, n).map_err(|e| HostError::engine("<arena>", e))?,
        ),
        None => None,
    };

    // Caller-owned nucleus scratch for top-p sampling (the engine sampler never allocates);
    // sized to the vocabulary once and reused every step. Greedy/full sampling ignore it.
    let mut probindex = vec![ProbIndex::default(); config.vocab_size];

    let stdout = io::stdout();
    let mut out = stdout.lock();

    // Prefill (processing the prompt) and decode (sampling new tokens) scale very
    // differently, so time them separately. `decode_start` marks the boundary: the
    // instant of the first sample, taken once the prompt has been fully consumed.
    let start = Instant::now();
    let mut decode_start: Option<Instant> = None;
    let mut decode_tokens = 0usize;

    let mut token = prompt_tokens[0];
    let mut pos = 0usize;
    while pos < steps {
        let logits = forward(
            config,
            weights,
            &mut state,
            token,
            pos,
            kernel,
            &mut scratch,
        );
        let next = if pos + 1 < prompt_tokens.len() {
            prompt_tokens[pos + 1] // still replaying the prompt
        } else {
            decode_start.get_or_insert_with(Instant::now); // prompt fully consumed
            decode_tokens += 1;
            sampler.sample(logits, &mut probindex) // generate the next token
        };
        pos += 1;
        // BOS delimits sequences in llama2.c — stop if the model emits it.
        if next == BOS_ID {
            break;
        }
        let piece = vocab.decode(token, next);
        // Flush every piece so the text streams to the terminal as it is produced,
        // instead of appearing in bursts at line boundaries (the lock is a
        // `LineWriter`) or only at the end (when stdout is piped/block-buffered).
        if let Err(e) = out.write_all(&piece).and_then(|()| out.flush()) {
            // A closed reader (e.g. `tiny-infer … | head`) is routine for a
            // streaming CLI: stop quietly instead of reporting an I/O error.
            if e.kind() == io::ErrorKind::BrokenPipe {
                return Ok(());
            }
            return Err(HostError::io("<stdout>", e));
        }
        token = next;
    }
    let _ = out.write_all(b"\n");
    let _ = out.flush();
    let end = Instant::now();

    report_throughput(
        start,
        decode_start,
        end,
        prompt_tokens.len().min(pos),
        decode_tokens,
    );
    Ok(())
}

/// Print prompt (prefill) and generation (decode) throughput to stderr.
///
/// The two phases are timed separately because they scale differently: prefill runs
/// one forward pass per prompt token before the first generated token appears, then
/// decode samples one token per forward pass. `decode_start` is `None` when no token
/// was generated (e.g. `--steps` shorter than the prompt), so only the prompt rate
/// is shown.
fn report_throughput(
    start: Instant,
    decode_start: Option<Instant>,
    end: Instant,
    prompt_tokens: usize,
    decode_tokens: usize,
) {
    let rate = |toks: usize, secs: f64| {
        if secs > 0.0 {
            toks as f64 / secs
        } else {
            f64::INFINITY
        }
    };
    match decode_start {
        Some(split) => {
            let prefill = split.duration_since(start).as_secs_f64();
            let decode = end.duration_since(split).as_secs_f64();
            eprintln!(
                "[prompt {prompt_tokens} tok in {prefill:.3}s ({:.1} tok/s), \
                 generated {decode_tokens} tok in {decode:.3}s ({:.1} tok/s)]",
                rate(prompt_tokens, prefill),
                rate(decode_tokens, decode),
            );
        }
        None => {
            let secs = end.duration_since(start).as_secs_f64();
            eprintln!(
                "[prompt {prompt_tokens} tok in {secs:.3}s ({:.1} tok/s), generated 0 tok]",
                rate(prompt_tokens, secs),
            );
        }
    }
}

/// Resolve the sampling RNG seed: an explicit `--seed` for reproducibility, else one drawn
/// from OS entropy. The engine's [`Sampler`] owns the PRNG itself (seeded from this); the
/// host's `rand` is used here only to draw that seed from the OS (the engine's
/// no-default-features `rand` carries no OS entropy of its own).
pub(crate) fn resolve_seed(seed: Option<u64>) -> u64 {
    seed.unwrap_or_else(rand::random)
}

/// Pick the matmul kernel from the flags, with a graceful fallback when `--dotprod`
/// is requested on a CPU without a hardware int8 dot-product instruction.
///
/// `--scalar` wins outright (reference path). `--dotprod` selects the int8 dot-product
/// kernel only when the CPU supports it — x86 AVX-512 VNNI or ARM NEON `sdot` (the engine
/// still falls back to SIMD internally for the fp32 path and x86 group sizes VNNI can't
/// take); otherwise it warns and uses SIMD. The default is SIMD.
pub(crate) fn select_kernel(scalar: bool, dotprod: bool) -> Kernel {
    if scalar {
        return Kernel::Scalar;
    }
    if dotprod {
        if dotprod_available() {
            return Kernel::Dotprod;
        }
        eprintln!(
            "[--dotprod: CPU lacks a hardware int8 dot-product instruction; using SIMD instead]"
        );
    }
    Kernel::Simd
}

/// Whether this CPU has the AVX-512 VNNI + VL features the x86 int8 kernel needs.
#[cfg(target_arch = "x86_64")]
fn dotprod_available() -> bool {
    std::is_x86_feature_detected!("avx512vnni") && std::is_x86_feature_detected!("avx512vl")
}

/// Whether this CPU has the NEON `dotprod` extension the ARM `sdot` int8 kernel needs.
#[cfg(target_arch = "aarch64")]
fn dotprod_available() -> bool {
    std::arch::is_aarch64_feature_detected!("dotprod")
}

/// Other targets have no hardware int8 dot product; the engine falls back to SIMD.
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn dotprod_available() -> bool {
    false
}

/// Reject a quantization group size that does not evenly divide the weight
/// dimensions. Quantization groups never straddle a row, so the group size must
/// divide every matmul's input dimension; otherwise the int8 tensors would be
/// silently truncated/corrupted.
pub(crate) fn check_group_size(c: &Config, gs: usize) -> Result<(), HostError> {
    if gs == 0 {
        return Err(HostError::Usage("--group-size must be at least 1".into()));
    }
    for (name, d) in [
        ("dim", c.dim),
        ("hidden_dim", c.hidden_dim),
        ("kv_dim", c.kv_dim()),
    ] {
        if !d.is_multiple_of(gs) {
            return Err(HostError::Usage(format!(
                "--group-size {gs} must divide {name} ({d})"
            )));
        }
    }
    Ok(())
}

/// Note the int8 quantization on stderr: the resident int8 weight size (data +
/// scales, embedding included) against the fp32 checkpoint we freed afterward.
pub(crate) fn report_quantization(
    data: &[i8],
    scales: &[f32],
    fp32_weight_bytes: usize,
    gs: usize,
) {
    let int8_bytes = data.len() + scales.len() * 4;
    eprintln!(
        "[quantized to int8 (group_size {gs}): {} weights resident \
         ({} int8 + {} scales); freed the {} fp32 checkpoint]",
        format_size(int8_bytes, BINARY),
        format_size(data.len(), BINARY),
        format_size(scales.len() * 4, BINARY),
        format_size(fp32_weight_bytes, BINARY),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stories_like_config() -> Config {
        Config {
            dim: 288,
            hidden_dim: 768,
            n_layers: 6,
            n_heads: 6,
            n_kv_heads: 6,
            vocab_size: 32000,
            seq_len: 256,
            shared_weights: true,
        }
    }

    #[test]
    fn check_group_size_accepts_divisors_rejects_others() {
        let c = stories_like_config();
        // 32 and 96 divide dim (288), hidden_dim (768), and kv_dim (288).
        assert!(check_group_size(&c, 32).is_ok());
        assert!(check_group_size(&c, 96).is_ok());
        // 64 divides 768 but not 288.
        assert!(matches!(check_group_size(&c, 64), Err(HostError::Usage(_))));
        assert!(matches!(check_group_size(&c, 0), Err(HostError::Usage(_))));
    }

    #[test]
    fn resolve_seed_prefers_an_explicit_seed() {
        // An explicit --seed passes through unchanged (the reproducibility contract);
        // without one, a seed is drawn from the OS (just check it returns something).
        assert_eq!(resolve_seed(Some(42)), 42);
        let _ = resolve_seed(None);
    }
}
