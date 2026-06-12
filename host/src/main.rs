//! `tiny-infer` command-line host.
//!
//! Two modes:
//! * **Report** (default): load a model checkpoint and optional tokenizer, then
//!   print the parsed config, file-validation result, tokenizer metadata, and the
//!   pre-computed memory budget.
//! * **Generate** (`--prompt`): tokenize the prompt, run the forward pass per
//!   position, sample the next token (greedy at `--temperature 0`, otherwise from
//!   the temperature-scaled softmax), and stream the decoded text. Requires a
//!   tokenizer.

mod convert;
mod error;
mod loader;
mod seq2seq;
mod tokenizer;

use std::io::{self, Write};
use std::process::ExitCode;
use std::time::Instant;

use engine::math::{argmax, maxf};
use engine::memory::{arena_floats, max_proj_d_in, MemoryBudget};
use engine::quantize::{quantize_weights, quantized_scale_count, quantized_weight_count};
use engine::{
    forward, Arena, Config, Kernel, ModelFormat, ModelWeights, QuantScratch, QuantizedWeights,
    RunState,
};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use crate::convert::Target;
use crate::error::HostError;
use crate::loader::Model;
use crate::seq2seq::Seq2SeqModel;
use crate::tokenizer::{Vocab, BOS_ID};

const USAGE: &str = "\
tiny-infer — embedded-style Llama-2 inference engine

USAGE:
    tiny-infer [OPTIONS] <MODEL> [TOKENIZER]

ARGS:
    <MODEL>        Path to a llama2.c checkpoint (.bin)
    [TOKENIZER]    Path to tokenizer.bin (optional)

OPTIONS:
    -m, --model <PATH>        Model checkpoint path (alternative to positional)
    -t, --tokenizer <PATH>    Tokenizer path (alternative to positional)
    -p, --prompt <TEXT>       Prompt to continue (enables text generation)
    -n, --steps <N>           Max tokens to generate (default: model seq_len)
        --temperature <F>     Sampling temperature (default 0 = greedy/deterministic;
                              higher = more random)
        --topp <F>            Nucleus (top-p) sampling threshold in (0,1); sample only
                              from the most-probable tokens summing to F (default: off)
        --seed <N>            RNG seed for reproducible sampling (default: random)
    -q, --quantize            Quantize matmul weights to int8 (group-wise) before
                              generating — smaller weight footprint, slightly lossy
                              (scalar int8 is not faster than fp32; the win is memory)
        --group-size <N>      Int8 quantization group size; must divide the model's
                              dim, hidden_dim, and kv_dim (default: 32)
        --scalar              Use the scalar matmul kernels instead of the default
                              SIMD (core::simd) ones — the readable reference path
        --dotprod             Use the hardware int8 dot-product kernel for --quantize
                              (x86 AVX-512 VNNI or ARM NEON sdot; falls back to SIMD
                              if the CPU has neither)
        --convert <PATH>      Convert the checkpoint to another format, write it to
                              PATH, and exit (no generation)
        --to <v1|v2>          Conversion target (default v2): v1 = fp32 with the
                              256-byte versioned header; v2 = int8-quantized Q8_0,
                              runq.c's format (uses --group-size)
    -h, --help                Print this help and exit

Reads all three llama2.c checkpoint formats: legacy (v0), v1 (fp32), and v2
(pre-quantized int8 — implies the int8 path; -q is redundant). With no --prompt,
tiny-infer prints the model config, validates the files, and reports the memory
budget. With --prompt (and a tokenizer) it generates text.

Seq2seq (Marian / OPUS-MT translation) checkpoints exported by
scripts/export_marian.py are detected automatically by their magic; for now they
support report mode only (translation arrives in a later milestone).";

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(HostError::Usage(msg)) => {
            eprintln!("error: {msg}\n\n{USAGE}");
            ExitCode::from(2)
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), HostError> {
    let args = Args::parse(std::env::args().skip(1))?;
    if args.help {
        println!("{USAGE}");
        return Ok(());
    }
    let model_path = args
        .model
        .clone()
        .ok_or_else(|| HostError::Usage("no model file given".into()))?;
    if args.to.is_some() && args.convert.is_none() {
        return Err(HostError::Usage(
            "`--to` requires `--convert <PATH>`".into(),
        ));
    }
    if args.convert.is_some() && args.prompt.is_some() {
        return Err(HostError::Usage(
            "`--convert` writes a checkpoint and exits; it cannot be combined with --prompt".into(),
        ));
    }

    // Seq2seq (Marian) checkpoints are a different architecture with their own
    // loader; route them by the magic before touching the llama2.c parsers.
    if seq2seq::is_seq2seq_file(&model_path)? {
        return run_seq2seq(&model_path, &args);
    }

    let model = Model::load(&model_path)?;
    let config = model.config; // Copy, so generation can outlive the file buffer.

    // Conversion mode: write the requested format and exit.
    if let Some(out_path) = args.convert.as_deref() {
        let target = args.to.unwrap_or(Target::V2);
        if target == Target::V2 {
            check_group_size(&config, args.group_size)?;
        }
        return convert::convert(&model, out_path.as_ref(), target, args.group_size);
    }

    // Generation mode: requires a prompt and a tokenizer.
    if let Some(prompt) = args.prompt.as_deref() {
        let tok_path = args.tokenizer.ok_or_else(|| {
            HostError::Usage("generation (--prompt) requires a tokenizer (--tokenizer)".into())
        })?;
        let vocab = Vocab::load(&tok_path, config.vocab_size)?;
        let sampler = Sampler::new(config.vocab_size, args.temperature, args.topp, args.seed);
        let kernel = select_kernel(args.scalar, args.dotprod);

        // A v2 checkpoint is already int8: de-interleave its tensors into the
        // engine layout and go straight to the Q8 path — no quantization work,
        // and no fp32 weights ever resident.
        if let ModelFormat::V2 { group_size } = model.format {
            if args.quantize {
                eprintln!(
                    "[--quantize: checkpoint is already int8 (v2, group_size {group_size}); \
                     flag ignored]"
                );
            }
            let bufs = model.unpack_q8()?;
            drop(model); // free the raw file; the unpacked buffers are all we need
            let qw = QuantizedWeights::new(
                &bufs.data,
                &bufs.scales,
                &bufs.rms_att,
                &bufs.rms_ffn,
                &bufs.rms_final,
                bufs.group_size,
                &config,
            )
            .map_err(|e| HostError::engine(&model_path, e))?;
            return generate(
                &config,
                &ModelWeights::Q8(qw),
                &vocab,
                prompt,
                args.steps,
                sampler,
                kernel,
            );
        }

        if args.quantize {
            let gs = args.group_size;
            check_group_size(&config, gs)?;
            // Quantize from the fp32 file and copy out the tiny fp32 RMSNorm gains,
            // then drop the file so only the int8 weights stay resident while we
            // generate. (`QuantizedWeights` borrows none of the original file.)
            let mut data = vec![0i8; quantized_weight_count(&config)];
            let mut scales = vec![0.0f32; quantized_scale_count(&config, gs)];
            let (rms_att, rms_ffn, rms_final);
            {
                let weights = model.weights()?;
                quantize_weights(&weights, &mut data, &mut scales, gs, &config);
                rms_att = weights.rms_att.to_vec();
                rms_ffn = weights.rms_ffn.to_vec();
                rms_final = weights.rms_final.to_vec();
            }
            report_quantization(&data, &scales, model.weight_bytes(), gs);
            drop(model); // free the fp32 checkpoint (the bulk of the memory)
            let qw = engine::QuantizedWeights::new(
                &data, &scales, &rms_att, &rms_ffn, &rms_final, gs, &config,
            )
            .map_err(|e| HostError::engine(&model_path, e))?;
            return generate(
                &config,
                &ModelWeights::Q8(qw),
                &vocab,
                prompt,
                args.steps,
                sampler,
                kernel,
            );
        }

        let weights = model.weights()?;
        return generate(
            &config,
            &ModelWeights::F32(weights),
            &vocab,
            prompt,
            args.steps,
            sampler,
            kernel,
        );
    }

    // Report mode: validate the weight layout, then print the model summary.
    // (v2 has no fp32 view; its layout was already size-validated at load.)
    if !matches!(model.format, ModelFormat::V2 { .. }) {
        let _ = model.weights()?;
    }
    report_model(&model_path, &model);
    if let Some(tok_path) = args.tokenizer {
        let vocab = Vocab::load(&tok_path, model.config.vocab_size)?;
        report_tokenizer(&tok_path, &vocab);
    } else {
        println!("\nTokenizer: (none provided — pass a path to inspect tokenizer.bin)");
    }

    report_memory(&model);
    Ok(())
}

/// Handle a seq2seq (Marian) checkpoint: validate it and print the report.
///
/// Translation, conversion, and quantization for this architecture arrive in
/// later milestones, so any flag that implies them is rejected up front with a
/// clear message rather than a confusing downstream failure.
fn run_seq2seq(path: &str, args: &Args) -> Result<(), HostError> {
    if args.prompt.is_some() || args.convert.is_some() || args.quantize {
        return Err(HostError::Usage(
            "seq2seq (Marian) checkpoints currently support report mode only; \
             translation, conversion, and quantization arrive in later milestones"
                .into(),
        ));
    }

    let model = Seq2SeqModel::load(path)?;
    // Validate the tensor carve, not just the byte count.
    let _ = model.weights()?;
    report_seq2seq(path, &model);
    if args.tokenizer.is_some() {
        println!("\nTokenizer: (SentencePiece support for seq2seq models lands with translation)");
    }
    Ok(())
}

/// Generate text from `prompt` and stream it to stdout.
///
/// Tokenizes the prompt (with BOS), then for each position runs [`forward`] and
/// picks the next token with a [`Sampler`]: while still inside the prompt it
/// replays the prompt tokens; afterwards it samples from the logits (greedy when
/// `temperature == 0`). Stops at `steps` or when the model emits BOS (the sequence
/// delimiter). Each decoded piece is flushed immediately, so the text streams live
/// rather than appearing in bursts; a closed reader (e.g. `| head`) ends the run
/// quietly. A throughput line — prompt (prefill) and generation (decode) rates —
/// is written to stderr.
fn generate(
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
            QuantScratch::new(xq, sc, gs, config).map_err(|e| HostError::engine("<arena>", e))?,
        ),
        None => None,
    };

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
            sampler.sample(logits) // generate the next token
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

/// Turns a logits vector into the next token id.
///
/// Owns the RNG and a reusable scratch buffer so [`Sampler::sample`] allocates
/// nothing per token. Three modes, decided by `temperature` / `topp`:
/// * `temperature == 0` → deterministic greedy ([`argmax`]); this is the path the
///   llama2.c parity test exercises.
/// * `temperature > 0`, `topp` disabled (`<= 0` or `>= 1`) → draw from the full
///   softmax(`logits / temperature`) distribution.
/// * `temperature > 0`, `0 < topp < 1` → "nucleus" sampling: draw only from the
///   smallest set of most-probable tokens whose cumulative probability reaches
///   `topp`, trimming the unreliable long tail.
struct Sampler {
    temperature: f32,
    topp: f32,
    rng: StdRng,
    /// `(probability, token id)` scratch for top-p, sized to the vocab up front and
    /// reused every step so sorting the nucleus needs no per-token allocation.
    probindex: Vec<(f32, usize)>,
}

impl Sampler {
    fn new(vocab_size: usize, temperature: f32, topp: f32, seed: Option<u64>) -> Sampler {
        // An explicit `--seed` makes sampling reproducible; otherwise seed from the
        // OS. (The RNG is unused at temperature 0, which is greedy.)
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_rng(&mut rand::rng()),
        };
        Sampler {
            temperature,
            topp,
            rng,
            probindex: Vec::with_capacity(vocab_size),
        }
    }

    fn sample(&mut self, logits: &[f32]) -> usize {
        if self.temperature == 0.0 {
            return argmax(logits);
        }

        // Unnormalized softmax weight of one logit (max-shifted for stability).
        // Capture by value so the closure doesn't borrow `self` (the sampling
        // helpers below need `&mut self`).
        let (max, temperature) = (maxf(logits), self.temperature);
        let weight = move |v: f32| f32::exp((v - max) / temperature);
        let sum: f32 = logits.iter().copied().map(weight).sum();

        if self.topp <= 0.0 || self.topp >= 1.0 {
            return self.sample_full(logits, &weight, sum);
        }
        self.sample_topp(logits, &weight, sum)
    }

    /// Draw from the full distribution via inverse-transform sampling, comparing a
    /// scaled uniform draw against the running unnormalized cumulative weight.
    fn sample_full(&mut self, logits: &[f32], weight: &impl Fn(f32) -> f32, sum: f32) -> usize {
        let target = self.rng.random::<f32>() * sum; // uniform in [0, sum)
        let mut cdf = 0.0f32;
        for (i, &v) in logits.iter().enumerate() {
            cdf += weight(v);
            if cdf > target {
                return i;
            }
        }
        logits.len() - 1 // only reached if f32 rounding leaves cdf just under target
    }

    /// Top-p (nucleus) sampling: keep the smallest set of highest-probability
    /// tokens whose cumulative probability reaches `topp`, then sample within it.
    fn sample_topp(&mut self, logits: &[f32], weight: &impl Fn(f32) -> f32, sum: f32) -> usize {
        // Tokens below this probability cannot belong to the nucleus, so crop them
        // before sorting. Holds whenever `topp >= 1/n`.
        let cutoff = (1.0 - self.topp) / (logits.len() - 1) as f32;
        self.probindex.clear();
        for (i, &v) in logits.iter().enumerate() {
            let p = weight(v) / sum;
            if p >= cutoff {
                self.probindex.push((p, i));
            }
        }
        if self.probindex.is_empty() {
            // `topp` smaller than 1/vocab cropped everything; fall back to greedy.
            return argmax(logits);
        }
        // Highest probability first. `total_cmp` gives a total order (no NaN unwrap).
        self.probindex.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));

        // Truncate where the cumulative probability first exceeds `topp`.
        let mut cumulative = 0.0f32;
        let mut last = self.probindex.len() - 1;
        for (i, &(p, _)) in self.probindex.iter().enumerate() {
            cumulative += p;
            if cumulative > self.topp {
                last = i;
                break;
            }
        }

        // Sample within the nucleus, renormalized by its cumulative mass.
        let target = self.rng.random::<f32>() * cumulative;
        let mut cdf = 0.0f32;
        for &(p, idx) in &self.probindex[..=last] {
            cdf += p;
            if cdf > target {
                return idx;
            }
        }
        self.probindex[last].1 // f32 rounding fallback
    }
}

fn report_model(path: &str, model: &Model) {
    let c = &model.config;
    println!("Model: {path}");
    println!("  status         OK (file size matches config)");
    println!("  format         {}", format_label(model.format));
    println!("  dim            {}", c.dim);
    println!("  hidden_dim     {}", c.hidden_dim);
    println!("  n_layers       {}", c.n_layers);
    println!("  n_heads        {}", c.n_heads);
    println!("  n_kv_heads     {}", c.n_kv_heads);
    println!("  vocab_size     {}", c.vocab_size);
    println!("  seq_len        {}", c.seq_len);
    println!("  head_size      {}", c.head_size());
    println!("  kv_dim         {}", c.kv_dim());
    println!("  kv_mul         {}", c.kv_mul());
    println!(
        "  shared_weights {} (output projection {})",
        c.shared_weights,
        if c.shared_weights {
            "reuses token embedding"
        } else {
            "is a separate matrix"
        }
    );
    println!(
        "  weights        {} ({})",
        human_bytes(model.weight_bytes()),
        commas(model.weight_bytes())
    );
}

fn report_tokenizer(path: &str, vocab: &Vocab) {
    println!("\nTokenizer: {path}");
    println!("  tokens             {}", vocab.len());
    println!("  max_token_length   {}", vocab.max_token_length);
    println!("  sample tokens:");
    for id in 0..4.min(vocab.len()) {
        print_sample(id, vocab);
    }
    if let Some(id) = vocab.tokens.iter().position(|t| t.is_byte_fallback()) {
        println!("  first byte-fallback token:");
        print_sample(id, vocab);
    }
}

fn print_sample(id: usize, vocab: &Vocab) {
    let t = &vocab.tokens[id];
    println!("    [{id:>5}] score {:>9.4}  {:?}", t.score, t.display());
}

fn report_memory(model: &Model) {
    let budget = MemoryBudget::for_config(&model.config);
    println!("\nMemory budget (working arena, pre-allocated once):");
    println!(
        "  activations    {:>10}  ({} f32)",
        human_bytes(budget.activation_bytes()),
        commas(budget.activation_floats)
    );
    println!(
        "  KV cache       {:>10}  ({} f32)",
        human_bytes(budget.kv_cache_bytes()),
        commas(budget.kv_cache_floats)
    );
    println!(
        "  arena total    {:>10}  ({} f32)",
        human_bytes(budget.total_bytes()),
        commas(budget.total_floats())
    );
    println!("  weights (disk) {:>10}", human_bytes(model.weight_bytes()));
    println!(
        "  peak RAM       {:>10}  (weights + arena)",
        human_bytes(model.weight_bytes() + budget.total_bytes())
    );
}

fn report_seq2seq(path: &str, model: &Seq2SeqModel) {
    use engine::seq2seq::Seq2SeqMemoryBudget;
    use engine::Activation;

    let c = &model.config;
    println!("Model: {path}");
    println!("  status          OK (file size matches config)");
    println!("  format          tiny-infer seq2seq v1 (Marian encoder-decoder), fp32 weights");
    println!("  d_model         {}", c.d_model);
    println!(
        "  encoder         {} layers, {} heads (head_dim {}), ffn {}",
        c.enc_layers,
        c.enc_heads,
        c.enc_head_dim(),
        c.enc_ffn
    );
    println!(
        "  decoder         {} layers, {} heads (head_dim {}), ffn {}",
        c.dec_layers,
        c.dec_heads,
        c.dec_head_dim(),
        c.dec_ffn
    );
    println!("  vocab_size      {}", c.vocab_size);
    println!("  max_src/max_tgt {} / {}", c.max_src, c.max_tgt);
    println!(
        "  pad/eos/bos     {} / {} / {}  (decoder starts at pad)",
        c.pad_id, c.eos_id, c.bos_id
    );
    println!(
        "  norm placement  {}",
        if c.norm_before {
            "pre-norm"
        } else {
            "post-norm (Marian default)"
        }
    );
    println!(
        "  activation      {}",
        match c.activation {
            Activation::Gelu => "gelu",
            Activation::Swish => "swish (SiLU)",
        }
    );
    println!(
        "  scale_embedding {}{}",
        c.scale_embedding,
        if c.scale_embedding {
            " (embeddings × √d_model)"
        } else {
            ""
        }
    );
    println!(
        "  weights         {} ({})",
        human_bytes(model.weight_bytes()),
        commas(model.weight_bytes())
    );

    let budget = Seq2SeqMemoryBudget::for_config(c, c.max_src, c.max_tgt);
    println!(
        "\nMemory budget (working arena at max_src={} / max_tgt={}):",
        c.max_src, c.max_tgt
    );
    println!(
        "  encoder buffers {:>10}  ({} f32)",
        human_bytes(budget.encoder_bytes()),
        commas(budget.encoder_floats)
    );
    println!(
        "  cross-KV cache  {:>10}  ({} f32)",
        human_bytes(budget.cross_kv_bytes()),
        commas(budget.cross_kv_floats)
    );
    println!(
        "  self-KV cache   {:>10}  ({} f32)",
        human_bytes(budget.self_kv_bytes()),
        commas(budget.self_kv_floats)
    );
    println!(
        "  step scratch    {:>10}  ({} f32)",
        human_bytes(budget.step_bytes()),
        commas(budget.step_floats)
    );
    println!(
        "  arena total     {:>10}  ({} f32)",
        human_bytes(budget.total_bytes()),
        commas(budget.total_floats())
    );
    println!(
        "  weights (disk)  {:>10}",
        human_bytes(model.weight_bytes())
    );
    println!(
        "  peak RAM        {:>10}  (weights + arena)",
        human_bytes(model.weight_bytes() + budget.total_bytes())
    );
}

/// Human-readable name of a checkpoint format, for the report.
fn format_label(f: ModelFormat) -> String {
    match f {
        ModelFormat::Legacy => "legacy (v0), fp32 weights".into(),
        ModelFormat::V1 => "v1, fp32 weights".into(),
        ModelFormat::V2 { group_size } => {
            format!("v2, int8 weights (group_size {group_size})")
        }
    }
}

/// Pick the matmul kernel from the flags, with a graceful fallback when `--dotprod`
/// is requested on a CPU without a hardware int8 dot-product instruction.
///
/// `--scalar` wins outright (reference path). `--dotprod` selects the int8 dot-product
/// kernel only when the CPU supports it — x86 AVX-512 VNNI or ARM NEON `sdot` (the engine
/// still falls back to SIMD internally for the fp32 path and x86 group sizes VNNI can't
/// take); otherwise it warns and uses SIMD. The default is SIMD.
fn select_kernel(scalar: bool, dotprod: bool) -> Kernel {
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
fn check_group_size(c: &Config, gs: usize) -> Result<(), HostError> {
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
fn report_quantization(data: &[i8], scales: &[f32], fp32_weight_bytes: usize, gs: usize) {
    let int8_bytes = data.len() + scales.len() * 4;
    eprintln!(
        "[quantized to int8 (group_size {gs}): {} weights resident \
         ({} int8 + {} scales); freed the {} fp32 checkpoint]",
        human_bytes(int8_bytes),
        human_bytes(data.len()),
        human_bytes(scales.len() * 4),
        human_bytes(fp32_weight_bytes),
    );
}

/// Render a byte count as a human-friendly KiB/MiB/GiB string.
fn human_bytes(n: usize) -> String {
    const UNITS: [&str; 4] = ["B", "KiB", "MiB", "GiB"];
    let mut v = n as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{n} B")
    } else {
        format!("{v:.2} {}", UNITS[u])
    }
}

/// Group an integer with thousands separators (e.g. `884736` → `884,736`).
fn commas(n: usize) -> String {
    let s = n.to_string();
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, &b) in bytes.iter().enumerate() {
        if i > 0 && (bytes.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(b as char);
    }
    out
}

/// Parsed command-line arguments.
struct Args {
    model: Option<String>,
    tokenizer: Option<String>,
    prompt: Option<String>,
    steps: Option<usize>,
    temperature: f32,
    topp: f32,
    seed: Option<u64>,
    quantize: bool,
    group_size: usize,
    scalar: bool,
    dotprod: bool,
    convert: Option<String>,
    to: Option<Target>,
    help: bool,
}

impl Args {
    fn parse(args: impl Iterator<Item = String>) -> Result<Args, HostError> {
        let mut model = None;
        let mut tokenizer = None;
        let mut prompt = None;
        let mut steps = None;
        let mut temperature = 0.0;
        let mut topp = 0.0;
        let mut seed = None;
        let mut quantize = false;
        let mut group_size = 32usize;
        let mut scalar = false;
        let mut dotprod = false;
        let mut convert = None;
        let mut to = None;
        let mut help = false;
        let mut positionals = Vec::new();

        let mut it = args.peekable();
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "-h" | "--help" => help = true,
                "-m" | "--model" => model = Some(expect_value(&mut it, &arg)?),
                "-t" | "--tokenizer" => tokenizer = Some(expect_value(&mut it, &arg)?),
                "-p" | "--prompt" => prompt = Some(expect_value(&mut it, &arg)?),
                "-n" | "--steps" => steps = Some(parse_usize(&expect_value(&mut it, &arg)?, &arg)?),
                "--temperature" => {
                    temperature = parse_f32(&expect_value(&mut it, &arg)?, &arg)?;
                }
                "--topp" => topp = parse_f32(&expect_value(&mut it, &arg)?, &arg)?,
                "--seed" => seed = Some(parse_u64(&expect_value(&mut it, &arg)?, &arg)?),
                "-q" | "--quantize" => quantize = true,
                "--scalar" => scalar = true,
                "--dotprod" => dotprod = true,
                "--group-size" => group_size = parse_usize(&expect_value(&mut it, &arg)?, &arg)?,
                "--convert" => convert = Some(expect_value(&mut it, &arg)?),
                "--to" => to = Some(parse_target(&expect_value(&mut it, &arg)?)?),
                s if s.starts_with("--model=") => model = Some(after_eq(s)),
                s if s.starts_with("--tokenizer=") => tokenizer = Some(after_eq(s)),
                s if s.starts_with("--prompt=") => prompt = Some(after_eq(s)),
                s if s.starts_with("--steps=") => {
                    steps = Some(parse_usize(&after_eq(s), "--steps")?)
                }
                s if s.starts_with("--temperature=") => {
                    temperature = parse_f32(&after_eq(s), "--temperature")?;
                }
                s if s.starts_with("--topp=") => topp = parse_f32(&after_eq(s), "--topp")?,
                s if s.starts_with("--seed=") => seed = Some(parse_u64(&after_eq(s), "--seed")?),
                s if s.starts_with("--group-size=") => {
                    group_size = parse_usize(&after_eq(s), "--group-size")?
                }
                s if s.starts_with("--convert=") => convert = Some(after_eq(s)),
                s if s.starts_with("--to=") => to = Some(parse_target(&after_eq(s))?),
                s if s.starts_with('-') && s != "-" => {
                    return Err(HostError::Usage(format!("unknown option `{s}`")));
                }
                _ => positionals.push(arg),
            }
        }

        // Positionals fill model then tokenizer, but never override explicit flags.
        let mut pos = positionals.into_iter();
        if model.is_none() {
            model = pos.next();
        }
        if tokenizer.is_none() {
            tokenizer = pos.next();
        }
        if let Some(extra) = pos.next() {
            return Err(HostError::Usage(format!("unexpected argument `{extra}`")));
        }

        Ok(Args {
            model,
            tokenizer,
            prompt,
            steps,
            temperature,
            topp,
            seed,
            quantize,
            group_size,
            scalar,
            dotprod,
            convert,
            to,
            help,
        })
    }
}

/// Parse a `--to` conversion target.
fn parse_target(s: &str) -> Result<Target, HostError> {
    match s {
        "v1" => Ok(Target::V1),
        "v2" => Ok(Target::V2),
        _ => Err(HostError::Usage(format!(
            "`--to` expects `v1` or `v2`, got `{s}`"
        ))),
    }
}

fn expect_value(
    it: &mut std::iter::Peekable<impl Iterator<Item = String>>,
    flag: &str,
) -> Result<String, HostError> {
    it.next()
        .ok_or_else(|| HostError::Usage(format!("`{flag}` expects a value")))
}

fn after_eq(s: &str) -> String {
    s.split_once('=').map(|x| x.1).unwrap_or("").to_string()
}

fn parse_usize(s: &str, flag: &str) -> Result<usize, HostError> {
    s.parse::<usize>().map_err(|_| {
        HostError::Usage(format!(
            "`{flag}` expects a non-negative integer, got `{s}`"
        ))
    })
}

fn parse_u64(s: &str, flag: &str) -> Result<u64, HostError> {
    s.parse::<u64>().map_err(|_| {
        HostError::Usage(format!(
            "`{flag}` expects a non-negative integer, got `{s}`"
        ))
    })
}

fn parse_f32(s: &str, flag: &str) -> Result<f32, HostError> {
    s.parse::<f32>()
        .map_err(|_| HostError::Usage(format!("`{flag}` expects a number, got `{s}`")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Args, HostError> {
        Args::parse(args.iter().map(|s| s.to_string()))
    }

    #[test]
    fn positional_model_and_tokenizer() {
        let a = parse(&["model.bin", "tok.bin"]).unwrap();
        assert_eq!(a.model.as_deref(), Some("model.bin"));
        assert_eq!(a.tokenizer.as_deref(), Some("tok.bin"));
    }

    #[test]
    fn flags_with_values() {
        let a = parse(&["-m", "model.bin", "--tokenizer=tok.bin"]).unwrap();
        assert_eq!(a.model.as_deref(), Some("model.bin"));
        assert_eq!(a.tokenizer.as_deref(), Some("tok.bin"));
    }

    #[test]
    fn help_flag() {
        assert!(parse(&["--help"]).unwrap().help);
    }

    #[test]
    fn missing_flag_value_is_usage_error() {
        assert!(matches!(parse(&["-m"]), Err(HostError::Usage(_))));
    }

    #[test]
    fn unknown_flag_is_usage_error() {
        assert!(matches!(parse(&["--frobnicate"]), Err(HostError::Usage(_))));
    }

    #[test]
    fn too_many_positionals_is_error() {
        assert!(matches!(
            parse(&["a.bin", "b.bin", "c.bin"]),
            Err(HostError::Usage(_))
        ));
    }

    #[test]
    fn generation_flags_parse() {
        let a = parse(&["-m", "m.bin", "-p", "Once upon a time", "-n", "64"]).unwrap();
        assert_eq!(a.prompt.as_deref(), Some("Once upon a time"));
        assert_eq!(a.steps, Some(64));
        assert_eq!(a.temperature, 0.0);
        assert_eq!(a.seed, None);

        let b = parse(&[
            "m.bin",
            "--prompt=hi",
            "--temperature=0.9",
            "--topp=0.8",
            "--seed=42",
        ])
        .unwrap();
        assert_eq!(b.prompt.as_deref(), Some("hi"));
        assert_eq!(b.temperature, 0.9);
        assert_eq!(b.topp, 0.8);
        assert_eq!(b.seed, Some(42));
    }

    #[test]
    fn quantization_flags_parse() {
        // Off by default, with a group size that divides the stories15M dims.
        let def = parse(&["m.bin"]).unwrap();
        assert!(!def.quantize);
        assert_eq!(def.group_size, 32);
        assert!(!def.scalar); // SIMD kernels by default
        assert!(!def.dotprod);

        let a = parse(&["m.bin", "-q", "--group-size", "64", "--scalar"]).unwrap();
        assert!(a.quantize);
        assert_eq!(a.group_size, 64);
        assert!(a.scalar);

        let d = parse(&["m.bin", "-q", "--dotprod"]).unwrap();
        assert!(d.dotprod);
        assert!(!d.scalar);

        let b = parse(&["m.bin", "--quantize", "--group-size=96"]).unwrap();
        assert!(b.quantize);
        assert_eq!(b.group_size, 96);
    }

    #[test]
    fn convert_flags_parse() {
        let def = parse(&["m.bin"]).unwrap();
        assert!(def.convert.is_none());
        assert!(def.to.is_none());

        let a = parse(&["m.bin", "--convert", "out.bin", "--to", "v1"]).unwrap();
        assert_eq!(a.convert.as_deref(), Some("out.bin"));
        assert_eq!(a.to, Some(Target::V1));

        let b = parse(&["m.bin", "--convert=out.bin", "--to=v2"]).unwrap();
        assert_eq!(b.convert.as_deref(), Some("out.bin"));
        assert_eq!(b.to, Some(Target::V2));

        // Anything but v1/v2 is a usage error.
        assert!(matches!(
            parse(&["m.bin", "--convert", "o.bin", "--to", "v3"]),
            Err(HostError::Usage(_))
        ));
    }

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
    fn bad_steps_value_is_usage_error() {
        assert!(matches!(
            parse(&["m.bin", "-n", "lots"]),
            Err(HostError::Usage(_))
        ));
        assert!(matches!(
            parse(&["m.bin", "--temperature=warm"]),
            Err(HostError::Usage(_))
        ));
        assert!(matches!(
            parse(&["m.bin", "--seed=soon"]),
            Err(HostError::Usage(_))
        ));
        assert!(matches!(
            parse(&["m.bin", "--topp=most"]),
            Err(HostError::Usage(_))
        ));
    }

    #[test]
    fn sampler_temperature_zero_is_greedy() {
        let mut s = Sampler::new(4, 0.0, 0.0, Some(1));
        assert_eq!(s.sample(&[0.1, 9.0, 0.2, 0.3]), 1);
    }

    #[test]
    fn sampler_tiny_topp_collapses_to_top_token() {
        // A near-zero topp shrinks the nucleus to the single most-probable token,
        // so the draw is deterministic regardless of temperature or seed.
        let logits = [1.0f32, 5.0, 2.0, 0.5];
        let mut s = Sampler::new(4, 1.0, 1e-6, Some(123));
        for _ in 0..8 {
            assert_eq!(s.sample(&logits), 1);
        }
    }

    #[test]
    fn sampler_is_reproducible_with_a_seed() {
        let logits = [1.0f32, 2.0, 0.5, 1.5, 3.0];
        let draws = |seed| {
            let mut s = Sampler::new(5, 1.0, 0.9, Some(seed));
            (0..16).map(|_| s.sample(&logits)).collect::<Vec<_>>()
        };
        assert_eq!(draws(7), draws(7));
    }

    #[test]
    fn human_bytes_scales() {
        assert_eq!(human_bytes(512), "512 B");
        assert_eq!(human_bytes(2048), "2.00 KiB");
        assert_eq!(human_bytes(3 * 1024 * 1024), "3.00 MiB");
    }

    #[test]
    fn commas_groups_thousands() {
        assert_eq!(commas(884_736), "884,736");
        assert_eq!(commas(100), "100");
        assert_eq!(commas(1_000), "1,000");
    }
}
