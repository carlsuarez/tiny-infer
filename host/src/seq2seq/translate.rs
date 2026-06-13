//! Greedy translation: tokenize the source, run the encoder + autoregressive
//! decoder, and detokenize the result.

use std::time::Instant;

use engine::seq2seq::memory::seq2seq_arena_floats;
use engine::seq2seq::quantize::max_proj_d_in;
use engine::seq2seq::{greedy_decode, Config, KeptWeights, ModelWeights, RunState};
use engine::{Arena, Kernel, QuantScratch};

use crate::error::HostError;
use crate::seq2seq::tokenizer::Tokenizer;

/// Translate `text` greedily and print the result to stdout.
///
/// Encodes the source with the Marian tokenizer, carves the working arena once, runs
/// [`greedy_decode`] (encoder → cross-KV → autoregressive decoder, stopping at eos or
/// `max_tgt`) over the prepared weights, then detokenizes the target ids. `mw` carries the
/// matmul weights (fp32 or int8) and `kept` the fp32 biases / LayerNorms; on the int8 path
/// this allocates the W8A8 activation scratch. A throughput line goes to stderr.
pub(crate) fn translate(
    c: &Config,
    mw: &ModelWeights,
    kept: &KeptWeights,
    tok: &Tokenizer,
    text: &str,
    kernel: Kernel,
) -> Result<(), HostError> {
    let src_ids = tok.encode(text);
    if src_ids.len() > c.max_src {
        return Err(HostError::Usage(format!(
            "input is {} tokens but the model's maximum source length is {}",
            src_ids.len(),
            c.max_src
        )));
    }

    // Size the arena for the source length and the full target capacity; greedy
    // decode stops early at eos, so most of the self-KV cache often goes unused.
    let tgt_cap = c.max_tgt;
    let mut arena_buf = vec![0.0f32; seq2seq_arena_floats(c, src_ids.len(), tgt_cap)];
    let mut arena = Arena::new(&mut arena_buf);
    let mut state = RunState::new(&mut arena, c, src_ids.len(), tgt_cap)
        .map_err(|e| HostError::engine("<arena>", e))?;

    // The int8 (W8A8) path needs a per-matmul activation scratch; allocate it only when
    // the weights are quantized so an fp32 run carries none of it.
    let scratch_len = max_proj_d_in(c);
    let mut qbuf = matches!(mw, ModelWeights::Q8(_)).then(|| {
        (
            vec![0i8; scratch_len],
            vec![0.0f32; scratch_len],
            vec![0.0f32; scratch_len],
        )
    });
    let mut scratch = match qbuf.as_mut() {
        Some((xq, sc, gs)) => Some(
            QuantScratch::new(xq, sc, gs, scratch_len)
                .map_err(|e| HostError::engine("<arena>", e))?,
        ),
        None => None,
    };

    let mut out_ids = vec![0usize; tgt_cap];
    let start = Instant::now();
    let n = greedy_decode(
        c,
        mw,
        kept,
        &mut state,
        &src_ids,
        &mut out_ids,
        kernel,
        &mut scratch,
    );
    let elapsed = start.elapsed().as_secs_f64();

    println!("{}", tok.decode(&out_ids[..n]));

    let rate = if elapsed > 0.0 {
        n as f64 / elapsed
    } else {
        f64::INFINITY
    };
    eprintln!(
        "[translated {} source → {} target tokens in {elapsed:.3}s ({rate:.1} tok/s)]",
        src_ids.len(),
        n,
    );
    Ok(())
}
