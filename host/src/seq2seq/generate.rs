//! Greedy / sampled seq2seq generation: tokenize the source, run the encoder +
//! autoregressive decoder (picking each token with the shared [`Sampler`]), and detokenize
//! the output. The encoder-decoder maps any input sequence to a generated output one —
//! translation, summarization, paraphrasing — depending on the loaded model.

use std::time::Instant;

use engine::seq2seq::memory::seq2seq_arena_floats;
use engine::seq2seq::quantize::max_proj_d_in;
use engine::seq2seq::{
    decode_step, encode, precompute_cross_kv, Config, KeptWeights, ModelWeights, RunState,
};
use engine::sample::ProbIndex;
use engine::{Arena, Kernel, QuantScratch, Sampler};

use crate::error::HostError;
use crate::seq2seq::tokenizer::Tokenizer;

/// Generate the output sequence for `text` and print it to stdout.
///
/// Encodes the source with the Marian tokenizer, carves the working arena once, then drives
/// the decode loop — encoder → cross-KV → per-step [`decode_step`] with the next token
/// chosen by `sampler` — stopping at eos or `max_tgt`, and detokenizes the target ids. At
/// `temperature == 0` the sampler is greedy (`argmax`), matching the engine's `greedy_decode`
/// and the parity gate; `--temperature`/`--topp` add diversity (rarely what you want for a
/// faithful sequence-to-sequence output, but offered for parity with the Llama path). `mw`
/// carries the matmul weights (fp32 or int8) and `kept` the fp32 biases / LayerNorms; on the
/// int8 path this allocates the W8A8 activation scratch. A throughput line goes to stderr.
pub(crate) fn generate(
    c: &Config,
    mw: &ModelWeights,
    kept: &KeptWeights,
    tok: &Tokenizer,
    text: &str,
    kernel: Kernel,
    mut sampler: Sampler,
) -> Result<(), HostError> {
    let src_ids = tok.encode(text);
    if src_ids.len() > c.max_src {
        return Err(HostError::Usage(format!(
            "input is {} tokens but the model's maximum source length is {}",
            src_ids.len(),
            c.max_src
        )));
    }

    // Size the arena for the source length and the full target capacity; decoding stops
    // early at eos, so most of the self-KV cache often goes unused.
    let tgt_cap = c.max_tgt;
    let mut arena_buf = vec![0.0f32; seq2seq_arena_floats(c, src_ids.len(), tgt_cap)];
    let mut arena = Arena::new(&mut arena_buf);
    let mut state = RunState::new(&mut arena, c, src_ids.len(), tgt_cap)
        .map_err(|e| HostError::engine("<arena>", e))?;

    // The int8 (W8A8) path needs a per-matmul activation scratch; allocate it only when the
    // weights are quantized so an fp32 run carries none of it.
    let scratch_len = max_proj_d_in(c);
    let mut qbuf = matches!(mw, ModelWeights::Q8(_)).then(|| {
        (
            vec![0i8; scratch_len],
            vec![0.0f32; scratch_len],
            vec![0.0f32; scratch_len],
        )
    });
    let mut qs = match qbuf.as_mut() {
        Some((xq, sc, gs)) => Some(
            QuantScratch::new(xq, sc, gs, scratch_len)
                .map_err(|e| HostError::engine("<arena>", e))?,
        ),
        None => None,
    };

    // Caller-owned nucleus scratch for top-p sampling (the engine sampler never allocates);
    // ignored at temperature 0 and for full sampling.
    let mut probindex = vec![ProbIndex::default(); c.vocab_size];

    let mut out_ids = vec![0usize; tgt_cap];
    let start = Instant::now();

    // Encode once, project the cross-attention KV once, then decode token by token. This is
    // the engine's `greedy_decode` loop, driven here so the next token comes from `sampler`
    // (greedy when temperature 0) rather than a hardwired argmax.
    encode(c, mw, kept, &mut state, &src_ids, kernel, &mut qs);
    precompute_cross_kv(c, mw, kept, &mut state, src_ids.len(), kernel, &mut qs);
    let mut token = c.pad_id; // Marian seeds the decoder with the pad token
    let mut n = 0;
    for (pos, slot) in out_ids.iter_mut().enumerate() {
        let logits = decode_step(c, mw, kept, &mut state, token, pos, src_ids.len(), kernel, &mut qs);
        let next = sampler.sample(logits, &mut probindex);
        if next == c.eos_id {
            break;
        }
        *slot = next;
        token = next;
        n += 1;
    }
    let elapsed = start.elapsed().as_secs_f64();

    println!("{}", tok.decode(&out_ids[..n]));

    let rate = if elapsed > 0.0 {
        n as f64 / elapsed
    } else {
        f64::INFINITY
    };
    eprintln!(
        "[generated {} output tokens from {} input tokens in {elapsed:.3}s ({rate:.1} tok/s)]",
        n,
        src_ids.len(),
    );
    Ok(())
}
