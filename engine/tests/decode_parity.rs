//! M12 correctness gate: the engine's seq2seq greedy decode must reproduce Hugging
//! Face's `generate(num_beams=1, do_sample=False)` token stream on a fixed source.
//!
//! The fixture is produced by `scripts/dump_decode_ref.py` (needs torch +
//! transformers) and stored next to the model under `models/opus-mt-en-fr/`. Both
//! files are git-ignored, so — like the encoder parity test — this skips itself when
//! they are absent and runs the real comparison when present.
//!
//! Feeding the engine the *same* source token ids HF used isolates the decoder math
//! (self-attention, cross-attention, the tied lm_head) from the tokenizer. Unlike the
//! encoder test's float tolerance, this gate is exact: greedy argmax token ids must
//! match HF id-for-id.

#![cfg(feature = "seq2seq")]

use std::path::PathBuf;

use engine::seq2seq::config::SEQ2SEQ_HEADER_BYTES;
use engine::seq2seq::memory::seq2seq_arena_floats;
use engine::seq2seq::{greedy_decode, Config, KeptWeights, ModelWeights, RunState, Weights};
use engine::{Arena, Kernel};

fn as_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_u32(bytes: &[u8], at: usize) -> usize {
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]) as usize
}

fn model_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("models")
        .join("opus-mt-en-fr")
}

#[test]
fn greedy_decode_matches_hugging_face() {
    let dir = model_dir();
    let model_path = dir.join("model.bin");
    let ref_path = dir.join("decode_ref.bin");
    if !model_path.exists() || !ref_path.exists() {
        eprintln!(
            "skipping: {} / decode_ref.bin not present \
             (run scripts/export_marian.py + scripts/dump_decode_ref.py)",
            model_path.display()
        );
        return;
    }

    // --- reference fixture: n_src | n_gen | src ids | HF generated ids ---
    let fx = std::fs::read(&ref_path).unwrap();
    let n_src = read_u32(&fx, 0);
    let n_gen = read_u32(&fx, 4);
    let mut off = 8;
    let src_ids: Vec<usize> = (0..n_src).map(|i| read_u32(&fx, off + i * 4)).collect();
    off += n_src * 4;
    let gen_ids: Vec<usize> = (0..n_gen).map(|i| read_u32(&fx, off + i * 4)).collect();

    // --- load the checkpoint ---
    let model_bytes = std::fs::read(&model_path).unwrap();
    let floats = as_f32(&model_bytes);
    let config = Config::parse(&model_bytes).expect("parse seq2seq header");
    let weights = Weights::new(&floats[SEQ2SEQ_HEADER_BYTES / 4..], &config).unwrap();
    let kept = KeptWeights::from_weights(&weights);
    let mw = ModelWeights::F32(weights);

    // HF's stream is `[decoder_start, t1, …, tk, eos]`; the engine emits `[t1, …, tk]`
    // (it starts from the decoder-start token internally and stops before writing eos).
    assert_eq!(gen_ids[0], config.pad_id, "HF stream should start at decoder_start");
    assert_eq!(*gen_ids.last().unwrap(), config.eos_id, "HF stream should end at eos");
    let expected = &gen_ids[1..gen_ids.len() - 1];

    // --- run the engine's greedy decode over the same source ids ---
    let tgt_cap = expected.len().max(1);
    let mut arena_buf = vec![0.0f32; seq2seq_arena_floats(&config, n_src, tgt_cap)];
    let mut arena = Arena::new(&mut arena_buf);
    let mut state = RunState::new(&mut arena, &config, n_src, tgt_cap).unwrap();

    let mut out = vec![0usize; tgt_cap];
    let n = greedy_decode(
        &config, &mw, &kept, &mut state, &src_ids, &mut out, Kernel::Simd, &mut None,
    );
    let got = &out[..n];

    eprintln!("decode parity: src={src_ids:?}\n  engine={got:?}\n  hf    ={expected:?}");
    assert_eq!(got, expected, "engine greedy decode diverged from HF");
}
