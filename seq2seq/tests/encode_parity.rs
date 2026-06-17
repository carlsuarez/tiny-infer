//! M11 correctness gate: the engine's seq2seq `encode` must reproduce Hugging
//! Face's `MarianEncoder(...).last_hidden_state` on a fixed tokenized input.
//!
//! The fixture is produced by `scripts/dump_encoder_ref.py` (which needs torch +
//! transformers) and stored next to the model under `models/opus-mt-en-fr/`. Both
//! files are git-ignored, so — like the host's stories15M tests — this test skips
//! itself when they are absent and runs the real comparison when they are present.
//!
//! Feeding the engine the *same* token ids HF used isolates the encoder math from
//! any tokenizer differences (the tokenizer is a separate, later milestone).

use std::path::PathBuf;

use seq2seq::config::SEQ2SEQ_HEADER_BYTES;
use seq2seq::memory::seq2seq_arena_floats;
use seq2seq::{encode, Config, KeptWeights, ModelWeights, RunState, Weights};
use engine::{Arena, Kernel};

/// Reinterpret a little-endian byte buffer as `f32` (the host's job in production;
/// done safely here, one element at a time, for the test).
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
fn encoder_output_matches_hugging_face() {
    let dir = model_dir();
    let model_path = dir.join("model.bin");
    let ref_path = dir.join("encoder_ref.bin");
    if !model_path.exists() || !ref_path.exists() {
        eprintln!(
            "skipping: {} / encoder_ref.bin not present \
             (run scripts/export_marian.py + scripts/dump_encoder_ref.py)",
            model_path.display()
        );
        return;
    }

    // --- reference fixture: src_len | d_model | token ids | expected output ---
    let fx = std::fs::read(&ref_path).unwrap();
    let src_len = read_u32(&fx, 0);
    let d = read_u32(&fx, 4);
    let mut off = 8;
    let tokens: Vec<usize> = (0..src_len).map(|i| read_u32(&fx, off + i * 4)).collect();
    off += src_len * 4;
    let expected = as_f32(&fx[off..off + src_len * d * 4]);
    assert_eq!(expected.len(), src_len * d);

    // --- load the checkpoint and run the engine encoder ---
    let model_bytes = std::fs::read(&model_path).unwrap();
    let floats = as_f32(&model_bytes);
    let config = Config::parse(&model_bytes).expect("parse seq2seq header");
    assert_eq!(config.d_model, d, "fixture/model d_model disagree");
    let weights = Weights::new(&floats[SEQ2SEQ_HEADER_BYTES / 4..], &config).unwrap();
    let kept = KeptWeights::from_weights(&weights);
    let mw = ModelWeights::F32(weights);

    let mut arena_buf = vec![0.0f32; seq2seq_arena_floats(&config, src_len, 0)];
    let mut arena = Arena::new(&mut arena_buf);
    let mut state = RunState::new(&mut arena, &config, src_len, 0).unwrap();

    let out = encode(&config, &mw, &kept, &mut state, &tokens, Kernel::Simd, &mut None);
    assert_eq!(out.len(), expected.len());

    // --- compare ---
    let max_abs = expected.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    let max_diff = out
        .iter()
        .zip(&expected)
        .fold(0.0f32, |m, (&a, &b)| m.max((a - b).abs()));
    eprintln!(
        "encoder parity: src_len={src_len}, d_model={d}, max|ref|={max_abs:.4}, \
         max_abs_diff={max_diff:.3e}, rel={:.2e}",
        max_diff / max_abs
    );

    // Observed ~1.4e-6 (pure fp32 reassociation: engine scalar matmul vs torch,
    // bounded by the per-layer LayerNorm). 1e-4 leaves ~70× headroom — enough for a
    // future SIMD-kernel swap — while a real op-order/scale/norm bug shows up as a
    // diff orders of magnitude larger (O(0.01)+).
    assert!(
        max_diff < 1e-4,
        "engine encoder drifted from HF: max_abs_diff={max_diff} (max|ref|={max_abs})"
    );
}
