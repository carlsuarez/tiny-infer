//! End-to-end CLI tests for the seq2seq (Marian) checkpoint format.
//!
//! A synthetic tiny checkpoint (valid header + zero weights) exercises the
//! loader and report without any fixture. The real-model assertions run against
//! `models/opus-mt-en-fr/model.bin` when `scripts/export_marian.py` has been run;
//! otherwise they skip themselves, like `tests/cli.rs` does for stories15M.

use std::path::PathBuf;
use std::process::Command;

use engine::seq2seq::config::{SEQ2SEQ_HEADER_BYTES, SEQ2SEQ_VERSION};
use engine::seq2seq::seq2seq_weight_floats;
use engine::{Activation, Seq2SeqConfig};

const BIN: &str = env!("CARGO_BIN_EXE_tiny-infer");

/// The tiny config used by the engine's own seq2seq unit tests (482 weight
/// floats, hand-counted there).
fn tiny_config() -> Seq2SeqConfig {
    Seq2SeqConfig {
        d_model: 4,
        enc_layers: 1,
        dec_layers: 1,
        enc_heads: 2,
        dec_heads: 2,
        enc_ffn: 8,
        dec_ffn: 8,
        vocab_size: 10,
        max_src: 6,
        max_tgt: 5,
        pad_id: 9,
        eos_id: 0,
        bos_id: 0,
        norm_before: false,
        activation: Activation::Swish,
        scale_embedding: true,
    }
}

/// Serialize a config into the 256-byte on-disk header.
fn header_bytes(c: &Seq2SeqConfig) -> Vec<u8> {
    let mut out = Vec::with_capacity(SEQ2SEQ_HEADER_BYTES);
    out.extend_from_slice(b"tis2");
    out.extend_from_slice(&SEQ2SEQ_VERSION.to_le_bytes());
    for v in [
        c.d_model,
        c.enc_layers,
        c.dec_layers,
        c.enc_heads,
        c.dec_heads,
        c.enc_ffn,
        c.dec_ffn,
        c.vocab_size,
        c.max_src,
        c.max_tgt,
        c.pad_id,
        c.eos_id,
        c.bos_id,
    ] {
        out.extend_from_slice(&(v as i32).to_le_bytes());
    }
    out.push(c.norm_before as u8);
    out.push(match c.activation {
        Activation::Gelu => 0,
        Activation::Swish => 1,
    });
    out.push(c.scale_embedding as u8);
    out.resize(SEQ2SEQ_HEADER_BYTES, 0);
    out
}

/// Write a complete synthetic checkpoint (header + zeroed fp32 weights).
fn write_synthetic(name: &str) -> PathBuf {
    let c = tiny_config();
    let mut bytes = header_bytes(&c);
    bytes.extend(std::iter::repeat_n(0u8, seq2seq_weight_floats(&c) * 4));
    let path = std::env::temp_dir().join(name);
    std::fs::write(&path, &bytes).unwrap();
    path
}

#[test]
fn reports_synthetic_seq2seq_checkpoint() {
    let path = write_synthetic("tiny_infer_seq2seq_ok.bin");
    let out = Command::new(BIN).arg(&path).output().unwrap();
    let _ = std::fs::remove_file(&path);

    assert!(
        out.status.success(),
        "exit {:?}\nstderr: {}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("seq2seq v1"), "stdout:\n{stdout}");
    assert!(stdout.contains("d_model         4"), "stdout:\n{stdout}");
    assert!(
        stdout.contains("encoder         1 layers, 2 heads (head_dim 2), ffn 8"),
        "stdout:\n{stdout}"
    );
    assert!(stdout.contains("vocab_size      10"), "stdout:\n{stdout}");
    assert!(stdout.contains("swish"), "stdout:\n{stdout}");
    assert!(stdout.contains("cross-KV cache"), "stdout:\n{stdout}");
    assert!(stdout.contains("arena total"), "stdout:\n{stdout}");
}

#[test]
fn seq2seq_prompt_is_rejected_cleanly() {
    let path = write_synthetic("tiny_infer_seq2seq_prompt.bin");
    let out = Command::new(BIN)
        .arg(&path)
        .args(["--prompt", "Tom went to the park"])
        .output()
        .unwrap();
    let _ = std::fs::remove_file(&path);

    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("report mode only"), "stderr:\n{stderr}");
    assert!(!stderr.contains("panicked"), "stderr:\n{stderr}");
}

#[test]
fn truncated_seq2seq_reports_size_mismatch() {
    // A valid header with no weight payload must be rejected as a size
    // mismatch, not crash or report success.
    let path = std::env::temp_dir().join("tiny_infer_seq2seq_truncated.bin");
    std::fs::write(&path, header_bytes(&tiny_config())).unwrap();

    let out = Command::new(BIN).arg(&path).output().unwrap();
    let _ = std::fs::remove_file(&path);
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("size mismatch"), "stderr:\n{stderr}");
}

#[test]
fn reports_real_opus_mt_checkpoint() {
    // Produced by: python scripts/export_marian.py Helsinki-NLP/opus-mt-en-fr \
    //                -o models/opus-mt-en-fr
    let model = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("models")
        .join("opus-mt-en-fr")
        .join("model.bin");
    if !model.exists() {
        eprintln!("skipping: models/opus-mt-en-fr/model.bin not present");
        return;
    }

    let out = Command::new(BIN).arg(&model).output().unwrap();
    assert!(
        out.status.success(),
        "exit {:?}\nstderr: {}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("status          OK (file size matches config)"),
        "stdout:\n{stdout}"
    );
    // The published en-fr pair model: d_model 512, 6+6 layers, 8 heads.
    assert!(stdout.contains("d_model         512"), "stdout:\n{stdout}");
    assert!(
        stdout.contains("encoder         6 layers, 8 heads (head_dim 64), ffn 2048"),
        "stdout:\n{stdout}"
    );
}
