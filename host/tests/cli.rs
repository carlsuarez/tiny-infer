//! End-to-end CLI tests.
//!
//! The metadata assertions run against the real `stories15M.bin` /
//! `tokenizer.bin` fixtures when they are present under `models/`; if they have
//! not been downloaded the data-dependent tests skip themselves so the suite
//! still passes on a clean checkout. The argument-handling tests always run.

use std::path::PathBuf;
use std::process::Command;

const BIN: &str = env!("CARGO_BIN_EXE_tiny-infer");

fn models_dir() -> PathBuf {
    // CARGO_MANIFEST_DIR is the `host` crate; the fixtures live at the repo root.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("models")
}

fn fixture(name: &str) -> Option<PathBuf> {
    let p = models_dir().join(name);
    p.exists().then_some(p)
}

#[test]
fn dumps_stories15m_metadata() {
    let (Some(model), Some(tok)) = (fixture("stories15M.bin"), fixture("tokenizer.bin")) else {
        eprintln!("skipping: models/stories15M.bin or tokenizer.bin not present");
        return;
    };

    let out = Command::new(BIN).arg(&model).arg(&tok).output().unwrap();
    assert!(
        out.status.success(),
        "exit {:?}\nstderr: {}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);

    // Config of the 15M TinyStories checkpoint.
    assert!(stdout.contains("dim            288"), "stdout:\n{stdout}");
    assert!(stdout.contains("n_layers       6"), "stdout:\n{stdout}");
    assert!(stdout.contains("vocab_size     32000"), "stdout:\n{stdout}");
    assert!(stdout.contains("seq_len        256"), "stdout:\n{stdout}");
    // Tokenizer and memory sections present.
    assert!(stdout.contains("tokens             32000"), "stdout:\n{stdout}");
    assert!(stdout.contains("KV cache"), "stdout:\n{stdout}");
    assert!(stdout.contains("arena total"), "stdout:\n{stdout}");
}

#[test]
fn missing_model_file_fails_cleanly() {
    let out = Command::new(BIN)
        .arg("/no/such/model.bin")
        .output()
        .unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.starts_with("error:"), "stderr:\n{stderr}");
    // No panic / backtrace leaked to the user.
    assert!(!stderr.contains("panicked"), "stderr:\n{stderr}");
}

#[test]
fn unknown_flag_fails_with_usage() {
    let out = Command::new(BIN).arg("--nope").output().unwrap();
    assert_eq!(out.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("USAGE"), "stderr:\n{stderr}");
}

#[test]
fn unrecognized_format_fails_cleanly() {
    // A file matching neither the seq2seq magic nor a plausible llama header (a
    // foreign 4-byte tag followed by a zeroed `hidden_dim`) routes to neither loader
    // and reports a clean "unrecognized" error instead of crashing.
    let dir = std::env::temp_dir();
    let path = dir.join("tiny_infer_unrecognized.bin");
    let mut bytes = b"ZZZZ".to_vec();
    bytes.resize(64, 0); // 64 bytes total, f32-aligned
    std::fs::write(&path, &bytes).unwrap();

    let out = Command::new(BIN).arg(&path).output().unwrap();
    let _ = std::fs::remove_file(&path);
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("unrecognized"), "stderr:\n{stderr}");
    assert!(!stderr.contains("panicked"), "stderr:\n{stderr}");
}

#[test]
fn truncated_checkpoint_reports_size_mismatch() {
    // A 28-byte header claiming the stories15M shape, with no weight data:
    // the loader must reject it as a size mismatch, not crash.
    let dir = std::env::temp_dir();
    let path = dir.join("tiny_infer_truncated.bin");
    let header: [i32; 7] = [288, 768, 6, 6, 6, 32000, 256];
    let mut bytes = Vec::new();
    for v in header {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(&path, &bytes).unwrap();

    let out = Command::new(BIN).arg(&path).output().unwrap();
    let _ = std::fs::remove_file(&path);
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("size mismatch"), "stderr:\n{stderr}");
}
