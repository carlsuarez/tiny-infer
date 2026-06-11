//! End-to-end checkpoint-format tests: convert the legacy fixture to v1/v2 and
//! prove the converted files generate exactly what the legacy file does.
//!
//! Like `generate.rs`, these run the built binary against the real
//! `stories15M.bin` / `tokenizer.bin` fixtures and skip themselves when those
//! files are absent. The parity gates are exact, not fuzzy:
//!
//! * **v1** is a lossless fp32 re-serialization, so its greedy output must be
//!   byte-identical to the legacy file's.
//! * **v2** is written with the engine's own `quantize` kernel, so loading it
//!   back must reproduce `--quantize` (same group size) bit-for-bit — identical
//!   greedy output again.

use std::path::{Path, PathBuf};
use std::process::Command;

const BIN: &str = env!("CARGO_BIN_EXE_tiny-infer");

fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("models")
}

fn fixture(name: &str) -> Option<PathBuf> {
    let p = models_dir().join(name);
    p.exists().then_some(p)
}

/// A per-test scratch directory under the OS temp dir, removed on drop.
/// (Tests run concurrently in one process, so each gets its own directory.)
struct Scratch(PathBuf);

impl Scratch {
    fn new(test: &str) -> Scratch {
        let dir = std::env::temp_dir().join(format!("tiny-infer-fmt-{}-{test}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }

    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Convert `model` to `target` ("v1"/"v2") at `out`, asserting success.
fn convert(model: &Path, out: &Path, target: &str) {
    let res = Command::new(BIN)
        .arg(model)
        .args(["--convert", out.to_str().unwrap(), "--to", target])
        .output()
        .unwrap();
    assert!(
        res.status.success(),
        "convert to {target} failed: {}",
        String::from_utf8_lossy(&res.stderr)
    );
    assert!(out.exists(), "convert succeeded but wrote no file");
}

/// Greedy generation (fully deterministic) on `model`, returning stdout.
fn greedy(model: &Path, tok: &Path, extra: &[&str]) -> String {
    let out = Command::new(BIN)
        .arg(model)
        .arg(tok)
        .args(["-p", "Tom went to the park", "-n", "40", "--temperature", "0"])
        .args(extra)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "exit {:?}\nstderr: {}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).expect("output is valid UTF-8")
}

#[test]
fn v1_conversion_generates_byte_identical_output() {
    let (Some(model), Some(tok)) = (fixture("stories15M.bin"), fixture("tokenizer.bin")) else {
        eprintln!("skipping: fixtures not present");
        return;
    };
    let scratch = Scratch::new("v1");
    let v1 = scratch.path("stories15M.v1.bin");
    convert(&model, &v1, "v1");

    // fp32 -> fp32 is lossless, so greedy output is byte-for-byte the same.
    assert_eq!(
        greedy(&model, &tok, &[]),
        greedy(&v1, &tok, &[]),
        "v1 output diverged from legacy"
    );
}

#[test]
fn v2_conversion_matches_load_time_quantization() {
    let (Some(model), Some(tok)) = (fixture("stories15M.bin"), fixture("tokenizer.bin")) else {
        eprintln!("skipping: fixtures not present");
        return;
    };
    let scratch = Scratch::new("v2");
    let v2 = scratch.path("stories15M.v2.bin");
    convert(&model, &v2, "v2"); // default group_size 32, same as --quantize

    // The converter quantizes with the engine's own kernel, so running the v2
    // file must reproduce `--quantize` on the legacy file exactly.
    assert_eq!(
        greedy(&model, &tok, &["--quantize"]),
        greedy(&v2, &tok, &[]),
        "v2 output diverged from load-time quantization"
    );
}

#[test]
fn report_shows_the_detected_format() {
    let Some(model) = fixture("stories15M.bin") else {
        eprintln!("skipping: stories15M.bin not present");
        return;
    };
    let scratch = Scratch::new("report");
    let v1 = scratch.path("stories15M.v1.bin");
    let v2 = scratch.path("stories15M.v2.bin");
    convert(&model, &v1, "v1");
    convert(&model, &v2, "v2");

    let report = |m: &Path| {
        let out = Command::new(BIN).arg(m).output().unwrap();
        assert!(
            out.status.success(),
            "report failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8(out.stdout).unwrap()
    };

    assert!(report(&model).contains("legacy (v0), fp32 weights"));
    assert!(report(&v1).contains("v1, fp32 weights"));
    assert!(report(&v2).contains("v2, int8 weights (group_size 32)"));
}

#[test]
fn converting_an_int8_checkpoint_is_rejected() {
    let Some(model) = fixture("stories15M.bin") else {
        eprintln!("skipping: stories15M.bin not present");
        return;
    };
    let scratch = Scratch::new("reject");
    let v2 = scratch.path("stories15M.v2.bin");
    convert(&model, &v2, "v2");

    // int8 -> fp32 would bake quantization loss into an ostensibly fp32 file.
    let out = Command::new(BIN)
        .arg(&v2)
        .args(["--convert", scratch.path("bad.bin").to_str().unwrap(), "--to", "v1"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("already int8"), "stderr:\n{stderr}");
}
