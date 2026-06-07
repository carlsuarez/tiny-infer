//! End-to-end generation tests.
//!
//! These run the built binary against the real `stories15M.bin` / `tokenizer.bin`
//! fixtures and skip themselves when those files are absent (so a clean checkout
//! still passes). Greedy decoding (temperature 0) is fully deterministic, so its
//! output is pinned to a golden string; temperature sampling is checked for
//! reproducibility under a fixed `--seed`.
//!
//! The golden text below is byte-for-byte identical to Karpathy's `run.c` greedy
//! output on the same checkpoint and prompt (verified 2026-06-07 with
//! `run stories15M.bin -z tokenizer.bin -t 0 -n 40 -i "Tom went to the park"`).
//! So this both pins the reference parity and guards against regressions.

use std::path::PathBuf;
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

/// Greedy continuation of "Tom went to the park" for 40 steps on stories15M.
// The story ends with a newline token from the model; the CLI then appends its own
// trailing newline, hence the final "\n\n".
const GOLDEN: &str = "Tom went to the park with his mom. He saw a big slide and wanted \
to go on it. He ran to the slide and climbed up the ladder. He was very happy.\n\n";

#[test]
fn greedy_generation_is_deterministic_and_matches_golden() {
    let (Some(model), Some(tok)) = (fixture("stories15M.bin"), fixture("tokenizer.bin")) else {
        eprintln!("skipping: stories15M.bin or tokenizer.bin not present");
        return;
    };

    let run = || {
        let out = Command::new(BIN)
            .arg(&model)
            .arg(&tok)
            .args(["-p", "Tom went to the park", "-n", "40", "--temperature", "0"])
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "exit {:?}\nstderr: {}",
            out.status.code(),
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8(out.stdout).expect("output is valid UTF-8")
    };

    let first = run();
    assert_eq!(first, GOLDEN, "greedy output drifted from golden");
    // Determinism: a second run is byte-identical.
    assert_eq!(first, run(), "greedy output is not deterministic");
}

#[test]
fn generation_reports_throughput_on_stderr() {
    let (Some(model), Some(tok)) = (fixture("stories15M.bin"), fixture("tokenizer.bin")) else {
        eprintln!("skipping: fixtures not present");
        return;
    };

    let out = Command::new(BIN)
        .arg(&model)
        .arg(&tok)
        .args(["-p", "Once", "-n", "16"])
        .output()
        .unwrap();
    assert!(out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("tok/s"), "stderr:\n{stderr}");
}

#[test]
fn prompt_without_tokenizer_is_usage_error() {
    let Some(model) = fixture("stories15M.bin") else {
        eprintln!("skipping: stories15M.bin not present");
        return;
    };
    let out = Command::new(BIN)
        .arg(&model)
        .args(["-p", "hello"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("requires a tokenizer"), "stderr:\n{stderr}");
}

#[test]
fn sampled_generation_is_reproducible_under_a_fixed_seed() {
    let (Some(model), Some(tok)) = (fixture("stories15M.bin"), fixture("tokenizer.bin")) else {
        eprintln!("skipping: fixtures not present");
        return;
    };

    let run = |seed: &str| {
        let out = Command::new(BIN)
            .arg(&model)
            .arg(&tok)
            .args([
                "-p",
                "Once upon a time",
                "-n",
                "32",
                "--temperature",
                "0.9",
                "--seed",
                seed,
            ])
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "exit {:?}\nstderr: {}",
            out.status.code(),
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8(out.stdout).expect("output is valid UTF-8")
    };

    // Sampling is non-deterministic in general but fully reproducible for a fixed
    // seed, and it still produces non-empty text.
    let a = run("42");
    assert!(a.trim().len() > "Once upon a time".len(), "no text generated");
    assert_eq!(a, run("42"), "same seed must reproduce the same output");
}

#[test]
fn topp_generation_is_reproducible_under_a_fixed_seed() {
    let (Some(model), Some(tok)) = (fixture("stories15M.bin"), fixture("tokenizer.bin")) else {
        eprintln!("skipping: fixtures not present");
        return;
    };

    let run = || {
        let out = Command::new(BIN)
            .arg(&model)
            .arg(&tok)
            .args([
                "-p",
                "Once upon a time",
                "-n",
                "32",
                "--temperature",
                "1.0",
                "--topp",
                "0.9",
                "--seed",
                "42",
            ])
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "exit {:?}\nstderr: {}",
            out.status.code(),
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8(out.stdout).expect("output is valid UTF-8")
    };

    // Nucleus sampling is also reproducible under a fixed seed.
    let a = run();
    assert!(a.trim().len() > "Once upon a time".len(), "no text generated");
    assert_eq!(a, run(), "same seed must reproduce the same top-p output");
}
