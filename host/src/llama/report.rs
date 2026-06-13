//! Report mode for the Llama path: print the model config, tokenizer metadata, and
//! the pre-computed memory budget.

use engine::llama::memory::MemoryBudget;
use engine::llama::quantize::{quantized_scale_count, quantized_weight_count};
use engine::llama::{Config, ModelFormat};
use humansize::{format_size, BINARY};
use thousands::Separable;

use crate::llama::loader::Model;
use crate::llama::tokenizer::Vocab;

pub(crate) fn report_model(path: &str, model: &Model) {
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
        format_size(model.weight_bytes(), BINARY),
        model.weight_bytes().separate_with_commas()
    );
}

pub(crate) fn report_tokenizer(path: &str, vocab: &Vocab) {
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

pub(crate) fn report_memory(model: &Model) {
    let budget = MemoryBudget::for_config(&model.config);
    println!("\nMemory budget (working arena, pre-allocated once):");
    println!(
        "  activations    {:>10}  ({} f32)",
        format_size(budget.activation_bytes(), BINARY),
        budget.activation_floats.separate_with_commas()
    );
    println!(
        "  KV cache       {:>10}  ({} f32)",
        format_size(budget.kv_cache_bytes(), BINARY),
        budget.kv_cache_floats.separate_with_commas()
    );
    println!(
        "  arena total    {:>10}  ({} f32)",
        format_size(budget.total_bytes(), BINARY),
        budget.total_floats().separate_with_commas()
    );
    println!(
        "  weights (disk) {:>10}",
        format_size(model.weight_bytes(), BINARY)
    );

    // Resident weight memory and peak RAM, fp32 vs int8 — so the quantization
    // tradeoff is visible without re-running with -q. Same tensor set both ways: the
    // matmul weights (+ the tied embedding) quantize to one byte each plus a small
    // f32 scale per group; the RMSNorm gains stay fp32.
    let c = &model.config;
    let arena = budget.total_bytes();
    let rms_floats = 2 * c.n_layers * c.dim + c.dim;
    let fp32_w = (quantized_weight_count(c) + rms_floats) * 4;
    match quant_group_size(c) {
        Some(gs) => {
            let int8_w = quantized_weight_count(c) + quantized_scale_count(c, gs) * 4 + rms_floats * 4;
            let ratio = fp32_w as f64 / int8_w as f64;
            println!("\nWeights + peak RAM (fp32 vs int8, group_size {gs}):");
            println!("                      fp32         int8");
            println!(
                "  weights        {:>11}  {:>11}   ({ratio:.1}× smaller)",
                format_size(fp32_w, BINARY),
                format_size(int8_w, BINARY),
            );
            println!(
                "  peak RAM       {:>11}  {:>11}   (weights + arena)",
                format_size(fp32_w + arena, BINARY),
                format_size(int8_w + arena, BINARY),
            );
        }
        None => {
            // No common group size divides the dims, so int8 isn't applicable.
            println!(
                "  peak RAM       {:>10}  (weights + arena; not int8-quantizable)",
                format_size(fp32_w + arena, BINARY)
            );
        }
    }
}

/// A representative int8 group size for the report: the largest of a few common sizes
/// that divides `dim`, `hidden_dim`, and `kv_dim` (the quantization constraint), or
/// `None` if none do.
fn quant_group_size(c: &Config) -> Option<usize> {
    [64, 32, 16, 8]
        .into_iter()
        .find(|&gs| [c.dim, c.hidden_dim, c.kv_dim()].iter().all(|d| d % gs == 0))
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
