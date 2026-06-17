//! Report mode for the seq2seq path: print the Marian config and the four-group
//! memory budget.

use seq2seq::quantize::{kept_floats, quantized_scale_count, quantized_weight_count};
use seq2seq::{Activation, Config, MemoryBudget};
use humansize::{format_size, BINARY};
use thousands::Separable;

use crate::seq2seq::loader::Seq2SeqModel;

pub(crate) fn report_seq2seq(path: &str, model: &Seq2SeqModel) {
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
        format_size(model.weight_bytes(), BINARY),
        model.weight_bytes().separate_with_commas()
    );

    let budget = MemoryBudget::for_config(c, c.max_src, c.max_tgt);
    println!(
        "\nMemory budget (working arena at max_src={} / max_tgt={}):",
        c.max_src, c.max_tgt
    );
    println!(
        "  encoder buffers {:>10}  ({} f32)",
        format_size(budget.encoder_bytes(), BINARY),
        budget.encoder_floats.separate_with_commas()
    );
    println!(
        "  cross-KV cache  {:>10}  ({} f32)",
        format_size(budget.cross_kv_bytes(), BINARY),
        budget.cross_kv_floats.separate_with_commas()
    );
    println!(
        "  self-KV cache   {:>10}  ({} f32)",
        format_size(budget.self_kv_bytes(), BINARY),
        budget.self_kv_floats.separate_with_commas()
    );
    println!(
        "  step scratch    {:>10}  ({} f32)",
        format_size(budget.step_bytes(), BINARY),
        budget.step_floats.separate_with_commas()
    );
    println!(
        "  arena total     {:>10}  ({} f32)",
        format_size(budget.total_bytes(), BINARY),
        budget.total_floats().separate_with_commas()
    );
    println!(
        "  weights (disk)  {:>10}",
        format_size(model.weight_bytes(), BINARY)
    );

    // Resident weight memory and peak RAM, fp32 vs int8 — so the --quantize tradeoff is
    // visible without re-running. The 17 matmul matrices (the tied embedding included)
    // quantize to one byte each plus a small f32 scale per group; the biases, LayerNorms,
    // and final_logits_bias stay fp32.
    let arena = budget.total_bytes();
    let fp32_w = model.weight_bytes();
    match quant_group_size(c) {
        Some(gs) => {
            let int8_w =
                quantized_weight_count(c) + (quantized_scale_count(c, gs) + kept_floats(c)) * 4;
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
            println!(
                "  peak RAM       {:>10}  (weights + arena; not int8-quantizable)",
                format_size(fp32_w + arena, BINARY)
            );
        }
    }
}

/// A representative int8 group size for the report: the largest of a few common sizes that
/// divides `d_model`, `enc_ffn`, and `dec_ffn` (the quantization constraint), or `None` if
/// none do.
fn quant_group_size(c: &Config) -> Option<usize> {
    [64, 32, 16, 8]
        .into_iter()
        .find(|&gs| [c.d_model, c.enc_ffn, c.dec_ffn].iter().all(|d| d % gs == 0))
}
