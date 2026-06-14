//! The seq2seq entry point: generate an output sequence from a prompt, or report on
//! the model.

use std::path::{Path, PathBuf};

use engine::seq2seq::quantize::{
    kept_floats, pack_kept, quantize_weights, quantized_scale_count, quantized_weight_count,
    KeptWeights, QuantizedWeights,
};
use engine::seq2seq::{Config, ModelWeights};
use engine::Sampler;

use crate::args::Args;
use crate::error::HostError;
use crate::llama::generate::{report_quantization, resolve_seed, select_kernel};
use crate::seq2seq::loader::Seq2SeqModel;
use crate::seq2seq::report::report_seq2seq;
use crate::seq2seq::tokenizer::Tokenizer;
use crate::seq2seq::generate::generate;

/// Run a seq2seq (Marian) checkpoint: generate an output sequence from `--prompt`
/// (optionally quantized to int8), or report on the model.
pub fn run(path: &str, args: &Args) -> Result<(), HostError> {
    // There is no on-disk int8 seq2seq format yet, so conversion is undefined here.
    if args.convert.is_some() {
        return Err(HostError::Usage(
            "seq2seq (Marian) checkpoints do not support --convert (no on-disk int8 format yet)"
                .into(),
        ));
    }

    let model = Seq2SeqModel::load(path)?;
    let config = model.config; // Copy, so generation can outlive the file buffer.

    if let Some(text) = args.prompt.as_deref() {
        let tok_path = resolve_tokenizer(path, args)?;
        let tok = Tokenizer::load(&tok_path)?;
        // `--scalar` forces the reference kernels; `--dotprod` selects the int8 hardware
        // kernel when generating quantized (and the CPU supports it); else `core::simd`.
        let kernel = select_kernel(args.scalar, args.dotprod);
        // The shared sampler picks each token: greedy at temperature 0 (the default, and the
        // quality path for seq2seq), else temperature/top-p as on the Llama path.
        let sampler = Sampler::new(args.temperature, args.topp, resolve_seed(args.seed));

        // int8 path: quantize the matmul matrices at load, pack the kept fp32 tensors,
        // free the fp32 checkpoint, then generate over the int8 weights.
        if args.quantize {
            let gs = args.group_size;
            check_group_size(&config, gs)?;
            let mut data = vec![0i8; quantized_weight_count(&config)];
            let mut scales = vec![0.0f32; quantized_scale_count(&config, gs)];
            let mut kept_buf = vec![0.0f32; kept_floats(&config)];
            {
                let weights = model.weights()?;
                quantize_weights(&weights, &mut data, &mut scales, gs, &config);
                pack_kept(&weights, &mut kept_buf, &config);
            }
            report_quantization(&data, &scales, model.weight_bytes(), gs);
            drop(model); // free the fp32 checkpoint (the bulk of the memory)
            let qw = QuantizedWeights::new(&data, &scales, gs, &config)
                .map_err(|e| HostError::engine(path, e))?;
            let kept =
                KeptWeights::carve(&kept_buf, &config).map_err(|e| HostError::engine(path, e))?;
            return generate(&config, &ModelWeights::Q8(qw), &kept, &tok, text, kernel, sampler);
        }

        // fp32 path: weight views borrow straight from the loaded file.
        let weights = model.weights()?;
        let kept = KeptWeights::from_weights(&weights);
        return generate(
            &config,
            &ModelWeights::F32(weights),
            &kept,
            &tok,
            text,
            kernel,
            sampler,
        );
    }

    // Report mode: validate the tensor carve, then print the report. (`--quantize` only
    // affects generation; the report shows the fp32-vs-int8 comparison regardless.)
    let _ = model.weights()?;
    report_seq2seq(path, &model);
    Ok(())
}

/// Reject a quantization group size that does not evenly divide the seq2seq weight
/// dimensions. Groups never straddle a row, so the group size must divide every matmul's
/// input dimension — `d_model` (attention + `fc1` + lm_head) and the two ffn widths
/// (`fc2`).
fn check_group_size(c: &Config, gs: usize) -> Result<(), HostError> {
    if gs == 0 {
        return Err(HostError::Usage("--group-size must be at least 1".into()));
    }
    for (name, d) in [
        ("d_model", c.d_model),
        ("enc_ffn", c.enc_ffn),
        ("dec_ffn", c.dec_ffn),
    ] {
        if !d.is_multiple_of(gs) {
            return Err(HostError::Usage(format!(
                "--group-size {gs} must divide {name} ({d})"
            )));
        }
    }
    Ok(())
}

/// Locate the tokenizer artifact: an explicit `--tokenizer`, else `tokenizer.bin`
/// sitting next to the checkpoint (where `export_marian.py` writes it).
fn resolve_tokenizer(model_path: &str, args: &Args) -> Result<PathBuf, HostError> {
    if let Some(t) = args.tokenizer.as_deref() {
        return Ok(PathBuf::from(t));
    }
    let sibling = Path::new(model_path)
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("tokenizer.bin");
    if sibling.exists() {
        Ok(sibling)
    } else {
        Err(HostError::Usage(format!(
            "seq2seq generation needs a tokenizer; pass --tokenizer <path> or place \
             tokenizer.bin next to the model (looked for {})",
            sibling.display()
        )))
    }
}
