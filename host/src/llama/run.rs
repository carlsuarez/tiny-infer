//! The Llama-path entry point: dispatch a llama2.c checkpoint to conversion,
//! generation, or report mode based on the parsed arguments.

use engine::llama::quantize::{quantize_weights, quantized_scale_count, quantized_weight_count};
use engine::llama::{ModelFormat, ModelWeights, QuantizedWeights};
use engine::Sampler;

use crate::args::Args;
use crate::error::HostError;
use crate::llama::convert::{self, Target};
use crate::llama::generate::{
    check_group_size, generate, report_quantization, resolve_seed, select_kernel,
};
use crate::llama::loader::Model;
use crate::llama::report::{report_memory, report_model, report_tokenizer};
use crate::llama::tokenizer::Vocab;

/// Run a llama2.c checkpoint: convert it, generate from it, or report on it.
pub fn run(model_path: &str, args: &Args) -> Result<(), HostError> {
    let model = Model::load(model_path)?;
    let config = model.config; // Copy, so generation can outlive the file buffer.

    // Conversion mode: write the requested format and exit.
    if let Some(out_path) = args.convert.as_deref() {
        let target = args.to.unwrap_or(Target::V2);
        if target == Target::V2 {
            check_group_size(&config, args.group_size)?;
        }
        return convert::convert(&model, out_path.as_ref(), target, args.group_size);
    }

    // Generation mode: requires a prompt and a tokenizer.
    if let Some(prompt) = args.prompt.as_deref() {
        let tok_path = args.tokenizer.as_deref().ok_or_else(|| {
            HostError::Usage("generation (--prompt) requires a tokenizer (--tokenizer)".into())
        })?;
        let vocab = Vocab::load(tok_path, config.vocab_size)?;
        let sampler = Sampler::new(args.temperature, args.topp, resolve_seed(args.seed));
        let kernel = select_kernel(args.scalar, args.dotprod);

        // A v2 checkpoint is already int8: de-interleave its tensors into the
        // engine layout and go straight to the Q8 path — no quantization work,
        // and no fp32 weights ever resident.
        if let ModelFormat::V2 { group_size } = model.format {
            if args.quantize {
                eprintln!(
                    "[--quantize: checkpoint is already int8 (v2, group_size {group_size}); \
                     flag ignored]"
                );
            }
            let bufs = model.unpack_q8()?;
            drop(model); // free the raw file; the unpacked buffers are all we need
            let qw = QuantizedWeights::new(
                &bufs.data,
                &bufs.scales,
                &bufs.rms_att,
                &bufs.rms_ffn,
                &bufs.rms_final,
                bufs.group_size,
                &config,
            )
            .map_err(|e| HostError::engine(model_path, e))?;
            return generate(
                &config,
                &ModelWeights::Q8(qw),
                &vocab,
                prompt,
                args.steps,
                sampler,
                kernel,
            );
        }

        if args.quantize {
            let gs = args.group_size;
            check_group_size(&config, gs)?;
            // Quantize from the fp32 file and copy out the tiny fp32 RMSNorm gains,
            // then drop the file so only the int8 weights stay resident while we
            // generate. (`QuantizedWeights` borrows none of the original file.)
            let mut data = vec![0i8; quantized_weight_count(&config)];
            let mut scales = vec![0.0f32; quantized_scale_count(&config, gs)];
            let (rms_att, rms_ffn, rms_final);
            {
                let weights = model.weights()?;
                quantize_weights(&weights, &mut data, &mut scales, gs, &config);
                rms_att = weights.rms_att.to_vec();
                rms_ffn = weights.rms_ffn.to_vec();
                rms_final = weights.rms_final.to_vec();
            }
            report_quantization(&data, &scales, model.weight_bytes(), gs);
            drop(model); // free the fp32 checkpoint (the bulk of the memory)
            let qw = QuantizedWeights::new(
                &data, &scales, &rms_att, &rms_ffn, &rms_final, gs, &config,
            )
            .map_err(|e| HostError::engine(model_path, e))?;
            return generate(
                &config,
                &ModelWeights::Q8(qw),
                &vocab,
                prompt,
                args.steps,
                sampler,
                kernel,
            );
        }

        let weights = model.weights()?;
        return generate(
            &config,
            &ModelWeights::F32(weights),
            &vocab,
            prompt,
            args.steps,
            sampler,
            kernel,
        );
    }

    // Report mode: validate the weight layout, then print the model summary.
    // (v2 has no fp32 view; its layout was already size-validated at load.)
    if !matches!(model.format, ModelFormat::V2 { .. }) {
        let _ = model.weights()?;
    }
    report_model(model_path, &model);
    if let Some(tok_path) = args.tokenizer.as_deref() {
        let vocab = Vocab::load(tok_path, model.config.vocab_size)?;
        report_tokenizer(tok_path, &vocab);
    } else {
        println!("\nTokenizer: (none provided — pass a path to inspect tokenizer.bin)");
    }

    report_memory(&model);
    Ok(())
}
