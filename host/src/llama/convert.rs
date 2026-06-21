//! Checkpoint format conversion: write a loaded fp32 model back out as v1 or v2.
//!
//! `--convert <PATH> --to <v1|v2>` reads any fp32 checkpoint (legacy or v1) and
//! serializes it in the requested format:
//!
//! * **v1** — fp32 with the 256-byte versioned header and the v1 tensor order
//!   (norms first, no `freq_cis` tables). Lossless; bit-identical weights.
//! * **v2** — Q8_0: the RMSNorm gains in fp32, every matmul weight
//!   group-wise int8-quantized, each tensor's data followed by its scales. Uses
//!   the engine's own [`quantize`] kernel, so loading the file back reproduces
//!   `--quantize` **bit-for-bit** — and a pre-quantized file skips both the
//!   quantization work and the fp32 peak RSS at load time.
//!
//! Converting *from* a v2 checkpoint is rejected: the int8 → fp32 round trip
//! would silently bake quantization loss into an ostensibly fp32 file.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use llama::config::{MAGIC, VERSIONED_HEADER_BYTES};
use llama::{Config, ModelFormat, Weights};
use engine::quant::quantize;

use crate::error::HostError;
use crate::llama::loader::Model;

/// Which format `--convert` should write (`--to`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    /// Version 1: fp32, 256-byte header, norms-first tensor order.
    V1,
    /// Version 2: group-wise int8 (Q8_0), the quantized checkpoint format.
    V2,
}

/// Convert `model` to `target` format and write it to `out_path`.
///
/// `group_size` is only used for [`Target::V2`] (the caller validates it divides
/// the model's dimensions, as for `--quantize`). Prints a one-line summary to
/// stderr on success.
pub fn convert(
    model: &Model,
    out_path: &Path,
    target: Target,
    group_size: usize,
) -> Result<(), HostError> {
    if matches!(model.format, ModelFormat::V2 { .. }) {
        return Err(HostError::Format {
            path: model.path().to_path_buf(),
            msg: "already int8 (v2); converting from a quantized checkpoint is not supported"
                .into(),
        });
    }
    let w = model.weights()?;
    let c = &model.config;

    let file = File::create(out_path).map_err(|e| HostError::io(out_path, e))?;
    let mut out = BufWriter::new(file);
    match target {
        Target::V1 => write_v1(&mut out, &w, c),
        Target::V2 => write_v2(&mut out, &w, c, group_size),
    }
    .and_then(|()| out.flush())
    .map_err(|e| HostError::io(out_path, e))?;

    let written = std::fs::metadata(out_path)
        .map(|m| m.len() as usize)
        .unwrap_or(0);
    eprintln!(
        "[wrote {} checkpoint: {} ({} bytes){}]",
        match target {
            Target::V1 => "v1 (fp32)",
            Target::V2 => "v2 (int8)",
        },
        out_path.display(),
        written,
        match target {
            Target::V1 => String::new(),
            Target::V2 => format!(", group_size {group_size}"),
        },
    );
    Ok(())
}

/// Build the 256-byte versioned header: magic, version, the seven config fields,
/// the shared-classifier flag, the v2 group size, then zero padding.
fn header(c: &Config, version: i32, group_size: Option<i32>) -> [u8; VERSIONED_HEADER_BYTES] {
    let mut h = [0u8; VERSIONED_HEADER_BYTES];
    h[0..4].copy_from_slice(&MAGIC.to_le_bytes());
    h[4..8].copy_from_slice(&version.to_le_bytes());
    let fields = [
        c.dim,
        c.hidden_dim,
        c.n_layers,
        c.n_heads,
        c.n_kv_heads,
        c.vocab_size, // always positive here; sharing is the flag byte below
        c.seq_len,
    ];
    for (i, v) in fields.iter().enumerate() {
        h[8 + i * 4..8 + i * 4 + 4].copy_from_slice(&(*v as i32).to_le_bytes());
    }
    h[36] = c.shared_weights as u8;
    if let Some(gs) = group_size {
        h[37..41].copy_from_slice(&gs.to_le_bytes());
    }
    h
}

/// Serialize the model as a v1 checkpoint: header, then every fp32 tensor in the
/// v1 order (norms, embedding, projections, unshared classifier).
fn write_v1(out: &mut impl Write, w: &Weights, c: &Config) -> io::Result<()> {
    out.write_all(&header(c, 1, None))?;
    for s in [
        w.rms_att,
        w.rms_ffn,
        w.rms_final,
        w.token_embedding,
        w.wq,
        w.wk,
        w.wv,
        w.wo,
        w.w1,
        w.w2,
        w.w3,
    ] {
        out.write_all(bytemuck::cast_slice(s))?;
    }
    if !c.shared_weights {
        out.write_all(bytemuck::cast_slice(w.wcls))?;
    }
    Ok(())
}

/// Serialize the model as a v2 checkpoint: header, fp32 norms, then each matmul
/// tensor quantized to int8 — data immediately followed by its scales, one tensor
/// per layer, in the v2 tensor order (the order [`llama::quantize::v2_tensor_sizes`]
/// describes).
fn write_v2(out: &mut impl Write, w: &Weights, c: &Config, group_size: usize) -> io::Result<()> {
    out.write_all(&header(c, 2, Some(group_size as i32)))?;
    for s in [w.rms_att, w.rms_ffn, w.rms_final] {
        out.write_all(bytemuck::cast_slice(s))?;
    }

    // Quantize-and-write one tensor: int8 data, then its f32 scales. Scratch
    // buffers are reallocated per tensor — fine for a one-shot converter.
    let mut emit = |tensor: &[f32]| -> io::Result<()> {
        let n = tensor.len();
        let mut qd = vec![0i8; n];
        let mut qs = vec![0.0f32; n / group_size];
        quantize(&mut qd, &mut qs, tensor, group_size);
        let bytes: Vec<u8> = qd.iter().map(|&v| v as u8).collect();
        out.write_all(&bytes)?;
        out.write_all(bytemuck::cast_slice(&qs))
    };

    emit(w.token_embedding)?;
    // Each layer of each projection is its own tensor in the v2 file.
    let l = c.n_layers;
    for proj in [w.wq, w.wk, w.wv, w.wo, w.w1, w.w2, w.w3] {
        let per_layer = proj.len() / l;
        for li in 0..l {
            emit(&proj[li * per_layer..][..per_layer])?;
        }
    }
    if !c.shared_weights {
        emit(w.wcls)?;
    }
    Ok(())
}
