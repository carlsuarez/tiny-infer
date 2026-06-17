//! Checkpoint loading: read a llama2.c `.bin` file (legacy, v1, or v2) and expose
//! its config plus the right weight view for its format.
//!
//! ## Formats
//!
//! [`engine::parse_header`] detects the format from the leading bytes:
//! * **legacy / v1** (fp32): [`Model::weights`] returns a zero-copy [`Weights`]
//!   view straight into the file buffer (the two formats differ only in header
//!   size and tensor order).
//! * **v2** (pre-quantized int8): the file interleaves each tensor's int8 data
//!   with its `f32` scales per layer, which the engine's flat
//!   [`QuantizedTensor`](engine::QuantizedTensor) views cannot borrow directly.
//!   [`Model::unpack_q8`] de-interleaves them into owned buffers (one linear
//!   copy — no fp32 weights ever materialize), which feed
//!   [`engine::QuantizedWeights::new`].
//!
//! ## Alignment
//!
//! The fp32 views borrow `&[f32]` out of the file buffer, so it must be 4-byte
//! aligned. A `Vec<u8>` only guarantees alignment 1, so the file is read directly
//! into a `Vec<f32>` (4-aligned by construction) through a byte view of its storage
//! obtained with [`bytemuck::cast_slice_mut`]. That keeps the load to a single read
//! with no extra copy, and the `f32` ↔ `u8` reinterpretation is a *safe* `bytemuck`
//! cast (always sound — `u8` has alignment 1), so the host needs no `unsafe` of its
//! own. Every supported format is 4-byte–granular: headers are 28 or 256 bytes, fp32
//! payloads trivially, and v2 payloads because every tensor's weight count is a
//! multiple of `dim` (a multiple of 4 for any real model).

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use llama::config::{is_llama, HEADER_BYTES, VERSIONED_HEADER_BYTES};
use llama::quantize::{quantized_scale_count, quantized_weight_count, v2_tensor_sizes};
use llama::weights::expected_file_bytes;
use llama::{parse_header, Config, ModelFormat, Weights};

use crate::error::HostError;

/// Whether the file at `path` looks like a llama2.c checkpoint (legacy, v1, or v2) —
/// the sibling of [`is_seq2seq_file`](crate::seq2seq::is_seq2seq_file).
///
/// Reads only the leading header bytes and defers to [`is_llama`]. Unlike the seq2seq
/// check, the legacy llama format has no magic, so this is a **fallback**: a file whose
/// first bytes form a plausible seven-int header passes even if it belongs to another
/// architecture. The dispatcher in `main` must therefore check the magic-bearing
/// formats (seq2seq, and any future architecture) *before* this one — see [`is_llama`].
///
/// A missing file is an [`HostError::Io`]; a file too short to identify is simply not a
/// llama checkpoint.
pub fn is_llama_file(path: impl AsRef<Path>) -> Result<bool, HostError> {
    let path = path.as_ref();
    let mut file = File::open(path).map_err(|e| HostError::io(path, e))?;

    // The versioned magic sits in the first 4 bytes; the legacy header needs all
    // `HEADER_BYTES`. Read up to that many, tolerating a shorter file (which then
    // simply fails to identify as llama).
    let mut header = [0u8; HEADER_BYTES];
    let mut filled = 0;
    while filled < header.len() {
        match file.read(&mut header[filled..]) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(HostError::io(path, e)),
        }
    }
    Ok(is_llama(&header[..filled]))
}

/// A loaded checkpoint: parsed config and format, plus the whole file as `f32`.
pub struct Model {
    /// The validated model configuration.
    pub config: Config,
    /// Which on-disk format the file uses (decides the weight layout).
    pub format: ModelFormat,
    /// On-disk size in bytes (header + weights).
    pub file_bytes: usize,
    /// Entire file reinterpreted as `f32`: the header words followed by all weights.
    data: Vec<f32>,
    /// Path it was loaded from, for diagnostics.
    path: PathBuf,
}

/// The owned buffers de-interleaved from a v2 checkpoint, ready for
/// [`engine::QuantizedWeights::new`]. Holding them outside [`Model`] lets the
/// caller drop the file buffer before generating.
pub struct QuantBuffers {
    /// Every quantized tensor's int8 data, flat, in engine storage order.
    pub data: Vec<i8>,
    /// The matching `f32` scales, flat, in the same order.
    pub scales: Vec<f32>,
    /// Per-layer attention RMSNorm gains (fp32).
    pub rms_att: Vec<f32>,
    /// Per-layer feed-forward RMSNorm gains (fp32).
    pub rms_ffn: Vec<f32>,
    /// Final RMSNorm gains (fp32).
    pub rms_final: Vec<f32>,
    /// Quantization group size, from the v2 header.
    pub group_size: usize,
}

impl Model {
    /// Load and validate a checkpoint from `path`.
    ///
    /// Validates that the file is `f32`-sized, detects the format and parses the
    /// [`Config`] header, and confirms the file length equals what the config and
    /// format imply.
    pub fn load(path: impl AsRef<Path>) -> Result<Model, HostError> {
        let path = path.as_ref();
        let mut file = File::open(path).map_err(|e| HostError::io(path, e))?;
        let len = file.metadata().map_err(|e| HostError::io(path, e))?.len();

        if !len.is_multiple_of(4) {
            return Err(HostError::NotF32Aligned {
                path: path.to_path_buf(),
                len,
            });
        }
        let n_floats = (len / 4) as usize;

        // Read the whole file into a 4-aligned f32 buffer in one pass.
        let mut data = vec![0.0f32; n_floats];
        file.read_exact(bytemuck::cast_slice_mut(&mut data))
            .map_err(|e| HostError::io(path, e))?;

        // Detect the format and parse + validate the config.
        let (config, format) =
            parse_header(bytemuck::cast_slice(&data)).map_err(|e| HostError::engine(path, e))?;

        let expected = expected_file_bytes(&config, format);
        let file_bytes = len as usize;
        if expected != file_bytes {
            return Err(HostError::engine(
                path,
                engine::EngineError::SizeMismatch {
                    expected,
                    actual: file_bytes,
                },
            ));
        }

        Ok(Model {
            config,
            format,
            file_bytes,
            data,
            path: path.to_path_buf(),
        })
    }

    /// The weight region (everything after the header) as `f32`.
    fn weight_floats(&self) -> &[f32] {
        &self.data[self.format.header_bytes() / 4..]
    }

    /// Zero-copy view of all weight tensors (fp32 formats only).
    ///
    /// Dispatches on the detected format's tensor order. A v2 checkpoint has no
    /// fp32 weights to view — use [`Model::unpack_q8`] for those.
    pub fn weights(&self) -> Result<Weights<'_>, HostError> {
        match self.format {
            ModelFormat::Legacy => Weights::new(self.weight_floats(), &self.config),
            ModelFormat::V1 => Weights::new_v1(self.weight_floats(), &self.config),
            ModelFormat::V2 { .. } => {
                return Err(HostError::Format {
                    path: self.path.clone(),
                    msg: "checkpoint stores int8 (v2) weights; there is no fp32 view".into(),
                })
            }
        }
        .map_err(|e| HostError::engine(&self.path, e))
    }

    /// De-interleave a v2 checkpoint's quantized tensors into owned flat buffers.
    ///
    /// The file stores, per tensor (and per layer), int8 data followed by its
    /// `f32` scales; the engine wants one flat data buffer and one flat scales
    /// buffer ([`llama::quantize::quantize_weights`]' layout). Both walks visit
    /// the tensors in the same order ([`v2_tensor_sizes`]), so appending each
    /// part as it is encountered produces exactly that layout. The RMSNorm gains
    /// (fp32, stored before the tensors) are copied out alongside. One linear
    /// pass; the caller can drop the [`Model`] afterwards.
    pub fn unpack_q8(&self) -> Result<QuantBuffers, HostError> {
        let ModelFormat::V2 { group_size } = self.format else {
            return Err(HostError::Format {
                path: self.path.clone(),
                msg: "checkpoint stores fp32 weights; quantize at load with --quantize instead"
                    .into(),
            });
        };
        let c = &self.config;
        let (l, dim) = (c.n_layers, c.dim);

        // The fp32 norms sit right after the header; the f32 file buffer is
        // already aligned, so they are plain subslices.
        let f = self.weight_floats();
        let rms_att = f[..l * dim].to_vec();
        let rms_ffn = f[l * dim..2 * l * dim].to_vec();
        let rms_final = f[2 * l * dim..2 * l * dim + dim].to_vec();

        // Walk the interleaved tensors byte-wise (int8 data is unaligned by
        // nature; scales are decoded with `from_le_bytes`, which needs none).
        let bytes: &[u8] = bytemuck::cast_slice(&self.data);
        let mut off = VERSIONED_HEADER_BYTES + (2 * l * dim + dim) * 4;
        let mut data = Vec::with_capacity(quantized_weight_count(c));
        let mut scales = Vec::with_capacity(quantized_scale_count(c, group_size));
        for n in v2_tensor_sizes(c) {
            data.extend(bytes[off..off + n].iter().map(|&b| b as i8));
            off += n;
            let g = n / group_size;
            scales.extend(
                bytes[off..off + 4 * g]
                    .chunks_exact(4)
                    .map(|s| f32::from_le_bytes([s[0], s[1], s[2], s[3]])),
            );
            off += 4 * g;
        }
        // The size validation at load guarantees the walk lands exactly at EOF.
        debug_assert_eq!(off, self.file_bytes);

        Ok(QuantBuffers {
            data,
            scales,
            rms_att,
            rms_ffn,
            rms_final,
            group_size,
        })
    }

    /// On-disk size of just the weights (excludes the format's header).
    pub fn weight_bytes(&self) -> usize {
        self.file_bytes - self.format.header_bytes()
    }

    /// The path this model was loaded from, for diagnostics.
    pub fn path(&self) -> &Path {
        &self.path
    }
}
