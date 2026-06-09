//! Int8 weight quantization for the matmul-heavy projections.
//!
//! Storing the large projection matrices as 8-bit integers instead of `f32` cuts
//! their memory footprint ~4×. Accuracy stays high because the scheme is
//! *group-wise* and *symmetric*: every run of `group_size` consecutive weights
//! shares one `f32` scale, so quantization adapts to the local magnitude of each
//! group rather than using a single scale for an entire row. The dequantized value
//! of weight `k` is simply `data[k] as f32 * scales[k / group_size]`.
//!
//! This mirrors the scheme in llama2.c's `runq.c` (`GS` = group size). Only the
//! RMSNorm gains (`rms_att`, `rms_ffn`, `rms_final`) stay `f32` — they are 1-D and
//! tiny. Everything else is quantized, including the token-embedding / classifier
//! table:
//!
//! * the seven layer projections `wq, wk, wv, wo, w1, w2, w3`, and
//! * the `token_embedding` table, which doubles as the `wcls` classifier when the
//!   checkpoint shares weights. It is stored **once** as int8: the classifier reads
//!   it as a matmul ([`matmul_q8`](crate::math::matmul_q8)), and the embedding lookup
//!   dequantizes just the one row it needs ([`QuantizedTensor::layer`] +
//!   [`dequantize`]). When weights are not shared, `wcls` is a second quantized table.
//!
//! Storing the embedding once as int8 — instead of keeping a full fp32 copy just for
//! the lookup — is the bulk of the memory win on these embedding-dominated models.
//!
//! ## Where the buffers come from
//!
//! The engine never allocates. The host quantizes the fp32 checkpoint **once** at
//! load time into two host-owned buffers — an `i8` data buffer
//! ([`quantized_weight_count`] long) and an `f32` scales buffer
//! ([`quantized_scale_count`] long) — using [`quantize_weights`], then hands them to
//! [`QuantizedWeights::new`] along with the three fp32 RMSNorm gains.
//! [`quantize_weights`] and [`QuantizedWeights::new`] walk the tensors in the **same
//! order** (`token_embedding`, the seven projections, then `wcls` only when weights
//! are not shared), so the fill and the view always agree. Because `new` borrows
//! nothing from the original fp32 file, the host can free that file once the int8
//! buffers and the gains are in hand.

use crate::config::Config;
use crate::error::EngineError;
use crate::weights::Weights;

/// A group-wise int8-quantized weight matrix: 8-bit values plus per-group scales.
///
/// Element `k` dequantizes to `data[k] as f32 * scales[k / group_size]`. `data`
/// holds the quantized weights in the same `[d_out, d_in]` row-major layout the
/// fp32 [`Weights`] use, and `scales` holds one `f32` per group of `group_size`
/// consecutive weights, so `scales.len() == data.len() / group_size`. `group_size`
/// is chosen to divide `d_in`, so a group never straddles two rows.
///
/// A `QuantizedTensor` spans *every layer* of one projection (e.g. all of `wq`);
/// use [`QuantizedTensor::layer`] to slice out a single layer's matrix before
/// passing it to [`matmul_q8`](crate::math::matmul_q8).
#[derive(Debug, Clone, Copy)]
pub struct QuantizedTensor<'a> {
    /// The quantized weight values, `[..., d_out, d_in]` row-major.
    pub data: &'a [i8],
    /// One `f32` scale per group of `group_size` weights.
    pub scales: &'a [f32],
    /// How many consecutive weights share a single scale.
    pub group_size: usize,
}

impl<'a> QuantizedTensor<'a> {
    /// Borrow layer `l`'s `[rows, cols]` sub-matrix as its own `QuantizedTensor`.
    ///
    /// The flat tensor stores layers back to back, so layer `l` occupies `rows*cols`
    /// quantized values at offset `l*rows*cols`, with the matching
    /// `rows*cols/group_size` scales at offset `l*rows*cols/group_size`. The result
    /// is exactly what [`matmul_q8`](crate::math::matmul_q8) expects for a single
    /// projection — the quantized analogue of slicing
    /// `w.wq[l*rows*cols..][..rows*cols]` out of the fp32 [`Weights`].
    pub fn layer(&self, l: usize, rows: usize, cols: usize) -> QuantizedTensor<'a> {
        let n = rows * cols;
        let g = n / self.group_size;
        QuantizedTensor {
            data: &self.data[l * n..][..n],
            scales: &self.scales[l * g..][..g],
            group_size: self.group_size,
        }
    }
}

/// Quantize `x` into `data` (8-bit values) + `scales` (one `f32` per group).
///
/// Symmetric per-group quantization, matching llama2.c `runq.c`: for each group of
/// `group_size` consecutive values, the scale is `max(|x|) / 127`, which maps the
/// group's largest magnitude to ±127; every value is then divided by that scale and
/// rounded to the nearest integer. Dequantizing (`q * scale`) recovers each value to
/// within half a scale step. An all-zero group has scale `0`; its values quantize to
/// `0` (the divide is skipped so it never produces a NaN).
///
/// Writes `x.len()` values into `data` and `x.len() / group_size` scales into
/// `scales`. `x.len()` must be a multiple of `group_size`.
///
/// # Panics
/// In debug builds, if the slice lengths are inconsistent with `group_size`.
pub fn quantize(data: &mut [i8], scales: &mut [f32], x: &[f32], group_size: usize) {
    debug_assert_eq!(data.len(), x.len());
    debug_assert_eq!(x.len() % group_size, 0);
    debug_assert_eq!(scales.len(), x.len() / group_size);

    const Q_MAX: f32 = 127.0;

    for (g, group) in x.chunks_exact(group_size).enumerate() {
        // The largest magnitude in the group sets its scale.
        let mut wmax = 0.0f32;
        for &v in group {
            let a = libm::fabsf(v);
            if a > wmax {
                wmax = a;
            }
        }
        let scale = wmax / Q_MAX;
        scales[g] = scale;

        let base = g * group_size;
        for (k, &v) in group.iter().enumerate() {
            // Skip the divide for an all-zero group so scale==0 can't make a NaN.
            // `f32 as i8` saturates in Rust, but rounding keeps every value in range.
            let q = if scale != 0.0 {
                libm::roundf(v / scale)
            } else {
                0.0
            };
            data[base + k] = q as i8;
        }
    }
}

/// Quantize activations `x` into int8 `out` + per-group `scales` + per-group integer
/// `gsums`, in a single pass.
///
/// The activation-side counterpart of [`quantize`], called once per matmul on the W8A8
/// path. Same scheme (symmetric, group-wise, `scale = max(|x|)/127`, round to nearest),
/// producing the same `out`/`scales` as [`quantize`]. It *additionally* records, per
/// group, the **sum of the quantized int8 values** in `gsums` (held as exact `f32`).
/// That sum is the correction term the VNNI dot-product kernel needs: hardware
/// `vpdpbusd` multiplies *unsigned* × *signed* bytes, so the kernel offsets each weight
/// by `+128` and subtracts `128 · Σ(activations)` per group to recover the true signed
/// dot product. The scalar/SIMD kernels ignore `gsums`.
///
/// Writes `x.len()` values into `out` and `x.len() / group_size` entries into each of
/// `scales` and `gsums`. `x.len()` must be a multiple of `group_size`.
///
/// # Panics
/// In debug builds, if the slice lengths are inconsistent with `group_size`.
pub fn quantize_activation(
    out: &mut [i8],
    scales: &mut [f32],
    gsums: &mut [f32],
    x: &[f32],
    group_size: usize,
) {
    debug_assert_eq!(out.len(), x.len());
    debug_assert_eq!(x.len() % group_size, 0);
    debug_assert_eq!(scales.len(), x.len() / group_size);
    debug_assert_eq!(gsums.len(), x.len() / group_size);

    const Q_MAX: f32 = 127.0;

    for (g, group) in x.chunks_exact(group_size).enumerate() {
        let mut wmax = 0.0f32;
        for &v in group {
            let a = libm::fabsf(v);
            if a > wmax {
                wmax = a;
            }
        }
        let scale = wmax / Q_MAX;
        scales[g] = scale;

        let base = g * group_size;
        let mut sum: i32 = 0;
        for (k, &v) in group.iter().enumerate() {
            // All-zero group: scale==0, skip the divide so it can't make a NaN.
            let q = if scale != 0.0 {
                libm::roundf(v / scale) as i8
            } else {
                0
            };
            out[base + k] = q;
            sum += q as i32;
        }
        gsums[g] = sum as f32;
    }
}

/// Dequantize `qx` back into `out`: `out[k] = data[k] as f32 * scales[k/group_size]`.
///
/// The inverse of [`quantize`]; used mainly to measure round-trip error in tests.
pub fn dequantize(out: &mut [f32], qx: &QuantizedTensor) {
    debug_assert_eq!(out.len(), qx.data.len());
    for (k, o) in out.iter_mut().enumerate() {
        *o = qx.data[k] as f32 * qx.scales[k / qx.group_size];
    }
}

/// Lengths (in weights) of the always-quantized tensors, in storage order: the
/// `token_embedding` table first (it doubles as `wcls` when weights are shared), then
/// the seven layer projections.
///
/// This is the single source of truth for *which* tensors are quantized and in what
/// order; [`quantize_weights`] and [`QuantizedWeights::new`] both iterate it so the
/// fill and the view can never drift apart. A non-shared `wcls` is a separate
/// `[vocab, dim]` block appended after these (see [`quantized_weight_count`]).
fn quantized_lens(c: &Config) -> [usize; 8] {
    let att_dim = c.n_heads * c.head_size();
    let kv_dim = c.kv_dim();
    let l = c.n_layers;
    [
        c.vocab_size * c.dim,     // token_embedding (also wcls when shared)
        l * c.dim * att_dim,      // wq
        l * c.dim * kv_dim,       // wk
        l * c.dim * kv_dim,       // wv
        l * att_dim * c.dim,      // wo
        l * c.hidden_dim * c.dim, // w1
        l * c.dim * c.hidden_dim, // w2
        l * c.hidden_dim * c.dim, // w3
    ]
}

/// Total number of `i8` weights a checkpoint of this config needs once quantized.
/// Use this to size the host's data buffer. Includes a separate `wcls` table only
/// when the checkpoint does not share weights.
#[inline(always)]
pub fn quantized_weight_count(c: &Config) -> usize {
    let mut n: usize = quantized_lens(c).iter().sum();
    if !c.shared_weights {
        n += c.vocab_size * c.dim; // separate classifier
    }
    n
}

/// Total number of `f32` scales the quantized weights need for `group_size`. Use
/// this to size the host's scales buffer.
#[inline(always)]
pub fn quantized_scale_count(c: &Config, group_size: usize) -> usize {
    quantized_weight_count(c) / group_size
}

/// Quantize the matmul matrices of `fp32` into `data` + `scales`.
///
/// Fills the caller-provided buffers in storage order — `token_embedding`, then
/// `wq, wk, wv, wo, w1, w2, w3`, then `wcls` only when weights are not shared (when
/// they are, `wcls` *is* `token_embedding`, already written). This is the same order
/// [`QuantizedWeights::new`] reads them back. `data` must be
/// [`quantized_weight_count`] long and `scales` [`quantized_scale_count`] long; both
/// are filled completely. Allocates nothing, so the host can call it over buffers it
/// owns.
///
/// # Panics
/// In debug builds, if `data` / `scales` are not the exact required lengths, or if
/// `group_size` does not divide every tensor.
pub fn quantize_weights(
    fp32: &Weights,
    data: &mut [i8],
    scales: &mut [f32],
    group_size: usize,
    c: &Config,
) {
    debug_assert_eq!(data.len(), quantized_weight_count(c));
    debug_assert_eq!(scales.len(), quantized_scale_count(c, group_size));

    // Advance disjoint cursors through both output buffers in lockstep.
    let mut data = data;
    let mut scales = scales;
    let mut emit = |src: &[f32]| {
        let n = src.len();
        let g = n / group_size;
        let (qd, dtail) = core::mem::take(&mut data).split_at_mut(n);
        let (qs, stail) = core::mem::take(&mut scales).split_at_mut(g);
        quantize(qd, qs, src, group_size);
        data = dtail;
        scales = stail;
    };

    emit(fp32.token_embedding);
    emit(fp32.wq);
    emit(fp32.wk);
    emit(fp32.wv);
    emit(fp32.wo);
    emit(fp32.w1);
    emit(fp32.w2);
    emit(fp32.w3);
    if !c.shared_weights {
        emit(fp32.wcls);
    }
}

/// Borrowed, zero-copy views of every weight tensor, with the projections quantized.
///
/// The quantized analogue of [`Weights`]: the seven projection matrices are
/// [`QuantizedTensor`]s viewing the host's `i8`/`f32` buffers, while the RMSNorm
/// gains, the token embedding, and the classifier stay full-precision `&[f32]`.
/// `wcls` aliases `token_embedding` when the checkpoint shares weights.
#[derive(Debug, Clone, Copy)]
pub struct QuantizedWeights<'a> {
    /// Token embedding table, `[vocab, dim]`, quantized. The lookup dequantizes one
    /// row at a time; it also serves as `wcls` when the checkpoint shares weights.
    pub token_embedding: QuantizedTensor<'a>,
    /// Per-layer attention RMSNorm gains, `[n_layers, dim]` (kept fp32).
    pub rms_att: &'a [f32],
    /// Query projection, `[n_layers, dim, att_dim]`.
    pub wq: QuantizedTensor<'a>,
    /// Key projection, `[n_layers, dim, kv_dim]`.
    pub wk: QuantizedTensor<'a>,
    /// Value projection, `[n_layers, dim, kv_dim]`.
    pub wv: QuantizedTensor<'a>,
    /// Attention output projection, `[n_layers, att_dim, dim]`.
    pub wo: QuantizedTensor<'a>,
    /// Per-layer feed-forward RMSNorm gains, `[n_layers, dim]` (kept fp32).
    pub rms_ffn: &'a [f32],
    /// SwiGLU gate projection, `[n_layers, hidden, dim]`.
    pub w1: QuantizedTensor<'a>,
    /// SwiGLU down projection, `[n_layers, dim, hidden]`.
    pub w2: QuantizedTensor<'a>,
    /// SwiGLU up projection, `[n_layers, hidden, dim]`.
    pub w3: QuantizedTensor<'a>,
    /// Final RMSNorm gains, `[dim]` (kept fp32).
    pub rms_final: &'a [f32],
    /// Output classifier, `[vocab, dim]`, quantized. Aliases `token_embedding` when
    /// the checkpoint shares weights, otherwise its own quantized table.
    pub wcls: QuantizedTensor<'a>,
}

impl<'a> QuantizedWeights<'a> {
    /// Assemble quantized weight views from the host's quantized buffers and the
    /// three fp32 RMSNorm gains.
    ///
    /// Carves `token_embedding` and the seven projections from `data` / `scales`
    /// (filled by [`quantize_weights`] with the same config and `group_size`); `wcls`
    /// aliases `token_embedding` when weights are shared, else it is carved as a
    /// further table. The `rms_*` slices are stored as-is. This borrows **nothing**
    /// from the original fp32 checkpoint, so the host may free that file first.
    /// Returns [`EngineError::SizeMismatch`] if a buffer is shorter than required.
    pub fn new(
        data: &'a [i8],
        scales: &'a [f32],
        rms_att: &'a [f32],
        rms_ffn: &'a [f32],
        rms_final: &'a [f32],
        group_size: usize,
        c: &Config,
    ) -> Result<QuantizedWeights<'a>, EngineError> {
        let need_data = quantized_weight_count(c);
        let need_scales = quantized_scale_count(c, group_size);
        if data.len() < need_data || scales.len() < need_scales {
            return Err(EngineError::SizeMismatch {
                // Report the i8 shortfall (1 byte per weight); a wrong group_size is
                // the usual cause of a scales-length mismatch.
                expected: need_data,
                actual: data.len(),
            });
        }

        // Two cursors advancing in lockstep, mirroring `quantize_weights`' order.
        let mut drest = data;
        let mut srest = scales;
        let mut take = |n: usize| -> QuantizedTensor<'a> {
            let g = n / group_size;
            let (qd, dtail) = drest.split_at(n);
            let (qs, stail) = srest.split_at(g);
            drest = dtail;
            srest = stail;
            QuantizedTensor {
                data: qd,
                scales: qs,
                group_size,
            }
        };

        let lens = quantized_lens(c);
        let token_embedding = take(lens[0]);
        let wq = take(lens[1]);
        let wk = take(lens[2]);
        let wv = take(lens[3]);
        let wo = take(lens[4]);
        let w1 = take(lens[5]);
        let w2 = take(lens[6]);
        let w3 = take(lens[7]);
        // Shared weights reuse the embedding table as the classifier; otherwise it is
        // a separate quantized block right after the projections.
        let wcls = if c.shared_weights {
            token_embedding
        } else {
            take(c.vocab_size * c.dim)
        };

        Ok(QuantizedWeights {
            token_embedding,
            rms_att,
            wq,
            wk,
            wv,
            wo,
            rms_ffn,
            w1,
            w2,
            w3,
            rms_final,
            wcls,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    extern crate std;
    use std::vec;
    use std::vec::Vec;

    #[test]
    fn quantize_roundtrips_within_half_a_step() {
        let x: Vec<f32> = (0..8).map(|i| (i as f32 - 3.5) * 0.7).collect();
        let mut data = vec![0i8; 8];
        let mut scales = vec![0.0f32; 2]; // group_size 4 → 2 groups
        quantize(&mut data, &mut scales, &x, 4);

        let mut back = vec![0.0f32; 8];
        let qx = QuantizedTensor {
            data: &data,
            scales: &scales,
            group_size: 4,
        };
        dequantize(&mut back, &qx);

        // Each value recovers to within half a scale step of its group.
        for (i, (&orig, &deq)) in x.iter().zip(back.iter()).enumerate() {
            let step = scales[i / 4];
            assert!(
                (orig - deq).abs() <= step / 2.0 + 1e-6,
                "i={i}: {orig} vs {deq}"
            );
        }
    }

    #[test]
    fn quantize_activation_matches_weight_quantizer_and_sums_groups() {
        // The activation quantizer must produce the same int8 values and scales as the
        // weight quantizer, plus the per-group integer sum used by the VNNI correction.
        let x: Vec<f32> = (0..8).map(|i| (i as f32 - 3.5) * 0.7).collect();
        let mut qi = vec![0i8; 8];
        let mut si = vec![0.0f32; 2];
        quantize(&mut qi, &mut si, &x, 4);

        let mut qa = vec![0i8; 8];
        let mut sa = vec![0.0f32; 2];
        let mut gsums = vec![0.0f32; 2];
        quantize_activation(&mut qa, &mut sa, &mut gsums, &x, 4);

        assert_eq!(qi, qa);
        assert_eq!(si, sa);
        // gsums[g] is the exact integer sum of that group's quantized values.
        assert_eq!(gsums[0], qa[..4].iter().map(|&v| v as i32).sum::<i32>() as f32);
        assert_eq!(gsums[1], qa[4..].iter().map(|&v| v as i32).sum::<i32>() as f32);
    }

    #[test]
    fn all_zero_group_has_zero_scale_and_no_nan() {
        let x = [0.0f32; 4];
        let mut data = [0i8; 4];
        let mut scales = [1.0f32; 1];
        quantize(&mut data, &mut scales, &x, 4);
        assert_eq!(scales[0], 0.0);
        assert!(data.iter().all(|&q| q == 0));
    }

    #[test]
    fn max_magnitude_maps_to_the_rail() {
        // The largest |value| in a group quantizes to ±127.
        let x = [1.0f32, -2.0, 0.5, 2.0];
        let mut data = [0i8; 4];
        let mut scales = [0.0f32; 1];
        quantize(&mut data, &mut scales, &x, 4);
        assert_eq!(scales[0], 2.0 / 127.0);
        assert_eq!(data[1], -127);
        assert_eq!(data[3], 127);
    }

    #[test]
    fn layer_slices_out_one_matrix() {
        // 2 layers of a 2x4 matrix (rows=2, cols=4), group_size 4 → 1 scale/row.
        let data: Vec<i8> = (0..16).map(|i| i as i8).collect();
        let scales = vec![1.0, 2.0, 3.0, 4.0]; // 2 rows * 2 layers
        let t = QuantizedTensor {
            data: &data,
            scales: &scales,
            group_size: 4,
        };

        let l1 = t.layer(1, 2, 4);
        assert_eq!(l1.data, &[8, 9, 10, 11, 12, 13, 14, 15]);
        assert_eq!(l1.scales, &[3.0, 4.0]);
        assert_eq!(l1.group_size, 4);
    }

    fn tiny_config(shared: bool) -> Config {
        Config {
            dim: 4,
            hidden_dim: 8,
            n_layers: 1,
            n_heads: 2,
            n_kv_heads: 2,
            vocab_size: 6,
            seq_len: 4,
            shared_weights: shared,
        }
    }

    fn quantized(c: &Config, gs: usize) -> (Vec<i8>, Vec<f32>) {
        // Distinct per-element values so the embedding and a separate wcls differ.
        let wbuf: Vec<f32> = (0..crate::weights::weight_floats(c))
            .map(|i| (i as f32).sin() * 0.1)
            .collect();
        let fp32 = Weights::new(&wbuf, c).unwrap();
        let mut data = vec![0i8; quantized_weight_count(c)];
        let mut scales = vec![0.0f32; quantized_scale_count(c, gs)];
        quantize_weights(&fp32, &mut data, &mut scales, gs, c);
        (data, scales)
    }

    #[test]
    fn shared_weights_alias_wcls_to_the_embedding() {
        let c = tiny_config(true);
        let gs = 2;
        let (data, scales) = quantized(&c, gs);
        let rms = vec![1.0f32; c.dim]; // any fp32 gains; not under test here
        let qw = QuantizedWeights::new(&data, &scales, &rms, &rms, &rms, gs, &c).unwrap();
        // The classifier reuses the exact same int8 table as the embedding.
        assert_eq!(qw.wcls.data.as_ptr(), qw.token_embedding.data.as_ptr());
        assert_eq!(qw.token_embedding.data.len(), c.vocab_size * c.dim);
    }

    #[test]
    fn unshared_weights_give_wcls_its_own_table() {
        let c = tiny_config(false);
        let gs = 2;
        let (data, scales) = quantized(&c, gs);
        let rms = vec![1.0f32; c.dim];
        let qw = QuantizedWeights::new(&data, &scales, &rms, &rms, &rms, gs, &c).unwrap();
        assert_ne!(qw.wcls.data.as_ptr(), qw.token_embedding.data.as_ptr());
        assert_eq!(qw.wcls.data.len(), c.vocab_size * c.dim);
    }
}
