//! Group-wise int8 quantization **primitives** — the shared building blocks.
//!
//! These are architecture-agnostic: a [`QuantizedTensor`] view over an `i8` data
//! buffer plus its `f32` scales, and the routines that fill and read it
//! ([`quantize`], [`quantize_activation`], [`dequantize`]). The int8 matmul kernels
//! in [`crate::math`] operate on a [`QuantizedTensor`], so these primitives live in
//! the shared core rather than inside either architecture module. The Llama path's
//! weight-set quantization (which tensors are quantized, the on-disk v2 format, the
//! [`QuantizedWeights`](crate::llama::quantize::QuantizedWeights) bundle) builds on
//! these in [`crate::llama::quantize`]; the seq2seq path will reuse them too.
//!
//! The scheme is *group-wise* and *symmetric*, matching llama2.c's `runq.c`
//! (`GS` = group size): every run of `group_size` consecutive values shares one
//! `f32` scale, so quantization adapts to the local magnitude of each group rather
//! than using a single scale for a whole row. Value `k` dequantizes to
//! `data[k] as f32 * scales[k / group_size]`.

use crate::error::EngineError;

/// A group-wise int8-quantized weight matrix: 8-bit values plus per-group scales.
///
/// Element `k` dequantizes to `data[k] as f32 * scales[k / group_size]`. `data`
/// holds the quantized weights in `[d_out, d_in]` row-major layout (the same shape
/// the fp32 weight views use), and `scales` holds one `f32` per group of
/// `group_size` consecutive weights, so `scales.len() == data.len() / group_size`.
/// `group_size` is chosen to divide `d_in`, so a group never straddles two rows.
///
/// A `QuantizedTensor` typically spans *every layer* of one projection; use
/// [`QuantizedTensor::layer`] to slice out a single layer's matrix before passing it
/// to [`matmul_q8`](crate::math::matmul_q8).
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
    /// projection — the quantized analogue of slicing `w.wq[l*rows*cols..][..rows*cols]`
    /// out of an fp32 weight view.
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
/// The inverse of [`quantize`]; used by the int8 embedding lookup (one row at a time)
/// and to measure round-trip error in tests.
pub fn dequantize(out: &mut [f32], qx: &QuantizedTensor) {
    debug_assert_eq!(out.len(), qx.data.len());
    for (k, o) in out.iter_mut().enumerate() {
        *o = qx.data[k] as f32 * qx.scales[k / qx.group_size];
    }
}

/// Per-matmul activation scratch for the int8 (W8A8) path — architecture-agnostic.
///
/// Holds the quantized activation (`xq`, `i8`) plus its per-group `scales` and per-group
/// integer `gsums` (the VNNI correction term), all filled fresh by [`quantize_activation`]
/// before each int8 matmul by [`matmul_w8a8`](crate::math::matmul_w8a8). It is kept
/// separate from either architecture's `RunState` so an fp32 run carries none of it.
///
/// The three buffers are caller-owned (the arena vends only `f32`, and `xq` is `i8`), and
/// each must be at least as long as the largest matmul input dimension a forward pass will
/// feed it — `max_proj_d_in` on the Llama path, `seq2seq_max_proj_d_in` on the seq2seq
/// path. Build one only when running quantized; pass `None` to the forward pass otherwise.
#[derive(Debug)]
pub struct QuantScratch<'buf> {
    /// Quantized activation values, `i8`.
    pub xq: &'buf mut [i8],
    /// Per-group activation scales.
    pub scales: &'buf mut [f32],
    /// Per-group activation integer sums (VNNI `−128·Σa` correction).
    pub gsums: &'buf mut [f32],
}

impl<'buf> QuantScratch<'buf> {
    /// Bundle three caller-owned buffers as an activation scratch.
    ///
    /// Each buffer must be at least `need` long — the largest matmul input dimension
    /// (`d_in`) any projection will feed this scratch, with one group per element in the
    /// worst case. Returns [`EngineError::ArenaOverflow`] if any buffer is short.
    pub fn new(
        xq: &'buf mut [i8],
        scales: &'buf mut [f32],
        gsums: &'buf mut [f32],
        need: usize,
    ) -> Result<QuantScratch<'buf>, EngineError> {
        let have = xq.len().min(scales.len()).min(gsums.len());
        if have < need {
            return Err(EngineError::ArenaOverflow {
                requested: need,
                available: have,
            });
        }
        Ok(QuantScratch { xq, scales, gsums })
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
        assert_eq!(
            gsums[0],
            qa[..4].iter().map(|&v| v as i32).sum::<i32>() as f32
        );
        assert_eq!(
            gsums[1],
            qa[4..].iter().map(|&v| v as i32).sum::<i32>() as f32
        );
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
}
