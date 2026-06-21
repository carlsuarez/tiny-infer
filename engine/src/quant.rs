//! Group-wise int8 quantization **primitives** — the shared building blocks.
//!
//! These are architecture-agnostic: a [`QuantizedTensor`] view over an `i8` data
//! buffer plus its `f32` scales, and the routines that fill and read it
//! ([`quantize`], [`quantize_activation`], [`dequantize`]). The int8 matmul kernels
//! in [`crate::math`] operate on a [`QuantizedTensor`], so these primitives live in
//! the shared core rather than inside either architecture module. The Llama path's
//! weight-set quantization (which tensors are quantized, the on-disk v2 format, the
//! `QuantizedWeights` bundle) builds on
//! these in `llama::quantize`; the seq2seq path will reuse them too.
//!
//! The scheme is *group-wise* and *symmetric* (`GS` = group size): every run of
//! `group_size` consecutive values shares one
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
/// Symmetric per-group quantization: for each group of
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

/// Saturating rounding doubling high-multiply.
///
/// Computes `round(a * b / 2^31)` in `i64` and narrows to `i32`, the building block of the
/// fixed-point requantization used by integer-only inference. The lone saturating case is
/// `i32::MIN * i32::MIN`, which would overflow `i32::MAX`.
fn sat_round_doubling_high_mul(a: i32, b: i32) -> i32 {
    if a == i32::MIN && b == i32::MIN {
        return i32::MAX;
    }
    let ab = a as i64 * b as i64;
    // Round to nearest, ties away from zero. Note this is integer *division* (truncating
    // toward zero), not an arithmetic shift (flooring) — they differ for negative values.
    let nudge: i64 = if ab >= 0 { 1 << 30 } else { 1 - (1 << 30) };
    ((ab + nudge) / (1i64 << 31)) as i32
}

/// Round-to-nearest right shift by `exponent` bits (rounding divide by a power of two).
fn rounding_divide_by_pot(x: i32, exponent: i32) -> i32 {
    if exponent == 0 {
        return x;
    }
    let mask = (1i32 << exponent) - 1;
    let remainder = x & mask;
    let threshold = (mask >> 1) + i32::from(x < 0);
    (x >> exponent) + i32::from(remainder > threshold)
}

/// Multiply an `i32` accumulator by a quantized multiplier `mult * 2^shift` — the
/// integer-only **requantization** step (multiply by a quantized multiplier).
///
/// `mult` is a Q31 fixed-point mantissa in `[2^30, 2^31)` and `shift` is the signed binary
/// exponent, exactly as produced offline by quantizing a real scale
/// ratio `M = (s_in · s_w) / s_out` into a mantissa + shift. The result is `round(x · M)`. This is how an integer
/// matmul/conv rescales its `i32` accumulator into the next layer's int8 domain **without
/// any floating point** — the one operation that makes "integer-only" inference integer.
pub fn requantize(x: i32, mult: i32, shift: i32) -> i32 {
    let left = if shift > 0 { shift } else { 0 };
    let right = if shift > 0 { 0 } else { -shift };
    let x = if left > 0 { x.wrapping_shl(left as u32) } else { x };
    rounding_divide_by_pot(sat_round_doubling_high_mul(x, mult), right)
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

    // Reference `QuantizeMultiplier`: split a real M>0 into a Q31 mantissa in [2^30,2^31)
    // and a signed shift, matching the offline routine the export tool uses.
    fn quantize_multiplier(m: f64) -> (i32, i32) {
        if m == 0.0 {
            return (0, 0);
        }
        let (mut frac, mut shift) = {
            // frexp: m = frac * 2^exp, frac in [0.5, 1)
            let exp = m.abs().log2().floor() as i32 + 1;
            (m / (2f64).powi(exp), exp)
        };
        let mut q = (frac * (1i64 << 31) as f64).round() as i64;
        if q == (1i64 << 31) {
            q /= 2;
            shift += 1;
        }
        let _ = &mut frac;
        (q as i32, shift)
    }

    #[test]
    fn requantize_approximates_multiply_by_real_scale() {
        // requantize(x, QuantizeMultiplier(M)) must equal round(x*M) to within one unit.
        for &m in &[0.0009f64, 0.0123, 0.25, 0.5, 0.7777, 1.5, 3.25] {
            let (mult, shift) = quantize_multiplier(m);
            for &x in &[0i32, 1, -1, 7, -7, 1000, -1000, 32768, -32768, 1_000_000] {
                let got = requantize(x, mult, shift);
                let want = (x as f64 * m).round() as i32;
                assert!(
                    (got - want).abs() <= 1,
                    "M={m} x={x}: requantize={got} want={want}"
                );
            }
        }
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
