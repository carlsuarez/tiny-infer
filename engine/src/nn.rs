//! 1-D convolutional-network kernels: `Conv1D`, `ReLU`, and global average pooling.
//!
//! These are the building blocks of a small **1-D CNN** forward pass — the family of
//! model the on-device predictive-maintenance node (`edge-pm`) runs over a window of
//! extracted signal features. Like the rest of the shared core they are independent of
//! any one model shape: each writes into caller-provided slices, allocates nothing, and
//! depends only on `core`, so they are safe to call from the hot path over buffers
//! carved from the [`Arena`]. They sit beside the transformer kernels in [`math`] rather
//! than inside it because they are a separate op family; the *model* that wires them into
//! a bearing-health classifier lives in the host application, not the engine.
//!
//! ## Tensor layout (matches a PyTorch export)
//!
//! Everything is row-major and channel-major, exactly as `torch.nn.Conv1d` lays its
//! tensors out, so a checkpoint exported from PyTorch loads without a transpose:
//!
//! * a **signal** of `c` channels and length `l` is `[c, l]` — channel `i` occupies the
//!   contiguous run `x[i*l .. (i+1)*l]`;
//! * a **Conv1D weight** is `[c_out, c_in, k]` — output channel `o`'s tap `j` for input
//!   channel `i` is `w[(o*c_in + i)*k + j]`;
//! * a **bias** (optional, mirroring `Conv1d(bias=…)`) is `[c_out]`.
//!
//! ## Scope
//!
//! [`conv1d`] implements the common case: **valid** (zero) padding, `dilation = 1`, and
//! `groups = 1`. That covers a standard feature-window classifier; strided/dilated or
//! grouped convolutions would extend the indexing here if a future model needs them.
//!
//! [`Arena`]: crate::Arena
//! [`math`]: crate::math

use crate::quant::requantize;

/// Output length of a [`conv1d`] over an input of length `l_in`.
///
/// `(l_in - k) / stride + 1`, the valid-padding (`padding = 0`, `dilation = 1`) formula
/// `torch.nn.Conv1d` uses. A caller sizes the output buffer — and the model its arena —
/// with this. `k <= l_in` and `stride >= 1` are preconditions (otherwise the subtraction
/// underflows); [`conv1d`] checks them in debug builds.
pub const fn conv1d_out_len(l_in: usize, k: usize, stride: usize) -> usize {
    (l_in - k) / stride + 1
}

/// 1-D convolution `out = conv(input, weight) + bias` — valid padding, stride `stride`.
///
/// Slides each of the `c_out` kernels (length `k`, spanning all `c_in` input channels)
/// across the `[c_in, l_in]` input and writes a `[c_out, l_out]` result, where
/// `l_out = `[`conv1d_out_len`]`(l_in, k, stride)`. The value at output channel `o`,
/// position `t` is
///
/// ```text
/// out[o, t] = bias[o] + Σ_i Σ_j weight[o, i, j] · input[i, t·stride + j]
/// ```
///
/// summed over every input channel `i` and tap `j ∈ 0..k`. With `bias = None` the bias
/// term is zero (a `Conv1d(bias=false)` layer). The inner `Σ_j` is the dot product the
/// transformer [`matmul`](crate::math::matmul) reuses; this is the readable scalar
/// reference (correctness before speed — a SIMD twin can come later if a window proves
/// too slow on-device).
///
/// See the [module docs](self) for the row-major `[c, l]` / `[c_out, c_in, k]` layout.
///
/// # Panics
/// In debug builds, if any slice length disagrees with the dimensions, if `bias` is the
/// wrong length, or if `k`/`stride` are out of range (`k == 0`, `k > l_in`, `stride == 0`).
#[allow(clippy::too_many_arguments)]
pub fn conv1d(
    out: &mut [f32],
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    c_in: usize,
    c_out: usize,
    l_in: usize,
    k: usize,
    stride: usize,
) {
    debug_assert!(k >= 1, "kernel length must be at least 1");
    debug_assert!(stride >= 1, "stride must be at least 1");
    debug_assert!(k <= l_in, "kernel longer than the input");
    let l_out = conv1d_out_len(l_in, k, stride);
    debug_assert_eq!(out.len(), c_out * l_out);
    debug_assert_eq!(input.len(), c_in * l_in);
    debug_assert_eq!(weight.len(), c_out * c_in * k);
    debug_assert!(bias.is_none_or(|b| b.len() == c_out));

    for o in 0..c_out {
        let bias_o = bias.map_or(0.0, |b| b[o]);
        let out_row = &mut out[o * l_out..][..l_out];
        for (t, slot) in out_row.iter_mut().enumerate() {
            let start = t * stride;
            let mut sum = bias_o;
            for i in 0..c_in {
                let in_row = &input[i * l_in..][..l_in];
                let kernel = &weight[(o * c_in + i) * k..][..k];
                for (j, &wj) in kernel.iter().enumerate() {
                    sum += wj * in_row[start + j];
                }
            }
            *slot = sum;
        }
    }
}

/// Integer-only 1-D convolution — the quantized twin of [`conv1d`] for static int8 inference.
///
/// Every operand and the output are int8: a `[c_in, l_in]` int8 input (channel-major, like
/// [`conv1d`]), an int8 `[c_out, c_in, k]` weight, an `i32` `bias` (pre-scaled to the
/// accumulator's `s_in · s_w[o]` domain), and a **per-output-channel** requantizer
/// (`mult[o]`, `shift[o]`). Each output's `c_in * k`-term dot product accumulates in **`i32`**
/// (exact), the `i32` bias is folded in, and the accumulator is rescaled to the next layer's
/// int8 domain by [`requantize`](crate::quant::requantize) — **no floating point anywhere**.
/// The result is clamped to `[out_min, out_max]`; pass `out_min = 0` to fuse a following
/// `ReLU` (the activation never leaves the int8 domain between layers).
///
/// This is the standard integer-only convolution: weights and activations both
/// quantized, `i32` accumulation, fixed-point requantization. See the [module docs](self) for
/// the `[c, l]` / `[c_out, c_in, k]` layout.
///
/// # Panics
/// In debug builds, if any slice length disagrees with the dimensions, if `bias`/`mult`/
/// `shift` are not `c_out` long, or if `k`/`stride` are out of range.
#[allow(clippy::too_many_arguments)]
pub fn conv1d_i8(
    out: &mut [i8],
    input: &[i8],
    weight: &[i8],
    bias: &[i32],
    mult: &[i32],
    shift: &[i32],
    out_min: i8,
    out_max: i8,
    c_in: usize,
    c_out: usize,
    l_in: usize,
    k: usize,
    stride: usize,
) {
    debug_assert!(k >= 1, "kernel length must be at least 1");
    debug_assert!(stride >= 1, "stride must be at least 1");
    debug_assert!(k <= l_in, "kernel longer than the input");
    let l_out = conv1d_out_len(l_in, k, stride);
    debug_assert_eq!(out.len(), c_out * l_out);
    debug_assert_eq!(input.len(), c_in * l_in);
    debug_assert_eq!(weight.len(), c_out * c_in * k);
    debug_assert_eq!(bias.len(), c_out);
    debug_assert_eq!(mult.len(), c_out);
    debug_assert_eq!(shift.len(), c_out);

    for o in 0..c_out {
        let (m, s, b) = (mult[o], shift[o], bias[o]);
        let out_row = &mut out[o * l_out..][..l_out];
        for (t, slot) in out_row.iter_mut().enumerate() {
            let start = t * stride;
            let mut acc: i32 = b;
            for i in 0..c_in {
                let in_row = &input[i * l_in..][..l_in];
                let kernel = &weight[(o * c_in + i) * k..][..k];
                for (j, &wj) in kernel.iter().enumerate() {
                    acc += wj as i32 * in_row[start + j] as i32;
                }
            }
            let q = requantize(acc, m, s).clamp(out_min as i32, out_max as i32);
            *slot = q as i8;
        }
    }
}

/// Rectified linear unit, in place: `x[i] = max(0, x[i])`.
///
/// Written as a branch rather than `f32::max` so the kernel leans only on `core`'s
/// comparison, keeping the engine clear of `std`-only float intrinsics (the same reason
/// [`math`](crate::math) routes its transcendentals through `libm`). A `NaN` compares
/// `false` and so passes through unchanged.
pub fn relu(x: &mut [f32]) {
    for v in x.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

/// Global average pooling over the time axis: `[c, l] → [c]`.
///
/// Collapses each channel of a `[c, l]` signal to its mean across all `l` positions,
/// writing one value per channel into `out`. This is the bridge a 1-D CNN uses between
/// the convolutional stack and the dense classifier head — it turns a variable-length
/// feature map into a fixed-size vector.
///
/// # Panics
/// In debug builds, if `out` is not `c` long, `input` is not `c*l` long, or `l == 0`.
pub fn global_avg_pool(out: &mut [f32], input: &[f32], c: usize, l: usize) {
    debug_assert_eq!(out.len(), c);
    debug_assert_eq!(input.len(), c * l);
    debug_assert!(l >= 1, "cannot average an empty time axis");

    let inv = 1.0 / l as f32;
    for (ch, slot) in out.iter_mut().enumerate() {
        let row = &input[ch * l..][..l];
        *slot = row.iter().sum::<f32>() * inv;
    }
}

/// Integer-only global average pooling: `[c, l] → [c]`, int8 → int8.
///
/// Sums each channel's `l` int8 values in `i32`, then a single (per-tensor) requantizer
/// rescales the sum to the output int8 domain — the **division by `l`** is folded into
/// `mult`/`shift` (`M = s_in / (l · s_out)`), so the kernel itself is pure integer. The
/// result is clamped to `[out_min, out_max]`.
///
/// # Panics
/// In debug builds, if `out` is not `c` long, `input` is not `c*l` long, or `l == 0`.
#[allow(clippy::too_many_arguments)]
pub fn global_avg_pool_i8(
    out: &mut [i8],
    input: &[i8],
    c: usize,
    l: usize,
    mult: i32,
    shift: i32,
    out_min: i8,
    out_max: i8,
) {
    debug_assert_eq!(out.len(), c);
    debug_assert_eq!(input.len(), c * l);
    debug_assert!(l >= 1, "cannot average an empty time axis");

    for (ch, slot) in out.iter_mut().enumerate() {
        let row = &input[ch * l..][..l];
        let sum: i32 = row.iter().map(|&v| v as i32).sum();
        let q = requantize(sum, mult, shift).clamp(out_min as i32, out_max as i32);
        *slot = q as i8;
    }
}

/// Integer-only matrix–vector product `out = W · x` into `i32` accumulators (no bias).
///
/// The integer-only dense kernel: an int8 `[d_out, d_in]` weight (row-major) times an int8
/// activation, accumulated exactly in `i32`. It does **not** requantize or add a bias — it is
/// meant for the final classifier, whose `i32` weight-dot logits the caller dequantizes once
/// per output channel (`acc · s_in · s_w[o]`) and to which it then adds the `f32` bias before
/// the closing softmax. Deferring the bias to that `f32` step keeps it exact and, unlike an
/// `i32` bias in the `s_in · s_w[o]` accumulator domain, well-defined even for an all-zero
/// weight row (`s_w[o] = 0`). That dequantize is the only floating point in an otherwise
/// integer pass.
///
/// # Panics
/// In debug builds, if any slice length disagrees with `d_in`/`d_out`.
pub fn matmul_i8(out: &mut [i32], input: &[i8], weight: &[i8], d_in: usize, d_out: usize) {
    debug_assert_eq!(out.len(), d_out);
    debug_assert_eq!(input.len(), d_in);
    debug_assert_eq!(weight.len(), d_in * d_out);

    for (o, slot) in out.iter_mut().enumerate() {
        let row = &weight[o * d_in..][..d_in];
        let mut acc = 0i32;
        for (j, &w) in row.iter().enumerate() {
            acc += w as i32 * input[j] as i32;
        }
        *slot = acc;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    extern crate std;
    use std::vec;

    // Approximate float comparison for kernel outputs.
    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn conv1d_out_len_arithmetic() {
        assert_eq!(conv1d_out_len(10, 3, 1), 8);
        assert_eq!(conv1d_out_len(10, 3, 2), 4); // (10-3)/2 + 1 = 3 + 1
        assert_eq!(conv1d_out_len(5, 2, 2), 2);
        assert_eq!(conv1d_out_len(7, 7, 1), 1); // kernel spans the whole input
        assert_eq!(conv1d_out_len(512, 8, 4), 127);
    }

    #[test]
    fn conv1d_matches_hand_computation() {
        // One channel in/out, kernel [1,2,3] over [1,2,3,4,5], stride 1, no bias.
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let weight = [1.0, 2.0, 3.0];
        let mut out = [0.0; 3]; // l_out = (5-3)/1 + 1 = 3
        conv1d(&mut out, &input, &weight, None, 1, 1, 5, 3, 1);
        // out[t] = x[t] + 2·x[t+1] + 3·x[t+2]
        assert!(close(out[0], 1.0 + 4.0 + 9.0)); // 14
        assert!(close(out[1], 2.0 + 6.0 + 12.0)); // 20
        assert!(close(out[2], 3.0 + 8.0 + 15.0)); // 26
    }

    #[test]
    fn conv1d_honors_stride_and_bias() {
        // l_in=5, k=2, stride=2 ⇒ l_out=2, sampling windows at offsets 0 and 2.
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let weight = [1.0, 1.0]; // sum of the pair
        let bias = [10.0];
        let mut out = [0.0; 2];
        conv1d(&mut out, &input, &weight, Some(&bias), 1, 1, 5, 2, 2);
        assert!(close(out[0], 10.0 + 1.0 + 2.0)); // window [1,2] → 13
        assert!(close(out[1], 10.0 + 3.0 + 4.0)); // window [3,4] → 17
    }

    #[test]
    fn conv1d_mixes_channels_in_weight_order() {
        // c_in=2, c_out=2, l_in=3, k=2, stride=1 ⇒ l_out=2. Verifies the [c_out,c_in,k]
        // weight layout and the channel-major [c,l] input layout line up.
        let input = [1.0, 2.0, 3.0, /* ch1 */ 4.0, 5.0, 6.0];
        // oc0: ic0=[1,0], ic1=[0,1] (pass tap-0 of ch0 + tap-1 of ch1)
        // oc1: ic0=[1,1], ic1=[1,1] (sum every tap of both channels)
        let weight = [1.0, 0.0, 0.0, 1.0, /* oc1 */ 1.0, 1.0, 1.0, 1.0];
        let mut out = [0.0; 4];
        conv1d(&mut out, &input, &weight, None, 2, 2, 3, 2, 1);
        // oc0: t0 = 1·1 + 1·5 = 6 ; t1 = 1·2 + 1·6 = 8
        assert!(close(out[0], 6.0));
        assert!(close(out[1], 8.0));
        // oc1: t0 = (1+2) + (4+5) = 12 ; t1 = (2+3) + (5+6) = 16
        assert!(close(out[2], 12.0));
        assert!(close(out[3], 16.0));
    }

    fn max_abs(x: &[f32]) -> f32 {
        x.iter().fold(0.0f32, |m, &v| m.max(v.abs()))
    }

    // Reference QuantizeMultiplier (offline): real M>0 → (Q31 mantissa, signed shift).
    fn quantize_multiplier(m: f64) -> (i32, i32) {
        if m == 0.0 {
            return (0, 0);
        }
        let exp = m.abs().log2().floor() as i32 + 1;
        let frac = m / (2f64).powi(exp);
        let mut q = (frac * (1i64 << 31) as f64).round() as i64;
        let mut shift = exp;
        if q == (1i64 << 31) {
            q /= 2;
            shift += 1;
        }
        (q as i32, shift)
    }

    fn q_i8(v: f32, scale: f32) -> i8 {
        if scale == 0.0 {
            0
        } else {
            (v / scale).round().clamp(-127.0, 127.0) as i8
        }
    }

    #[test]
    fn conv1d_i8_tracks_fp32() {
        // The integer-only conv must reproduce the fp32 conv up to quantization error.
        // c_in=2, c_out=3, l_in=8, k=3, stride=2 ⇒ l_out=3.
        let (c_in, c_out, l_in, k, stride) = (2usize, 3usize, 8usize, 3usize, 2usize);
        let l_out = conv1d_out_len(l_in, k, stride);

        let input: std::vec::Vec<f32> = (0..c_in * l_in).map(|i| libm::sinf(i as f32 * 0.37)).collect();
        let weight: std::vec::Vec<f32> =
            (0..c_out * c_in * k).map(|i| libm::cosf(i as f32 * 0.21) * 0.5).collect();
        let bias: std::vec::Vec<f32> = (0..c_out).map(|o| o as f32 * 0.05).collect();

        let mut out_f = vec![0.0f32; c_out * l_out];
        conv1d(&mut out_f, &input, &weight, Some(&bias), c_in, c_out, l_in, k, stride);

        // Static quantization: per-tensor activation scale, per-output-channel weight scale,
        // a calibrated output scale, i32 bias, and per-channel requant multipliers.
        let s_x = max_abs(&input) / 127.0;
        let xq: std::vec::Vec<i8> = input.iter().map(|&v| q_i8(v, s_x)).collect();

        let gs = c_in * k;
        let mut wq = vec![0i8; weight.len()];
        let mut s_w = vec![0.0f32; c_out];
        for o in 0..c_out {
            let row = &weight[o * gs..][..gs];
            s_w[o] = max_abs(row) / 127.0;
            for j in 0..gs {
                wq[o * gs + j] = q_i8(row[j], s_w[o]);
            }
        }

        let s_out = max_abs(&out_f) / 127.0;
        let bias_i: std::vec::Vec<i32> =
            (0..c_out).map(|o| (bias[o] / (s_x * s_w[o])).round() as i32).collect();
        let mut mult = vec![0i32; c_out];
        let mut shift = vec![0i32; c_out];
        for o in 0..c_out {
            let (m, s) = quantize_multiplier((s_x * s_w[o] / s_out) as f64);
            mult[o] = m;
            shift[o] = s;
        }

        let mut out_q = vec![0i8; c_out * l_out];
        conv1d_i8(
            &mut out_q, &xq, &wq, &bias_i, &mult, &shift, -127, 127, c_in, c_out, l_in, k, stride,
        );

        // Dequantize the int8 output and compare to fp32.
        for (a, &qi) in out_f.iter().zip(&out_q) {
            let deq = qi as f32 * s_out;
            assert!((a - deq).abs() < 0.05, "fp32 {a} vs int {deq}");
        }
    }

    #[test]
    fn matmul_i8_is_exact_integer_dot() {
        // out[o] = Σ_j w[o,j]·x[j], exactly in i32 (bias is added later, in f32).
        let x = [1i8, -2, 3];
        let w = [1i8, 0, -1, /* row1 */ 2, 2, 2];
        let mut out = [0i32; 2];
        matmul_i8(&mut out, &x, &w, 3, 2);
        // row0·x = 1·1 + 0·(-2) + (-1)·3 = -2
        assert_eq!(out[0], -2);
        // row1·x = 2·1 + 2·(-2) + 2·3 = 4
        assert_eq!(out[1], 4);
    }

    #[test]
    fn global_avg_pool_i8_sums_and_requantizes() {
        // c=2, l=4. With mult=2^30, shift=0 the requantizer multiplies by ~0.5, so each
        // channel's int8 sum is halved (round-to-nearest).
        let input = [10i8, 10, 10, 10, /* ch1 */ -8, -8, -8, -8];
        let mut out = [0i8; 2];
        // M ≈ 0.5 ⇒ mantissa 2^30, shift 0.
        global_avg_pool_i8(&mut out, &input, 2, 4, 1 << 30, 0, -127, 127);
        assert_eq!(out[0], 20); // round(40 * 0.5)
        assert_eq!(out[1], -16); // round(-32 * 0.5)
    }

    #[test]
    fn relu_clamps_negatives_and_passes_positives() {
        let mut x = [-1.0, 0.0, 2.5, -3.5, 4.0];
        relu(&mut x);
        assert_eq!(x, [0.0, 0.0, 2.5, 0.0, 4.0]);
    }

    #[test]
    fn global_avg_pool_averages_each_channel() {
        // c=2, l=3: channel means 2 and 5.
        let input = [1.0, 2.0, 3.0, /* ch1 */ 4.0, 5.0, 6.0];
        let mut out = [0.0; 2];
        global_avg_pool(&mut out, &input, 2, 3);
        assert!(close(out[0], 2.0));
        assert!(close(out[1], 5.0));
    }

    #[test]
    fn conv_relu_pool_composes_into_a_pipeline() {
        // A miniature CNN block: conv (1→2 channels) → relu → global average pool,
        // exercising the three kernels in the order edge-pm will call them.
        let input = [1.0, -2.0, 3.0, -4.0, 5.0]; // 1 channel, length 5
        // oc0 = identity-ish [1,0,0]; oc1 = difference [-1,0,1]
        let weight = [1.0, 0.0, 0.0, /* oc1 */ -1.0, 0.0, 1.0];
        let l_out = conv1d_out_len(5, 3, 1); // 3
        let mut conv = vec![0.0f32; 2 * l_out];
        conv1d(&mut conv, &input, &weight, None, 1, 2, 5, 3, 1);
        // oc0 = [x0, x1, x2] = [1, -2, 3] ; oc1 = [x2-x0, x3-x1, x4-x2] = [2, -2, 2]
        assert!(close(conv[0], 1.0) && close(conv[1], -2.0) && close(conv[2], 3.0));
        assert!(close(conv[3], 2.0) && close(conv[4], -2.0) && close(conv[5], 2.0));

        relu(&mut conv);
        // oc0 → [1, 0, 3] ; oc1 → [2, 0, 2]
        let mut pooled = [0.0f32; 2];
        global_avg_pool(&mut pooled, &conv, 2, l_out);
        assert!(close(pooled[0], 4.0 / 3.0)); // mean(1,0,3)
        assert!(close(pooled[1], 4.0 / 3.0)); // mean(2,0,2)
    }
}
