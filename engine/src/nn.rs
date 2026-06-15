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
