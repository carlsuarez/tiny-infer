//! Scalar fp32 math kernels for the forward pass.
//!
//! Every kernel writes into caller-provided slices and allocates nothing, so they
//! are safe to call from the hot path over buffers carved from the [`Arena`]. They
//! depend on [`libm`] for the transcendental functions that `core` lacks
//! (`expf`, `powf`, `sinf`, `cosf`, `sqrtf`).
//!
//! The implementations follow Karpathy's llama2.c `run.c` operation-for-operation;
//! the details that matter for token-for-token parity at temperature 0 are called
//! out per function below.
//!
//! [`Arena`]: crate::Arena

/// Matrix–vector product `out = W · x` for a row-major weight matrix.
///
/// `w` is laid out as `[d_out, d_in]` (row `i` is `w[i*d_in .. (i+1)*d_in]`), so
/// `out[i] = Σ_j w[i*d_in + j] * x[j]`. This is the shape every projection weight
/// in the checkpoint uses, so no transpose is ever needed.
///
/// # Panics
/// In debug builds, if the slice lengths disagree with `d_in`/`d_out`.
pub fn matmul(out: &mut [f32], x: &[f32], w: &[f32], d_in: usize, d_out: usize) {
    debug_assert_eq!(out.len(), d_out);
    debug_assert_eq!(x.len(), d_in);
    debug_assert_eq!(w.len(), d_in * d_out);

    for i in 0..d_out {
        let mut sum = 0.0f32;
        for j in 0..d_in {
            sum += w[i * d_in + j] * x[j];
        }
        out[i] = sum;
    }
}

/// Root-mean-square layer norm: `out = x / √(mean(x²) + ε) ⊙ w`.
///
/// The mean of the squares is taken over the **full length** of `x`, and the gain
/// vector `w` is applied element-wise. `ε = 1e-5`, matching llama2.c — both the ε
/// value and "mean, not sum" are classic sources of parity drift, so they are
/// fixed here.
///
/// # Panics
/// In debug builds, if `out`, `x`, and `w` differ in length.
pub fn rmsnorm(out: &mut [f32], x: &[f32], w: &[f32]) {
    const EPS: f32 = 1e-5;

    debug_assert_eq!(out.len(), x.len());
    debug_assert_eq!(x.len(), w.len());

    let ss = x.iter().map(|&v| v * v).sum::<f32>() / x.len() as f32;
    let scale = 1.0f32 / libm::sqrtf(ss + EPS);

    for ((o, xi), wi) in out.iter_mut().zip(x.iter()).zip(w.iter()) {
        *o = xi * scale * wi;
    }
}

/// In-place numerically-stable softmax over `x`.
///
/// Subtracts the max before exponentiating (so the largest term is `e^0 = 1` and
/// nothing overflows), then normalizes to sum to 1. Used for the attention
/// weights. (The host's temperature sampler applies the same max-shift trick over
/// the logits, but inline, since it only borrows them immutably.)
pub fn softmax(x: &mut [f32]) {
    let max_val = maxf(x);

    // Exponentiate the max-shifted values and accumulate their sum.
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = libm::expf(*v - max_val);
        sum += *v;
    }

    // Normalize to a probability distribution.
    for v in x.iter_mut() {
        *v /= sum;
    }
}

/// SiLU / swish activation: `z * σ(z) = z / (1 + e^-z)`.
///
/// This is the gate non-linearity of the SwiGLU feed-forward block.
pub fn silu(z: f32) -> f32 {
    z / (1.0 + libm::expf(-z))
}

/// Apply rotary position embeddings (RoPE) in place to `q` and `k` at `pos`.
///
/// Walks the query in pairs (`i` stepping by 2 across `dim`). For each pair the
/// rotation angle is `pos · 10000^(-(i mod head_size)/head_size)`, so the
/// frequency depends only on the position *within* a head and repeats per head.
/// Each `(v[i], v[i+1])` pair is rotated by that angle.
///
/// The query is rotated across the full `dim`; the key is rotated **only while
/// `i < kv_dim`**. This mirrors llama2.c's `rotn = i < kv_dim ? 2 : 1`, which
/// matters when `n_kv_heads < n_heads` (grouped-query attention) and is a common
/// parity pitfall.
///
/// # Panics
/// In debug builds, if `q` is shorter than `dim` or `k` shorter than `kv_dim`.
pub fn rope(q: &mut [f32], k: &mut [f32], pos: usize, head_size: usize, dim: usize, kv_dim: usize) {
    debug_assert!(q.len() >= dim);
    debug_assert!(k.len() >= kv_dim);

    let mut i = 0;
    while i < dim {
        let head_dim = (i % head_size) as f32;
        let freq = 1.0 / libm::powf(10000.0, head_dim / head_size as f32);
        let angle = pos as f32 * freq;
        let cos = libm::cosf(angle);
        let sin = libm::sinf(angle);

        // Rotate the query across the full dimension.
        let q0 = q[i];
        let q1 = q[i + 1];
        q[i] = q0 * cos - q1 * sin;
        q[i + 1] = q0 * sin + q1 * cos;

        // Rotate the key only within kv_dim (GQA: fewer key heads than query heads).
        if i < kv_dim {
            let k0 = k[i];
            let k1 = k[i + 1];
            k[i] = k0 * cos - k1 * sin;
            k[i + 1] = k0 * sin + k1 * cos;
        }

        i += 2;
    }
}

/// Return the maximum element of `buf`.
///
/// A small helper used by [`softmax`] for its max-shift. Unlike `Iterator::max`,
/// this works directly on `f32` (which is only `PartialOrd`): it seeds with
/// `buf[0]` and keeps any element that compares strictly greater. A `NaN` later in
/// the slice is skipped (`val > max` is false), but a `NaN` in `buf[0]` seeds the
/// running max and is returned — matching llama2.c, which also seeds from `x[0]`.
/// Real inference inputs are finite, so this edge case never arises in practice.
///
/// # Panics
/// In debug builds, if `buf` is empty.
pub fn maxf(buf: &[f32]) -> f32 {
    debug_assert!(!buf.is_empty());

    let mut max = buf[0];
    for &val in buf.iter() {
        if val > max {
            max = val;
        }
    }
    max
}

/// Add `src` into `dst` element-wise: `dst[i] += src[i]`.
///
/// The residual connections of the transformer (`x += attn_out`, `x += ffn_out`)
/// are exactly this.
///
/// # Panics
/// In debug builds, if `dst` and `src` differ in length.
pub fn accumulate(dst: &mut [f32], src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d += *s;
    }
}

/// Index of the maximum element, taking the **first** on ties.
///
/// This is greedy (temperature-0) decoding: pick the most probable next token.
/// The first-on-ties rule (strict `>`) matches llama2.c's `argmax`, which keeps
/// the earliest index — important for token-for-token parity.
///
/// # Panics
/// In debug builds, if `buf` is empty.
pub fn argmax(buf: &[f32]) -> usize {
    debug_assert!(!buf.is_empty());

    let mut best = 0;
    let mut best_val = buf[0];
    for (i, &val) in buf.iter().enumerate() {
        if val > best_val {
            best_val = val;
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    // Approximate float comparison for kernel outputs.
    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn matmul_matches_hand_computation() {
        // W = [[1, 2, 3], [4, 5, 6]]  (d_out=2, d_in=3), x = [1, 0, -1]
        let w = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = [1.0, 0.0, -1.0];
        let mut out = [0.0; 2];
        matmul(&mut out, &x, &w, 3, 2);
        // row0: 1*1 + 2*0 + 3*-1 = -2 ; row1: 4*1 + 5*0 + 6*-1 = -2
        assert!(close(out[0], -2.0));
        assert!(close(out[1], -2.0));
    }

    #[test]
    fn rmsnorm_normalizes_to_unit_rms_with_unit_gain() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let g = [1.0, 1.0, 1.0, 1.0];
        let mut out = [0.0; 4];
        rmsnorm(&mut out, &x, &g);
        // mean(x²) = (1+4+9+16)/4 = 7.5 ; scale = 1/sqrt(7.5 + 1e-5)
        let scale = 1.0 / (7.5f32 + 1e-5).sqrt();
        for (o, xi) in out.iter().zip(x.iter()) {
            assert!(close(*o, xi * scale));
        }
    }

    #[test]
    fn rmsnorm_applies_gain() {
        let x = [2.0, 2.0];
        let g = [3.0, 0.5];
        let mut out = [0.0; 2];
        rmsnorm(&mut out, &x, &g);
        // mean(x²) = 4, scale = 1/sqrt(4 + 1e-5) ≈ 0.5
        let scale = 1.0 / (4.0f32 + 1e-5).sqrt();
        assert!(close(out[0], 2.0 * scale * 3.0));
        assert!(close(out[1], 2.0 * scale * 0.5));
    }

    #[test]
    fn softmax_is_a_distribution_and_shift_invariant() {
        let mut a = [1.0, 2.0, 3.0];
        let mut b = [101.0, 102.0, 103.0]; // shifted by a constant
        softmax(&mut a);
        softmax(&mut b);
        let sum: f32 = a.iter().sum();
        assert!(close(sum, 1.0));
        assert!(a.iter().all(|&p| p > 0.0));
        // Softmax is invariant to a constant offset; the big-valued input must not overflow.
        for (x, y) in a.iter().zip(b.iter()) {
            assert!(close(*x, *y));
        }
    }

    #[test]
    fn silu_known_values() {
        assert!(close(silu(0.0), 0.0)); // 0 * sigmoid(0)
        // silu(1) = 1 / (1 + e^-1) ≈ 0.7310586
        assert!(close(silu(1.0), 0.731_058_6));
        // Large positive ~ identity, large negative ~ 0.
        assert!(silu(20.0) > 19.99);
        assert!(silu(-20.0).abs() < 1e-6);
    }

    #[test]
    fn rope_at_pos_zero_is_identity() {
        // angle = 0 for every pair, so cos=1, sin=0 → no change.
        let mut q = [1.0, 2.0, 3.0, 4.0];
        let mut k = [5.0, 6.0, 7.0, 8.0];
        rope(&mut q, &mut k, 0, 2, 4, 4);
        assert_eq!(q, [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(k, [5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn rope_rotates_first_pair_by_pos_times_angle() {
        // head_size=2, dim=2, kv_dim=2, pos=1 → head_dim=0, freq=1, angle=1 rad.
        let mut q = [1.0, 0.0];
        let mut k = [1.0, 0.0];
        rope(&mut q, &mut k, 1, 2, 2, 2);
        // (1,0) rotated by 1 rad → (cos1, sin1)
        assert!(close(q[0], 1.0f32.cos()));
        assert!(close(q[1], 1.0f32.sin()));
        assert!(close(k[0], 1.0f32.cos()));
        assert!(close(k[1], 1.0f32.sin()));
    }

    #[test]
    fn rope_leaves_key_untouched_past_kv_dim() {
        // dim=4 but kv_dim=2: the second pair (i=2) rotates q but not k.
        let mut q = [1.0, 0.0, 1.0, 0.0];
        let mut k = [1.0, 0.0, 9.0, 9.0]; // only first 2 are "real" key data
        rope(&mut q, &mut k, 1, 2, 4, 2);
        // k[2], k[3] must be untouched (i=2 is >= kv_dim).
        assert_eq!(k[2], 9.0);
        assert_eq!(k[3], 9.0);
        // q's second pair WAS rotated (head_dim = 2 % 2 = 0, angle = 1 rad).
        assert!(close(q[2], 1.0f32.cos()));
        assert!(close(q[3], 1.0f32.sin()));
    }

    #[test]
    fn maxf_finds_max_ignoring_later_nan() {
        assert!(close(maxf(&[1.0, 5.0, -3.0, 2.0]), 5.0));
        // A NaN after the first element is skipped (`val > max` is false).
        assert!(close(maxf(&[1.0, f32::NAN, 2.0]), 2.0));
        // Single element returns itself.
        assert!(close(maxf(&[42.0]), 42.0));
    }

    #[test]
    fn accumulate_adds_in_place() {
        let mut dst = [1.0, 2.0, 3.0];
        accumulate(&mut dst, &[10.0, 20.0, 30.0]);
        assert_eq!(dst, [11.0, 22.0, 33.0]);
    }

    #[test]
    fn argmax_picks_max_and_first_on_ties() {
        assert_eq!(argmax(&[0.1, 0.7, 0.2]), 1);
        assert_eq!(argmax(&[5.0]), 0);
        // Ties resolve to the earliest index (strict `>`).
        assert_eq!(argmax(&[2.0, 2.0, 1.0, 2.0]), 0);
        assert_eq!(argmax(&[-3.0, -1.0, -1.0]), 1);
    }
}
