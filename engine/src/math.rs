//! Math kernels for the forward pass.
//!
//! Every kernel writes into caller-provided slices and allocates nothing, so they
//! are safe to call from the hot path over buffers carved from the [`Arena`]. They
//! depend on [`libm`] for the transcendental functions that `core` lacks
//! (`expf`, `powf`, `sinf`, `cosf`, `sqrtf`).
//!
//! The matrix–vector products — the bulk of the compute — come in two flavours: a
//! straightforward **scalar** version and a **SIMD** version built on `core::simd`
//! (the `portable_simd` feature). Both fp32 ([`matmul`] / [`matmul_simd`]) and
//! int8-quantized ([`matmul_q8`] / [`matmul_q8_simd`]) matmuls have both. The scalar
//! kernels are the readable reference; the SIMD kernels widen the inner dot product
//! to 8 lanes of `f32` and fall back to scalar for any tail that does not fill a lane.
//! The fp32 pair agree up to floating-point reassociation; the int8 pair agree
//! **exactly** (integer accumulation is associative). The caller
//! ([`crate::ModelWeights::matmul`]) picks which via [`crate::Kernel`].
//!
//! The int8 matmul additionally has a hardware **dot-product** kernel, selected by
//! [`Kernel::Dotprod`](crate::Kernel): x86 AVX-512 VNNI (`vpdpbusd`) and ARM NEON
//! `sdot`, both [`matmul_q8_dotprod`] (one per `cfg(target_arch)`). They run the same
//! exact integer dot product as the scalar kernel, just far faster, and the std host
//! only selects them after detecting the CPU feature at runtime.
//!
//! The remaining kernels follow Karpathy's llama2.c `run.c` operation-for-operation;
//! the details that matter for token-for-token parity at temperature 0 are called
//! out per function below.
//!
//! [`Arena`]: crate::Arena

use crate::quantize::QuantizedTensor;
use core::simd::num::{SimdFloat, SimdInt};
use core::simd::{f32x8, i32x8, i8x8};

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

/// SIMD width (lanes) used by the vectorized matmuls.
const LANES: usize = 8;

/// Byte width of an AVX2/AVX-512 256-bit int8 lane — the chunk size of the VNNI kernel.
#[cfg(target_arch = "x86_64")]
const VNNI_LANES: usize = 32;

/// Byte width of one NEON `sdot` int8 chunk — 16 bytes per `int8x16` register.
#[cfg(target_arch = "aarch64")]
const SDOT_LANES: usize = 16;

/// Matrix–vector product `out = W · x` for a row-major weight matrix (scalar).
///
/// `w` is laid out as `[d_out, d_in]` (row `i` is `w[i*d_in .. (i+1)*d_in]`), so
/// `out[i] = Σ_j w[i*d_in + j] * x[j]`. This is the shape every projection weight
/// in the checkpoint uses, so no transpose is ever needed.
///
/// This is the readable reference implementation; [`matmul_simd`] computes the same
/// product with `core::simd` and is selected on the hot path via [`crate::Kernel`].
///
/// # Panics
/// In debug builds, if the slice lengths disagree with `d_in`/`d_out`.
pub fn matmul(out: &mut [f32], x: &[f32], w: &[f32], d_in: usize, d_out: usize) {
    debug_assert_eq!(out.len(), d_out);
    debug_assert_eq!(x.len(), d_in);
    debug_assert_eq!(w.len(), d_in * d_out);

    for (i, o) in out.iter_mut().enumerate() {
        let row = &w[i * d_in..][..d_in];
        let mut sum = 0.0f32;
        for (j, &xj) in x.iter().enumerate() {
            sum += row[j] * xj;
        }
        *o = sum;
    }
}

/// SIMD matrix–vector product `out = W · x` — the vectorized twin of [`matmul`].
///
/// Same row-major `[d_out, d_in]` layout and result. The inner dot product is
/// accumulated 8 `f32` lanes at a time (`f32x8`) and horizontally reduced once per
/// row; the `d_in % 8` tail that does not fill a lane is summed scalar-wise. Because
/// the lane reduction reassociates the additions, the result can differ from
/// [`matmul`] in the last bit — within fp32 rounding, never enough to change a token.
///
/// # Panics
/// In debug builds, if the slice lengths disagree with `d_in`/`d_out`.
pub fn matmul_simd(out: &mut [f32], x: &[f32], w: &[f32], d_in: usize, d_out: usize) {
    debug_assert_eq!(out.len(), d_out);
    debug_assert_eq!(x.len(), d_in);
    debug_assert_eq!(w.len(), d_in * d_out);

    let chunks = d_in / LANES; // whole 8-wide lanes per row
    for (i, o) in out.iter_mut().enumerate() {
        let row = &w[i * d_in..][..d_in];

        // Widened multiply/accumulate over the lane-aligned head of the row.
        let mut acc = f32x8::splat(0.0);
        for j in 0..chunks {
            let vw = f32x8::from_slice(&row[j * LANES..][..LANES]);
            let vx = f32x8::from_slice(&x[j * LANES..][..LANES]);
            acc += vw * vx;
        }
        let mut sum = acc.reduce_sum();

        // Scalar tail for the leftover < LANES columns.
        for j in (chunks * LANES)..d_in {
            sum += row[j] * x[j];
        }
        *o = sum;
    }
}

/// Full int8 (**W8A8**) matrix–vector product `out = W · x` (scalar).
///
/// Both operands are int8: `qw` is the group-wise quantized weight, and the activation
/// is pre-quantized by [`quantize_activation`](crate::quantize::quantize_activation)
/// into `xq` (integer-valued `f32`, range `−127..=127`) with one scale per group in
/// `x_scales`. Within each group the dot product is accumulated in **`i32`** (exact —
/// no float rounding), then the integer sum is scaled by `w_scale · x_scale` and folded
/// into the `f32` output. This is the scheme in llama2.c's `runq.c`: quantizing the
/// activation too is what turns the inner loop into integer arithmetic, which is where
/// int8's *compute* advantage lives (weight-only int8 only saves memory). The matmul
/// output stays `f32` — it feeds rmsnorm/residual/attention, all `f32` — and the next
/// matmul re-quantizes its input.
///
/// `qw` must be a **single** projection's matrix shaped `[d_out, d_in]` — slice a
/// per-layer view with [`QuantizedTensor::layer`] first. The weight and activation
/// share one `group_size`, which must divide `d_in` so a group never straddles a row.
/// [`matmul_q8_simd`] is the vectorized twin.
///
/// # Panics
/// In debug builds, if the lengths disagree or `group_size` does not divide `d_in`.
pub fn matmul_q8(
    out: &mut [f32],
    xq: &[i8],
    x_scales: &[f32],
    qw: &QuantizedTensor,
    d_in: usize,
    d_out: usize,
) {
    debug_assert_eq!(out.len(), d_out);
    debug_assert_eq!(xq.len(), d_in);
    debug_assert_eq!(qw.data.len(), d_in * d_out);
    debug_assert_eq!(d_in % qw.group_size, 0);
    debug_assert_eq!(qw.scales.len(), d_out * (d_in / qw.group_size));
    debug_assert_eq!(x_scales.len(), d_in / qw.group_size);

    let gs = qw.group_size;
    let groups_per_row = d_in / gs;
    for (i, o) in out.iter_mut().enumerate() {
        let row = &qw.data[i * d_in..][..d_in];
        let mut total = 0.0f32;
        for (g, &x_scale) in x_scales.iter().enumerate() {
            let base = g * gs;
            // Exact integer dot product over the group.
            let mut ival: i32 = 0;
            for k in 0..gs {
                ival += row[base + k] as i32 * xq[base + k] as i32;
            }
            let w_scale = qw.scales[i * groups_per_row + g];
            total += ival as f32 * w_scale * x_scale;
        }
        *o = total;
    }
}

/// SIMD W8A8 int8 matmul — the vectorized twin of [`matmul_q8`].
///
/// Same int8×int8 → `i32` per-group dot product, widened to 8 lanes: the weight is
/// loaded as `i8x8` and the (integer-valued) activation as `f32x8`, both `cast` to
/// `i32x8`, multiplied, and accumulated in an `i32x8`; the lanes are reduced once per
/// group and scaled by `w_scale · x_scale`. A scalar tail handles any `group_size % 8`
/// remainder. Integer accumulation is associative, so unlike the fp32 SIMD kernel this
/// matches the scalar [`matmul_q8`] **exactly** (no reassociation error) as long as the
/// `i32` group sum does not overflow — group sizes stay far below that bound.
///
/// # Panics
/// In debug builds, if the lengths disagree or `group_size` does not divide `d_in`.
pub fn matmul_q8_simd(
    out: &mut [f32],
    xq: &[i8],
    x_scales: &[f32],
    qw: &QuantizedTensor,
    d_in: usize,
    d_out: usize,
) {
    debug_assert_eq!(out.len(), d_out);
    debug_assert_eq!(xq.len(), d_in);
    debug_assert_eq!(qw.data.len(), d_in * d_out);
    debug_assert_eq!(d_in % qw.group_size, 0);
    debug_assert_eq!(qw.scales.len(), d_out * (d_in / qw.group_size));
    debug_assert_eq!(x_scales.len(), d_in / qw.group_size);

    let gs = qw.group_size;
    let groups_per_row = d_in / gs;
    let chunks = gs / LANES; // whole 8-wide lanes per group
    for (i, o) in out.iter_mut().enumerate() {
        let row = &qw.data[i * d_in..][..d_in];
        let mut total = 0.0f32;
        for (g, &x_scale) in x_scales.iter().enumerate() {
            let base = g * gs;
            let wg = &row[base..][..gs];
            let xg = &xq[base..][..gs];

            // Per-group integer dot product: widen both i8 sides to i32, multiply, accumulate.
            let mut acc = i32x8::splat(0);
            for c in 0..chunks {
                let vw: i32x8 = i8x8::from_slice(&wg[c * LANES..][..LANES]).cast();
                let vx: i32x8 = i8x8::from_slice(&xg[c * LANES..][..LANES]).cast();
                acc += vw * vx;
            }
            let mut ival = acc.reduce_sum();
            for k in (chunks * LANES)..gs {
                ival += wg[k] as i32 * xg[k] as i32;
            }

            total += ival as f32 * qw.scales[i * groups_per_row + g] * x_scale;
        }
        *o = total;
    }
}

/// Full int8 (**W8A8**) matmul using x86 AVX-512 **VNNI** (`vpdpbusd`).
///
/// The hardware-accelerated twin of [`matmul_q8`]. `_mm256_dpbusd_epi32` does 32 int8
/// multiply-accumulates into 8 `i32` lanes per instruction (accumulating directly in
/// `i32`, so unlike the older `pmaddubsw` path it never saturates). VNNI multiplies
/// *unsigned* × *signed* bytes, so each weight is offset by `+128` in-register (mapping
/// the stored `i8` to the `u8` the instruction wants) and the per-group correction
/// `128 · Σ(activations)` — precomputed in `x_gsums` by
/// [`quantize_activation`](crate::quantize::quantize_activation) — is subtracted back
/// out: `Σ wₖaₖ = Σ(wₖ+128)aₖ − 128·Σaₖ`. The integer dot product is exact, so this is
/// **bit-identical** to the scalar [`matmul_q8`].
///
/// Requires `group_size` to be a multiple of 32 (the dispatch falls back to the portable
/// kernel otherwise). Each group is processed in 32-byte chunks accumulated in one `i32`
/// vector, reduced once per group, corrected, scaled, and folded into the `f32` output.
///
/// # Safety
/// The CPU must support `avx2`, `avx512vl`, and `avx512vnni`. The std host verifies this
/// with `is_x86_feature_detected!` before selecting [`Kernel::Dotprod`](crate::Kernel);
/// any other caller must guarantee it.
///
/// # Panics
/// In debug builds, if the lengths disagree or `group_size` is not a multiple of 32.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avx512vl,avx512vnni")]
pub unsafe fn matmul_q8_dotprod(
    out: &mut [f32],
    xq: &[i8],
    x_scales: &[f32],
    x_gsums: &[f32],
    qw: &QuantizedTensor,
    d_in: usize,
    d_out: usize,
) {
    debug_assert_eq!(out.len(), d_out);
    debug_assert_eq!(xq.len(), d_in);
    debug_assert_eq!(qw.data.len(), d_in * d_out);
    debug_assert_eq!(qw.group_size % VNNI_LANES, 0);
    debug_assert_eq!(d_in % qw.group_size, 0);
    debug_assert_eq!(qw.scales.len(), d_out * (d_in / qw.group_size));
    debug_assert_eq!(x_scales.len(), d_in / qw.group_size);
    debug_assert_eq!(x_gsums.len(), d_in / qw.group_size);

    let gs = qw.group_size;
    let groups_per_row = d_in / gs;
    let chunks = gs / VNNI_LANES;
    // SAFETY: every 32-byte load lands inside its row (group `base` + chunk offset stays
    // below `d_in`, and `qw.data` is `d_in * d_out` long); the `target_feature` contract
    // guarantees the AVX2/AVX-512 intrinsics are available on this CPU.
    unsafe {
        let bias = _mm256_set1_epi8(-128i8); // +128 mod 256: maps i8 weight to u8
        for (i, o) in out.iter_mut().enumerate() {
            let row = qw.data.as_ptr().add(i * d_in);
            let mut total = 0.0f32;
            for (g, &x_scale) in x_scales.iter().enumerate() {
                let base = g * gs;
                let mut acc = _mm256_setzero_si256();
                for c in 0..chunks {
                    let off = base + c * VNNI_LANES;
                    let w = _mm256_loadu_si256(row.add(off) as *const __m256i);
                    let wu = _mm256_add_epi8(w, bias); // unsigned-offset weight
                    let a = _mm256_loadu_si256(xq.as_ptr().add(off) as *const __m256i);
                    acc = _mm256_dpbusd_epi32(acc, wu, a);
                }
                // Horizontal-sum the 8 i32 lanes to one scalar, then undo the +128 offset.
                let lo = _mm256_castsi256_si128(acc);
                let hi = _mm256_extracti128_si256::<1>(acc);
                let mut s = _mm_add_epi32(lo, hi);
                s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b01_00_11_10>(s));
                s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b00_00_00_01>(s));
                let p = _mm_cvtsi128_si32(s);
                let dot = p - 128 * (x_gsums[g] as i32);
                total += dot as f32 * qw.scales[i * groups_per_row + g] * x_scale;
            }
            *o = total;
        }
    }
}

/// Full int8 (**W8A8**) matmul using the ARM NEON **dot-product** extension (`sdot`).
///
/// The aarch64 counterpart of the x86 [`matmul_q8_dotprod`] above. `vdotq_s32`
/// (`sdot`) does 16 int8 multiply-accumulates into four `i32` lanes per instruction.
/// Crucially, unlike x86 VNNI's `vpdpbusd`, `sdot` multiplies **signed × signed**
/// bytes directly, so the int8 weights and activations are consumed exactly as stored:
/// there is no `+128`/`−128` unsigned-offset trick and this kernel needs no per-group
/// activation sums (`x_gsums`) at all. The integer accumulation is exact, so the result
/// is **bit-identical** to the scalar [`matmul_q8`].
///
/// Each group is processed in 16-byte chunks accumulated in one `int32x4` vector and
/// reduced once per group; a scalar tail handles any `group_size % 16` remainder, so —
/// unlike the x86 path, which requires a multiple of 32 — every group size is supported.
///
/// `qw` must be a **single** projection's matrix shaped `[d_out, d_in]` (slice it with
/// [`QuantizedTensor::layer`] first), and `group_size` must divide `d_in`.
///
/// # Safety
/// The CPU must support the NEON `dotprod` feature. The std host verifies this with
/// `is_aarch64_feature_detected!("dotprod")` before selecting
/// [`Kernel::Dotprod`](crate::Kernel); any other caller must guarantee it.
///
/// # Panics
/// In debug builds, if the slice lengths disagree with `d_in`/`d_out`/`group_size`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
pub unsafe fn matmul_q8_dotprod(
    out: &mut [f32],
    xq: &[i8],
    x_scales: &[f32],
    qw: &QuantizedTensor,
    d_in: usize,
    d_out: usize,
) {
    debug_assert_eq!(out.len(), d_out);
    debug_assert_eq!(xq.len(), d_in);
    debug_assert_eq!(qw.data.len(), d_in * d_out);
    debug_assert_eq!(d_in % qw.group_size, 0);
    debug_assert_eq!(qw.scales.len(), d_out * (d_in / qw.group_size));
    debug_assert_eq!(x_scales.len(), d_in / qw.group_size);

    let gs = qw.group_size;
    let groups_per_row = d_in / gs;
    let chunks = gs / SDOT_LANES; // whole 16-wide lanes per group
    for (i, o) in out.iter_mut().enumerate() {
        let row = &qw.data[i * d_in..][..d_in];
        let mut total = 0.0f32;
        for (g, &x_scale) in x_scales.iter().enumerate() {
            let base = g * gs;
            let wg = &row[base..][..gs];
            let xg = &xq[base..][..gs];

            // Per-group signed int8 dot product: 16 lanes/instruction via `sdot`.
            let mut acc = vdupq_n_s32(0);
            for c in 0..chunks {
                let wc = &wg[c * SDOT_LANES..][..SDOT_LANES];
                let xc = &xg[c * SDOT_LANES..][..SDOT_LANES];
                // SAFETY: each load reads exactly the 16 in-bounds bytes of `wc`/`xc`;
                // the `target_feature` contract guarantees NEON `dotprod` on this CPU.
                let vw = unsafe { vld1q_s8(wc.as_ptr()) };
                let vx = unsafe { vld1q_s8(xc.as_ptr()) };
                acc = vdotq_s32(acc, vw, vx);
            }
            let mut ival = vaddvq_s32(acc);

            // Scalar tail for the leftover < 16 columns (when 16 ∤ group_size).
            for k in (chunks * SDOT_LANES)..gs {
                ival += wg[k] as i32 * xg[k] as i32;
            }

            total += ival as f32 * qw.scales[i * groups_per_row + g] * x_scale;
        }
        *o = total;
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

    extern crate std;
    use std::vec;
    use std::vec::Vec;

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
    fn matmul_simd_matches_scalar_including_the_tail() {
        // d_in = 11 is not a multiple of LANES (8), so this exercises the scalar
        // tail of the SIMD kernel alongside one full 8-wide lane.
        let (d_in, d_out) = (11usize, 3usize);
        let w: Vec<f32> = (0..d_in * d_out).map(|i| (i as f32 * 0.13).sin()).collect();
        let x: Vec<f32> = (0..d_in).map(|i| (i as f32 * 0.7).cos()).collect();

        let mut a = vec![0.0f32; d_out];
        let mut b = vec![0.0f32; d_out];
        matmul(&mut a, &x, &w, d_in, d_out);
        matmul_simd(&mut b, &x, &w, d_in, d_out);
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-5, "{x} vs {y}");
        }
    }

    // Quantize an activation the way the engine does before a W8A8 matmul:
    // int8 values, per-group scales, and per-group integer sums.
    fn quant_act(x: &[f32], gs: usize) -> (Vec<i8>, Vec<f32>, Vec<f32>) {
        let mut xq = vec![0i8; x.len()];
        let mut xs = vec![0.0f32; x.len() / gs];
        let mut xg = vec![0.0f32; x.len() / gs];
        crate::quantize::quantize_activation(&mut xq, &mut xs, &mut xg, x, gs);
        (xq, xs, xg)
    }

    #[test]
    fn matmul_q8_simd_matches_scalar_exactly() {
        // Two rows of 16 int8 weights, group_size 8 → one full SIMD lane per group.
        // Integer accumulation is associative, so scalar and SIMD must agree *exactly*.
        let (d_in, d_out, gs) = (16usize, 2usize, 8usize);
        let data: Vec<i8> = (0..d_in * d_out).map(|i| (i as i8) - 16).collect();
        let scales: Vec<f32> = vec![0.5, 0.25, 0.125, 0.0625]; // d_out * (d_in/gs) = 4
        let qw = QuantizedTensor {
            data: &data,
            scales: &scales,
            group_size: gs,
        };
        let x: Vec<f32> = (0..d_in).map(|i| (i as f32 * 0.3).sin()).collect();
        let (xq, xs, _xg) = quant_act(&x, gs);

        let mut a = vec![0.0f32; d_out];
        let mut b = vec![0.0f32; d_out];
        matmul_q8(&mut a, &xq, &xs, &qw, d_in, d_out);
        matmul_q8_simd(&mut b, &xq, &xs, &qw, d_in, d_out);
        assert_eq!(a, b);
    }

    #[test]
    fn matmul_q8_computes_the_scaled_integer_dot_product() {
        // group_size 4 < LANES: the whole group runs through the scalar tail path.
        // Weights are ±1-scaled; the activation is all-ones (which quantizes to a
        // single group scale of 1/127 with every value at the +127 rail).
        let (d_in, d_out, gs) = (8usize, 1usize, 4usize);
        let data: Vec<i8> = vec![1, -2, 3, -4, 5, -6, 7, -8];
        let scales: Vec<f32> = vec![1.0, 2.0]; // two groups of 4
        let qw = QuantizedTensor {
            data: &data,
            scales: &scales,
            group_size: gs,
        };
        let x = vec![1.0f32; d_in];
        let (xq, xs, _xg) = quant_act(&x, gs);
        // Each group is all-ones → scale 1/127, every quantized activation = 127.
        assert!(xq.iter().all(|&v| v == 127));

        let mut scalar = vec![0.0f32; d_out];
        let mut simd = vec![0.0f32; d_out];
        matmul_q8(&mut scalar, &xq, &xs, &qw, d_in, d_out);
        matmul_q8_simd(&mut simd, &xq, &xs, &qw, d_in, d_out);
        // group0: (1-2+3-4)*127*(1)*(1/127) = -2 ; group1: (5-6+7-8)*127*(2)*(1/127) = -4.
        assert!(close(scalar[0], -6.0));
        assert_eq!(scalar, simd);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn matmul_q8_dotprod_matches_scalar_exactly() {
        // VNNI computes the same exact integer dot product, so it must agree with the
        // scalar kernel to the bit — on a machine that actually has the instructions.
        if !std::is_x86_feature_detected!("avx512vnni")
            || !std::is_x86_feature_detected!("avx512vl")
        {
            std::eprintln!("skipping: CPU lacks AVX-512 VNNI");
            return;
        }
        // group_size 32 (one VNNI chunk per group), 96 columns → 3 groups, 5 rows.
        let (d_in, d_out, gs) = (96usize, 5usize, 32usize);
        let data: Vec<i8> = (0..d_in * d_out)
            .map(|i| (((i * 37) % 255) as i32 - 127) as i8)
            .collect();
        let scales: Vec<f32> = (0..d_out * (d_in / gs))
            .map(|i| 0.01 + i as f32 * 0.003)
            .collect();
        let qw = QuantizedTensor {
            data: &data,
            scales: &scales,
            group_size: gs,
        };
        let x: Vec<f32> = (0..d_in).map(|i| (i as f32 * 0.21).sin() * 1.7).collect();
        let (xq, xs, xg) = quant_act(&x, gs);

        let mut scalar = vec![0.0f32; d_out];
        let mut vnni = vec![0.0f32; d_out];
        matmul_q8(&mut scalar, &xq, &xs, &qw, d_in, d_out);
        // SAFETY: guarded by the feature check above.
        unsafe { matmul_q8_dotprod(&mut vnni, &xq, &xs, &xg, &qw, d_in, d_out) };
        assert_eq!(scalar, vnni);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn matmul_q8_dotprod_matches_scalar_exactly() {
        // ARM `sdot` computes the same exact signed integer dot product, so it must
        // agree with the scalar kernel to the bit — on a machine that has the extension.
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            std::eprintln!("skipping: CPU lacks the NEON dotprod extension");
            return;
        }
        // group_size 40 = 16 + 16 + 8, so every group runs two full `sdot` lanes plus
        // an 8-wide scalar tail — exercising both code paths. 80 columns → 2 groups,
        // 5 rows.
        let (d_in, d_out, gs) = (80usize, 5usize, 40usize);
        let data: Vec<i8> = (0..d_in * d_out)
            .map(|i| (((i * 37) % 255) as i32 - 127) as i8)
            .collect();
        let scales: Vec<f32> = (0..d_out * (d_in / gs))
            .map(|i| 0.01 + i as f32 * 0.003)
            .collect();
        let qw = QuantizedTensor {
            data: &data,
            scales: &scales,
            group_size: gs,
        };
        let x: Vec<f32> = (0..d_in).map(|i| (i as f32 * 0.21).sin() * 1.7).collect();
        let (xq, xs, _xg) = quant_act(&x, gs);

        let mut scalar = vec![0.0f32; d_out];
        let mut sdot = vec![0.0f32; d_out];
        matmul_q8(&mut scalar, &xq, &xs, &qw, d_in, d_out);
        // SAFETY: guarded by the feature check above.
        unsafe { matmul_q8_dotprod(&mut sdot, &xq, &xs, &qw, d_in, d_out) };
        assert_eq!(scalar, sdot);
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
