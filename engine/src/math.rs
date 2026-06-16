//! Math kernels for the forward pass.
//!
//! Every kernel writes into caller-provided slices and allocates nothing, so they
//! are safe to call from the hot path over buffers carved from the [`Arena`]. They
//! depend on [`libm`] for the transcendental functions that `core` lacks
//! (`expf`, `powf`, `sinf`, `cosf`, `sqrtf`).
//!
//! The matrix–vector products — the bulk of the compute — come in two flavours: a
//! straightforward **scalar** version and a **SIMD** version built on the [`wide`] crate
//! (portable SIMD vectors on stable Rust — no nightly `core::simd`). Both fp32
//! ([`matmul`] / [`matmul_simd`]) and int8-quantized ([`matmul_q8`] / [`matmul_q8_simd`])
//! matmuls have both. The scalar
//! kernels are the readable reference; the SIMD kernels widen the inner dot product
//! to 8 lanes of `f32` and fall back to scalar for any tail that does not fill a lane.
//! The fp32 pair agree up to floating-point reassociation; the int8 pair agree
//! **exactly** (integer accumulation is associative). The caller picks which via
//! [`Kernel`], the architecture-agnostic kernel selector both forward passes thread
//! through.
//!
//! The int8 matmul additionally has a hardware **dot-product** kernel, selected by
//! [`Kernel::Dotprod`]: x86 AVX-512 VNNI (`vpdpbusd`) and ARM NEON
//! `sdot`, both [`matmul_q8_dotprod`] (one per `cfg(target_arch)`). They run the same
//! exact integer dot product as the scalar kernel, just far faster, and the std host
//! only selects them after detecting the CPU feature at runtime.
//!
//! The remaining kernels follow Karpathy's llama2.c `run.c` operation-for-operation;
//! the details that matter for token-for-token parity at temperature 0 are called
//! out per function below.
//!
//! [`Arena`]: crate::Arena

use crate::quant::{quantize_activation, QuantScratch, QuantizedTensor};

/// Which matmul implementation a forward pass should use.
///
/// Architecture-agnostic and orthogonal to the weight representation: every matmul
/// has a scalar reference kernel and a [`wide`] SIMD kernel ([`matmul`]/[`matmul_simd`]
/// for fp32, [`matmul_q8`]/[`matmul_q8_simd`] for int8), and the int8 path also has a
/// hardware dot-product kernel. Both the Llama and seq2seq forward passes thread a
/// `Kernel` through and dispatch their matmuls on it. The choice does not change token
/// output beyond fp32 rounding noise; it is a speed/reference knob.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    /// Straightforward scalar kernels — the readable reference path.
    Scalar,
    /// Vectorized [`wide`] kernels (8-wide lanes).
    Simd,
    /// Hardware int8 **dot-product** kernel ([`matmul_q8_dotprod`]) for the int8 path —
    /// x86 AVX-512 **VNNI** or ARM NEON **`sdot`**. The fp32 path always falls back to
    /// [`Kernel::Simd`], as does the x86 kernel for any group size not a multiple of 32;
    /// the ARM kernel handles any group size. Targets with neither instruction fall back
    /// to SIMD entirely. The std host only selects this after detecting the CPU feature
    /// at runtime.
    Dotprod,
}
use wide::{f32x8, i32x8};

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

/// SIMD width (lanes) used by the vectorized matmuls.
const LANES: usize = 8;

/// Widen one 8-byte int8 chunk to `[i32; 8]` so it can fill an [`i32x8`] (`wide` has no
/// i8 vector type, so the int8 matmul widens each chunk before the vectorized multiply).
#[inline]
fn widen_i8(chunk: &[i8]) -> [i32; LANES] {
    let mut out = [0i32; LANES];
    for (o, &v) in out.iter_mut().zip(chunk) {
        *o = v as i32;
    }
    out
}

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
/// product with [`wide`] SIMD and is selected on the hot path via [`crate::llama::Kernel`].
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

    for (i, o) in out.iter_mut().enumerate() {
        let row = &w[i * d_in..][..d_in];

        // Widened multiply/accumulate over the lane-aligned head of the row.
        let mut wj = row.chunks_exact(LANES);
        let mut xj = x.chunks_exact(LANES);
        let mut acc = f32x8::splat(0.0);
        for (wc, xc) in wj.by_ref().zip(xj.by_ref()) {
            let vw = f32x8::new(<[f32; LANES]>::try_from(wc).unwrap());
            let vx = f32x8::new(<[f32; LANES]>::try_from(xc).unwrap());
            acc += vw * vx;
        }
        let mut sum = acc.reduce_add();

        // Scalar tail for the leftover < LANES columns.
        for (&wt, &xt) in wj.remainder().iter().zip(xj.remainder()) {
            sum += wt * xt;
        }
        *o = sum;
    }
}

/// Full int8 (**W8A8**) matrix–vector product `out = W · x` (scalar).
///
/// Both operands are int8: `qw` is the group-wise quantized weight, and the activation
/// is pre-quantized by [`quantize_activation`]
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
/// Same int8×int8 → `i32` per-group dot product, widened to 8 lanes: each 8-byte chunk
/// of the weight and the activation is widened from `i8` to `[i32; 8]` (`wide` has no i8
/// vector), loaded into an `i32x8`, multiplied, and accumulated in an `i32x8`; the lanes
/// are reduced once per group and scaled by `w_scale · x_scale`. A scalar tail handles any
/// `group_size % 8` remainder. Integer accumulation is associative, so unlike the fp32 SIMD kernel this
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
    for (i, o) in out.iter_mut().enumerate() {
        let row = &qw.data[i * d_in..][..d_in];
        let mut total = 0.0f32;
        for (g, &x_scale) in x_scales.iter().enumerate() {
            let base = g * gs;
            let wg = &row[base..][..gs];
            let xg = &xq[base..][..gs];

            // Per-group integer dot product: widen both i8 sides to i32, multiply, accumulate.
            // `wide` has no i8 vector, so each 8-byte chunk is widened to an `i32x8`.
            let mut wc = wg.chunks_exact(LANES);
            let mut xc = xg.chunks_exact(LANES);
            let mut acc = i32x8::splat(0);
            for (w8, x8) in wc.by_ref().zip(xc.by_ref()) {
                acc += i32x8::new(widen_i8(w8)) * i32x8::new(widen_i8(x8));
            }
            let mut ival = acc.reduce_add();
            for (&wk, &xk) in wc.remainder().iter().zip(xc.remainder()) {
                ival += wk as i32 * xk as i32;
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
/// [`quantize_activation`] — is subtracted back
/// out: `Σ wₖaₖ = Σ(wₖ+128)aₖ − 128·Σaₖ`. The integer dot product is exact, so this is
/// **bit-identical** to the scalar [`matmul_q8`].
///
/// Requires `group_size` to be a multiple of 32 (the dispatch falls back to the portable
/// kernel otherwise). Each group is processed in 32-byte chunks accumulated in one `i32`
/// vector, reduced once per group, corrected, scaled, and folded into the `f32` output.
///
/// # Safety
/// The CPU must support `avx2`, `avx512vl`, and `avx512vnni`. The std host verifies this
/// with `is_x86_feature_detected!` before selecting [`Kernel::Dotprod`](crate::llama::Kernel);
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
/// [`Kernel::Dotprod`](crate::llama::Kernel); any other caller must guarantee it.
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

/// One full int8 (**W8A8**) projection `out = W · x` over an fp32 activation `x`.
///
/// The single entry point both architectures use for a quantized matmul: it quantizes the
/// fp32 activation `x` into the caller-owned [`QuantScratch`] (int8 values + per-group
/// scales + the per-group sums the VNNI correction needs) with
/// [`quantize_activation`], then runs the integer dot product against the int8 weights
/// `wl` (a single projection's `[d_out, d_in]` matrix — slice a layer with
/// [`QuantizedTensor::layer`] first) on the kernel `kernel` selects:
///
/// * [`Kernel::Scalar`] → [`matmul_q8`] (readable reference);
/// * [`Kernel::Simd`] → [`matmul_q8_simd`] (portable [`wide`] SIMD);
/// * [`Kernel::Dotprod`] → the hardware dot-product kernel (x86 VNNI / ARM `sdot`), which
///   itself falls back to SIMD on group sizes / targets it cannot serve.
///
/// All three produce the same answer (the int8 kernels agree exactly; only the fp32 output
/// fold rounds), so `kernel` is a pure speed/reference knob. The `[..d_in]` / `[..groups]`
/// prefixes of the scratch are filled and read, so one scratch sized for the largest
/// projection serves every matmul.
///
/// # Panics
/// In debug builds, via the kernels' length assertions, or if the scratch buffers are
/// shorter than `d_in` / `d_in / group_size`.
pub fn matmul_w8a8(
    kernel: Kernel,
    out: &mut [f32],
    x: &[f32],
    wl: &QuantizedTensor,
    d_in: usize,
    d_out: usize,
    qs: &mut QuantScratch,
) {
    let gs = wl.group_size;
    let groups = d_in / gs;
    let xqd = &mut qs.xq[..d_in];
    let xqs = &mut qs.scales[..groups];
    let xqg = &mut qs.gsums[..groups];
    quantize_activation(xqd, xqs, xqg, x, gs);
    match kernel {
        Kernel::Scalar => matmul_q8(out, xqd, xqs, wl, d_in, d_out),
        Kernel::Simd => matmul_q8_simd(out, xqd, xqs, wl, d_in, d_out),
        Kernel::Dotprod => dispatch_q8_dotprod(out, xqd, xqs, xqg, wl, d_in, d_out),
    }
}

/// Dispatch the int8 matmul to a hardware dot-product kernel when one applies, else the
/// portable SIMD kernel. The two hardware kernels differ in shape, so this is the single
/// place that reconciles them:
///
/// * **x86 AVX-512 VNNI** (`vpdpbusd`): unsigned×signed, so it consumes the per-group
///   `xq_gsums` correction and needs a `group_size` that is a multiple of 32 — other
///   sizes fall through to SIMD.
/// * **ARM NEON `sdot`**: signed×signed, so it ignores `xq_gsums` entirely and its scalar
///   tail covers any `group_size`.
/// * **Everything else** (and the `thumbv7em` embedded build): portable SIMD.
fn dispatch_q8_dotprod(
    out: &mut [f32],
    xq: &[i8],
    xq_scales: &[f32],
    xq_gsums: &[f32],
    wl: &QuantizedTensor,
    d_in: usize,
    d_out: usize,
) {
    #[cfg(target_arch = "x86_64")]
    if wl.group_size.is_multiple_of(32) {
        // SAFETY: the std host only selects `Kernel::Dotprod` after detecting avx2 +
        // avx512vl + avx512vnni at runtime; the group-size precondition is checked here.
        unsafe { matmul_q8_dotprod(out, xq, xq_scales, xq_gsums, wl, d_in, d_out) };
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        let _ = xq_gsums; // sdot is signed×signed — no +128 correction needed
                          // SAFETY: the std host only selects `Kernel::Dotprod` after detecting the NEON
                          // `dotprod` feature at runtime.
        unsafe { matmul_q8_dotprod(out, xq, xq_scales, wl, d_in, d_out) };
    }

    // SIMD fallback for non-aarch64 targets (and x86 group sizes VNNI can't take above).
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = xq_gsums; // unused by the portable fallback
        matmul_q8_simd(out, xq, xq_scales, wl, d_in, d_out);
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

/// Layer normalization with weight **and** bias: `out = (x − μ)/√(σ² + ε) ⊙ w + b`.
///
/// `μ` and `σ²` are the mean and the **biased** variance (divided by `N`, not `N−1`)
/// over the full length of `x`, and `ε = 1e-5` — matching PyTorch's `nn.LayerNorm`
/// defaults, which is what Marian / OPUS-MT uses. This is the encoder-decoder
/// counterpart of [`rmsnorm`]: RMSNorm drops the mean-centering and the bias, so the
/// Llama path uses that while the Marian path uses this. Matching `ε` and the
/// biased-variance convention exactly is what keeps the greedy MT token stream aligned
/// with the reference.
///
/// # Panics
/// In debug builds, if `out`, `x`, `w`, and `b` differ in length.
pub fn layernorm(out: &mut [f32], x: &[f32], w: &[f32], b: &[f32]) {
    const EPS: f32 = 1e-5;

    debug_assert_eq!(out.len(), x.len());
    debug_assert_eq!(x.len(), w.len());
    debug_assert_eq!(w.len(), b.len());

    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv_std = 1.0f32 / libm::sqrtf(var + EPS);

    for (((o, &xi), &wi), &bi) in out.iter_mut().zip(x).zip(w).zip(b) {
        *o = (xi - mean) * inv_std * wi + bi;
    }
}

/// In-place numerically-stable softmax over `x`.
///
/// Subtracts the max before exponentiating (so the largest term is `e^0 = 1` and
/// nothing overflows), then normalizes to sum to 1. Used for the attention
/// weights. (The host's temperature sampler applies the same max-shift trick over
/// the logits, but inline, since it only borrows them immutably.)
pub fn softmax(x: &mut [f32]) {
    let max_val = x.iter().copied().max_by(|a, b| a.total_cmp(b)).unwrap();

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

/// One attention head's key and value vectors as they sit in a cache region.
///
/// `keys` and `values` are parallel flat slices holding one or more positions laid end
/// to end; position `t`'s head vector is `[t*stride + head_off ..][..head_size]`. This
/// is exactly the addressing the engine's KV caches use, so a head view can be built
/// over any of them:
///
/// * decoder self-attention — `stride = kv_dim`, `head_off = (h / kv_mul) * head_size`
///   (grouped-query: several query heads share one KV head);
/// * encoder self-attention / cross-attention — `stride = d_model`, `head_off =
///   h * head_size` (every head has its own K/V).
#[derive(Clone, Copy)]
pub struct KvHead<'a> {
    /// Flat key cache; position `t`'s head vector starts at `t*stride + head_off`.
    pub keys: &'a [f32],
    /// Flat value cache, addressed identically to [`keys`](Self::keys).
    pub values: &'a [f32],
    /// Per-position width of the cache (e.g. `kv_dim`, or `d_model` for the seq2seq path).
    pub stride: usize,
    /// Offset of this head within one position's row.
    pub head_off: usize,
}

/// Scaled dot-product attention for a single query head.
///
/// Scores `q` against the first `n_valid` key positions of `kv` (each score scaled by
/// `scale`), softmaxes them (max-shifted) into `scores`, and writes the value-weighted
/// sum into `out`. This is the one code path behind all three attention flavors — the
/// only thing that varies is the valid window, expressed by `n_valid`:
///
/// * **causal** (decoder self-attention): `n_valid = pos + 1`, i.e. keys `0..=pos`;
/// * **bidirectional** (encoder self-attention) and **cross-attention**: `n_valid =
///   src_len`, i.e. every source key.
///
/// `out` and `q` are one head each (`head_size = out.len()`), and `scores` must hold at
/// least `n_valid` elements (only `0..n_valid` is read or written). Allocates nothing.
/// The score accumulation, softmax, and value sum run in the same order as the original
/// hand-written decoder loop, so substituting this helper there is bit-for-bit neutral.
///
/// # Panics
/// In debug builds, if `q` is not `head_size` long or `scores` is shorter than `n_valid`.
pub fn attention_head(
    out: &mut [f32],
    q: &[f32],
    kv: &KvHead,
    n_valid: usize,
    scale: f32,
    scores: &mut [f32],
) {
    let head_size = out.len();
    debug_assert_eq!(q.len(), head_size);
    debug_assert!(scores.len() >= n_valid);

    // Scaled dot-product score against every valid key.
    for (t, score) in scores[..n_valid].iter_mut().enumerate() {
        let off = t * kv.stride + kv.head_off;
        let k_t = &kv.keys[off..off + head_size];
        let mut dot = 0.0f32;
        for i in 0..head_size {
            dot += q[i] * k_t[i];
        }
        *score = dot * scale;
    }

    // Softmax over exactly the valid window.
    softmax(&mut scores[..n_valid]);

    // Weighted sum of the values back into this head's output.
    out.fill(0.0);
    for (t, &a) in scores[..n_valid].iter().enumerate() {
        let off = t * kv.stride + kv.head_off;
        let v_t = &kv.values[off..off + head_size];
        for i in 0..head_size {
            out[i] += a * v_t[i];
        }
    }
}

/// SiLU / swish activation: `z * σ(z) = z / (1 + e^-z)`.
///
/// This is the gate non-linearity of the SwiGLU feed-forward block.
pub fn silu(z: f32) -> f32 {
    z / (1.0 + libm::expf(-z))
}

/// Exact-erf GELU activation: `z · Φ(z) = 0.5·z·(1 + erf(z/√2))`.
///
/// `Φ` is the standard-normal CDF, computed exactly through `libm::erff` rather than the
/// `tanh` approximation. This is Hugging Face's default `"gelu"` activation; the Marian
/// feed-forward uses it when the model's `activation_function` is `gelu`, while [`silu`]
/// covers the `swish` case (the two activations Marian / OPUS-MT models use). The engine
/// never hardcodes one — the checkpoint's activation flag selects between them.
pub fn gelu(z: f32) -> f32 {
    use core::f32::consts::FRAC_1_SQRT_2; // 1/√2
    0.5 * z * (1.0 + libm::erff(z * FRAC_1_SQRT_2))
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

/// Write Marian's sinusoidal absolute position encoding for `pos` into `buf` (`dim`).
///
/// This is the OPUS-MT / Marian layout, which — unlike the classic *interleaved*
/// "Attention Is All You Need" encoding — **concatenates** the two components: the first
/// `⌈dim/2⌉` entries are sines and the rest are cosines (HF
/// `MarianSinusoidalPositionalEmbedding`). Sine entry `s` and cosine entry `c` each use
/// the frequency `10000^(−2k/dim)` of their own index `k`:
///
/// ```text
/// buf[s]            = sin(pos · 10000^(−2s/dim))    for s in 0..⌈dim/2⌉
/// buf[⌈dim/2⌉ + c]  = cos(pos · 10000^(−2c/dim))    for c in 0..⌊dim/2⌋
/// ```
///
/// Positions start at index 0 (Marian has no BART `+2` offset). The result is *added* to
/// the scaled token embedding (see [`embed_scale`]); it is recomputed per position rather
/// than stored, so the checkpoint carries no position table. This is the Marian path's
/// replacement for the Llama path's rotary [`rope`].
///
/// # Panics
/// In debug builds, if `buf.len() != dim`.
pub fn sinusoidal_into(buf: &mut [f32], pos: usize, dim: usize) {
    debug_assert_eq!(buf.len(), dim);

    // ⌈dim/2⌉: sines fill [0, sentinel), cosines fill [sentinel, dim).
    let sentinel = dim.div_ceil(2);
    let p = pos as f32;
    let (sines, cosines) = buf.split_at_mut(sentinel);
    for (s, slot) in sines.iter_mut().enumerate() {
        let freq = 1.0 / libm::powf(10000.0, 2.0 * s as f32 / dim as f32);
        *slot = libm::sinf(p * freq);
    }
    for (c, slot) in cosines.iter_mut().enumerate() {
        let freq = 1.0 / libm::powf(10000.0, 2.0 * c as f32 / dim as f32);
        *slot = libm::cosf(p * freq);
    }
}

/// Embedding scale factor `√dim` for Marian's `scale_embedding`.
///
/// When a Marian model sets `scale_embedding = true`, every token embedding is multiplied
/// by `√d_model` before the sinusoidal positions ([`sinusoidal_into`]) are added. Keeping
/// it as a named helper puts the constant — and this parity note — in one place; models
/// with `scale_embedding = false` simply skip the multiply.
pub fn embed_scale(dim: usize) -> f32 {
    libm::sqrtf(dim as f32)
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

/// Add a bias vector into a linear layer's output in place: `out[i] += bias[i]`.
///
/// Marian's projections (q/k/v/o, fc1/fc2) and LayerNorms are all biased, unlike the
/// Llama path's bias-free linears — a [`matmul`] followed by `add_bias` is one biased
/// linear layer. Mechanically identical to [`accumulate`]; the separate name marks the
/// separate intent (adding a learned bias, not folding in a residual).
pub fn add_bias(out: &mut [f32], bias: &[f32]) {
    accumulate(out, bias);
}

/// Index of the maximum element, taking the **first** on ties.
///
/// This is greedy (temperature-0) decoding: pick the most probable next token.
/// The first-on-ties rule (strict `>`) matches llama2.c's `argmax`, which keeps
/// the earliest index — important for token-for-token parity.
///
/// # Panics
/// In debug builds, if `buf` is empty.
pub fn argmax(buf: &[f32]) -> (usize, f32) {
    debug_assert!(!buf.is_empty());

    let mut best = 0;
    let mut best_val = buf[0];
    for (i, &val) in buf.iter().enumerate() {
        if val > best_val {
            best_val = val;
            best = i;
        }
    }
    (best, best_val)
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
        crate::quant::quantize_activation(&mut xq, &mut xs, &mut xg, x, gs);
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
    fn accumulate_adds_in_place() {
        let mut dst = [1.0, 2.0, 3.0];
        accumulate(&mut dst, &[10.0, 20.0, 30.0]);
        assert_eq!(dst, [11.0, 22.0, 33.0]);
    }

    #[test]
    fn argmax_picks_max_and_first_on_ties() {
        assert_eq!(argmax(&[0.1, 0.7, 0.2]).0, 1);
        assert_eq!(argmax(&[5.0]).0, 0);
        // Ties resolve to the earliest index (strict `>`).
        assert_eq!(argmax(&[2.0, 2.0, 1.0, 2.0]).0, 0);
        assert_eq!(argmax(&[-3.0, -1.0, -1.0]).0, 1);
    }

    #[test]
    fn layernorm_standardizes_then_applies_affine() {
        // x=[1,2,3,4]: mean=2.5, biased var = mean((x-2.5)²) = 1.25.
        let x = [1.0, 2.0, 3.0, 4.0];
        let (mean, var) = (2.5f32, 1.25f32);
        let inv = 1.0 / (var + 1e-5).sqrt();

        // Unit gain, zero bias → pure standardization (≈ zero mean, unit variance).
        let mut out = [0.0; 4];
        layernorm(&mut out, &x, &[1.0; 4], &[0.0; 4]);
        for (o, xi) in out.iter().zip(x.iter()) {
            assert!(close(*o, (xi - mean) * inv));
        }
        let out_mean: f32 = out.iter().sum::<f32>() / 4.0;
        assert!(out_mean.abs() < 1e-5);

        // Affine weight + bias applied element-wise.
        let w = [2.0, 0.5, 1.0, -1.0];
        let b = [0.1, 0.2, 0.3, 0.4];
        let mut out2 = [0.0; 4];
        layernorm(&mut out2, &x, &w, &b);
        for i in 0..4 {
            assert!(close(out2[i], (x[i] - mean) * inv * w[i] + b[i]));
        }
    }

    #[test]
    fn gelu_known_values() {
        assert!(close(gelu(0.0), 0.0));
        // gelu(1) = 0.5·(1 + erf(1/√2)) ≈ 0.8413447
        assert!(close(gelu(1.0), 0.841_344_7));
        // gelu(-1) ≈ -0.1586553
        assert!(close(gelu(-1.0), -0.158_655_3));
        // Saturates to identity for large positive, to zero for large negative.
        assert!(gelu(8.0) > 7.999);
        assert!(gelu(-8.0).abs() < 1e-6);
    }

    #[test]
    fn sinusoidal_matches_marian_layout() {
        // Concatenated sin|cos (NOT interleaved): first ⌈dim/2⌉ sines, then cosines.
        // pos=0 → sin(0)=0 across the first half, cos(0)=1 across the second.
        let mut buf = [9.0f32; 8];
        sinusoidal_into(&mut buf, 0, 8);
        assert_eq!(buf, [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);

        // pos=1, dim=8 — ground truth dumped from HF MarianSinusoidalPositionalEmbedding.
        let mut buf = [0.0f32; 8];
        sinusoidal_into(&mut buf, 1, 8);
        let hf = [
            0.841471, 0.099833, 0.01, 0.001, 0.540302, 0.995004, 0.99995, 1.0,
        ];
        for (b, h) in buf.iter().zip(hf.iter()) {
            assert!(close(*b, *h), "{b} vs {h}");
        }

        // Odd dim: ⌈5/2⌉=3 sines then ⌊5/2⌋=2 cosines.
        let mut buf = [7.0f32; 5];
        sinusoidal_into(&mut buf, 0, 5);
        assert_eq!(buf, [0.0, 0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn embed_scale_is_sqrt_dim() {
        assert!(close(embed_scale(4), 2.0));
        assert!(close(embed_scale(512), 512.0f32.sqrt()));
    }

    #[test]
    fn add_bias_adds_in_place() {
        let mut out = [1.0, -2.0, 3.0];
        add_bias(&mut out, &[0.5, 0.5, -1.0]);
        assert_eq!(out, [1.5, -1.5, 2.0]);
    }

    #[test]
    fn attention_head_single_key_returns_that_value() {
        // One valid key ⇒ softmax is [1.0] ⇒ output is that position's value exactly,
        // whatever the query or scale.
        let keys = [0.3, -0.7, 1.1];
        let values = [5.0, 6.0, 7.0];
        let kv = KvHead {
            keys: &keys,
            values: &values,
            stride: 3,
            head_off: 0,
        };
        let q = [10.0, -4.0, 2.0];
        let mut out = [0.0; 3];
        let mut scores = [0.0; 1];
        attention_head(&mut out, &q, &kv, 1, 0.25, &mut scores);
        assert!(close(out[0], 5.0) && close(out[1], 6.0) && close(out[2], 7.0));
    }

    #[test]
    fn attention_head_equal_scores_average_the_values() {
        // Identical keys ⇒ equal scores ⇒ uniform softmax ⇒ the mean of the values.
        // head_size 2, 3 positions, stride 2, head_off 0.
        let keys = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let values = [0.0, 9.0, 3.0, 0.0, 6.0, 3.0];
        let kv = KvHead {
            keys: &keys,
            values: &values,
            stride: 2,
            head_off: 0,
        };
        let q = [0.5, -0.5];
        let mut out = [0.0; 2];
        let mut scores = [0.0; 3];
        attention_head(&mut out, &q, &kv, 3, 1.0, &mut scores);
        assert!(close(out[0], (0.0 + 3.0 + 6.0) / 3.0)); // 3.0
        assert!(close(out[1], (9.0 + 0.0 + 3.0) / 3.0)); // 4.0
    }

    #[test]
    fn attention_head_matches_inline_reference_with_offset() {
        // The helper must reproduce the original hand-written decoder loop bit-for-bit,
        // with a non-zero head_off and a stride wider than head_size (several heads
        // packed per cache row, as grouped-query attention produces).
        let (head_size, n_valid, stride, head_off) = (3usize, 4usize, 6usize, 3usize);
        let total = n_valid * stride;
        let keys: Vec<f32> = (0..total).map(|i| (i as f32 * 0.13).sin()).collect();
        let values: Vec<f32> = (0..total).map(|i| (i as f32 * 0.21).cos()).collect();
        let q: Vec<f32> = (0..head_size).map(|i| (i as f32 * 0.7).cos()).collect();
        let scale = 1.0 / (head_size as f32).sqrt();

        // Reference: exactly the loop forward() ran before attention_head existed.
        let mut ref_out = vec![0.0f32; head_size];
        let mut att = vec![0.0f32; n_valid];
        for (t, a) in att.iter_mut().enumerate() {
            let off = t * stride + head_off;
            let mut s = 0.0f32;
            for i in 0..head_size {
                s += q[i] * keys[off + i];
            }
            *a = s * scale;
        }
        softmax(&mut att);
        for (t, &a) in att.iter().enumerate() {
            let off = t * stride + head_off;
            for i in 0..head_size {
                ref_out[i] += a * values[off + i];
            }
        }

        let kv = KvHead {
            keys: &keys,
            values: &values,
            stride,
            head_off,
        };
        let mut out = vec![0.0f32; head_size];
        let mut scores = vec![0.0f32; n_valid];
        attention_head(&mut out, &q, &kv, n_valid, scale, &mut scores);
        assert_eq!(out, ref_out); // identical op order ⇒ exact
    }
}
