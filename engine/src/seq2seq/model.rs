//! The seq2seq forward passes: the Marian **encoder** ([`encode`]), the
//! autoregressive **decoder** ([`decode_step`] with [`precompute_cross_kv`]), and the
//! [`greedy_decode`] convenience that drives a full translation.
//!
//! [`encode`] runs the Marian encoder over a tokenized source sentence and leaves
//! the `src_len × d_model` encoder output in [`RunState::enc_x`] — the
//! `last_hidden_state` the decoder's cross-attention will read. It assembles the
//! shared `no_std` building-block kernels from [`crate::math`] (LayerNorm, biased
//! linears, GELU/swish, sinusoidal positions, the generalized attention helper)
//! into Marian's post-norm layer, following Hugging Face `MarianEncoder`
//! operation-for-operation so the output matches `encoder(...).last_hidden_state`.
//!
//! **Post-norm layer order** (Marian `normalize_before = False`, which every
//! OPUS-MT model — and Hugging Face's own `MarianEncoderLayer` — uses):
//! ```text
//! h = LN_attn(h + self_attn(h))      # bidirectional, all-to-all, no mask
//! h = LN_ffn (h + fc2(act(fc1(h))))  # position-wise feed-forward
//! ```
//! There is no embedding LayerNorm and no final encoder LayerNorm (the layer's two
//! norms are the only ones), and `q` is scaled by `head_dim^-1/2` on the scores —
//! all matching the reference. The token embedding is scaled by `√d_model` (when
//! `scale_embedding` is set) and the sinusoidal positions are *added*, computed on
//! the fly by [`math::sinusoidal_into`].

use crate::math::{self, Kernel, KvHead};
use crate::quant::{QuantScratch, QuantizedTensor};
use crate::seq2seq::config::{Activation, Config};
use crate::seq2seq::quantize::{KeptWeights, QuantizedWeights};
use crate::seq2seq::state::RunState;
use crate::seq2seq::weights::Weights;

/// Which matmul matrix a [`ModelWeights::matmul`] call should apply.
///
/// The 17 matmul matrices — the only weights that differ between the fp32 and int8 paths.
/// Everything else (biases, LayerNorms, `final_logits_bias`) is fp32 in both and read
/// through [`KeptWeights`]. `LmHead` is the tied token embedding used as the output
/// classifier.
#[derive(Clone, Copy)]
enum Proj {
    EncWq,
    EncWk,
    EncWv,
    EncWo,
    EncFc1,
    EncFc2,
    DecWq,
    DecWk,
    DecWv,
    DecWo,
    DecCrossWq,
    DecCrossWk,
    DecCrossWv,
    DecCrossWo,
    DecFc1,
    DecFc2,
    LmHead,
}

impl Proj {
    /// `(d_in, d_out)` for this projection — the dimensions [`math::matmul`] expects.
    /// Attention projections are `d × d`; the FFN matrices widen to / narrow from the ffn
    /// dimension; `LmHead` projects `d → vocab`. `LmHead` is addressed as "layer 0".
    fn shape(self, c: &Config) -> (usize, usize) {
        let d = c.d_model;
        match self {
            Proj::EncWq | Proj::EncWk | Proj::EncWv | Proj::EncWo => (d, d),
            Proj::DecWq | Proj::DecWk | Proj::DecWv | Proj::DecWo => (d, d),
            Proj::DecCrossWq | Proj::DecCrossWk | Proj::DecCrossWv | Proj::DecCrossWo => (d, d),
            Proj::EncFc1 => (d, c.enc_ffn),
            Proj::EncFc2 => (c.enc_ffn, d),
            Proj::DecFc1 => (d, c.dec_ffn),
            Proj::DecFc2 => (c.dec_ffn, d),
            Proj::LmHead => (d, c.vocab_size),
        }
    }
}

/// A Marian model's weights in either representation, so one set of forward passes serves
/// both.
///
/// The encoder/decoder passes are identical across the two arms except the matmul
/// matrices, which [`ModelWeights::matmul`] dispatches to the matching kernel
/// ([`math::matmul`]/[`matmul_simd`](math::matmul_simd) for `F32`,
/// [`matmul_w8a8`](math::matmul_w8a8) for `Q8`). The token embedding is read through
/// [`ModelWeights::embed`]; the biases and LayerNorms are fp32 in both and travel
/// separately in [`KeptWeights`].
// Both variants are bundles of slice views (no heap); the enum is built once per run and
// only ever passed by reference, and the engine is no_std — the variant-size gap is
// harmless here, exactly as on the Llama path.
#[allow(clippy::large_enum_variant)]
pub enum ModelWeights<'a> {
    /// Full-precision fp32 weights.
    F32(Weights<'a>),
    /// Group-wise int8-quantized matmul matrices.
    Q8(QuantizedWeights<'a>),
}

impl<'a> ModelWeights<'a> {
    /// Write token `token`'s embedding row (length `d`) into `out`.
    ///
    /// fp32 copies the row straight from the table; int8 dequantizes just that one row.
    /// (The seq2seq embedding is *not* pre-scaled here — the caller applies Marian's
    /// `√d_model` embedding scale afterward.)
    fn embed(&self, token: usize, d: usize, out: &mut [f32]) {
        match self {
            ModelWeights::F32(w) => {
                out.copy_from_slice(&w.token_embedding[token * d..token * d + d]);
            }
            ModelWeights::Q8(w) => {
                let row = w.token_embedding.layer(token, 1, d);
                crate::quant::dequantize(out, &row);
            }
        }
    }

    /// Apply projection `p` of layer `l`: `out = W · x`, dispatching on both the weight
    /// representation and the requested [`Kernel`]. For `LmHead` (a single matrix) pass
    /// `l = 0`.
    ///
    /// On the int8 (`Q8`) path this is **W8A8**: the activation `x` is quantized into the
    /// caller-provided [`QuantScratch`] and the matmul runs in integer arithmetic; `qs`
    /// must be `Some`. The fp32 path ignores `qs` (pass `None`) and, having no integer
    /// kernel, degrades `Dotprod` to SIMD.
    #[allow(clippy::too_many_arguments)]
    fn matmul(
        &self,
        p: Proj,
        l: usize,
        out: &mut [f32],
        x: &[f32],
        c: &Config,
        kernel: Kernel,
        qs: Option<&mut QuantScratch>,
    ) {
        let (d_in, d_out) = p.shape(c);
        match self {
            ModelWeights::F32(w) => {
                let n = d_in * d_out;
                let wl = &f32_proj(w, p)[l * n..l * n + n];
                match kernel {
                    Kernel::Scalar => math::matmul(out, x, wl, d_in, d_out),
                    Kernel::Simd | Kernel::Dotprod => math::matmul_simd(out, x, wl, d_in, d_out),
                }
            }
            ModelWeights::Q8(w) => {
                let wl = q8_proj(w, p).layer(l, d_out, d_in);
                let qs = qs.expect("the Q8 forward path requires a QuantScratch");
                math::matmul_w8a8(kernel, out, x, &wl, d_in, d_out, qs);
            }
        }
    }
}

/// The flat fp32 slice backing projection `p` (`LmHead` is the tied token embedding).
fn f32_proj<'a>(w: &Weights<'a>, p: Proj) -> &'a [f32] {
    match p {
        Proj::EncWq => w.enc_wq,
        Proj::EncWk => w.enc_wk,
        Proj::EncWv => w.enc_wv,
        Proj::EncWo => w.enc_wo,
        Proj::EncFc1 => w.enc_fc1_w,
        Proj::EncFc2 => w.enc_fc2_w,
        Proj::DecWq => w.dec_wq,
        Proj::DecWk => w.dec_wk,
        Proj::DecWv => w.dec_wv,
        Proj::DecWo => w.dec_wo,
        Proj::DecCrossWq => w.dec_cross_wq,
        Proj::DecCrossWk => w.dec_cross_wk,
        Proj::DecCrossWv => w.dec_cross_wv,
        Proj::DecCrossWo => w.dec_cross_wo,
        Proj::DecFc1 => w.dec_fc1_w,
        Proj::DecFc2 => w.dec_fc2_w,
        Proj::LmHead => w.token_embedding,
    }
}

/// The quantized tensor backing projection `p` (`LmHead` is the tied token embedding).
fn q8_proj<'a>(w: &QuantizedWeights<'a>, p: Proj) -> QuantizedTensor<'a> {
    match p {
        Proj::EncWq => w.enc_wq,
        Proj::EncWk => w.enc_wk,
        Proj::EncWv => w.enc_wv,
        Proj::EncWo => w.enc_wo,
        Proj::EncFc1 => w.enc_fc1_w,
        Proj::EncFc2 => w.enc_fc2_w,
        Proj::DecWq => w.dec_wq,
        Proj::DecWk => w.dec_wk,
        Proj::DecWv => w.dec_wv,
        Proj::DecWo => w.dec_wo,
        Proj::DecCrossWq => w.dec_cross_wq,
        Proj::DecCrossWk => w.dec_cross_wk,
        Proj::DecCrossWv => w.dec_cross_wv,
        Proj::DecCrossWo => w.dec_cross_wo,
        Proj::DecFc1 => w.dec_fc1_w,
        Proj::DecFc2 => w.dec_fc2_w,
        Proj::LmHead => w.token_embedding,
    }
}

/// Slice tensor `t`'s layer-`l` block of `len` elements: `t[l*len ..][..len]`.
///
/// Every seq2seq weight is stored flat across layers, so a per-layer projection or
/// bias is one of these contiguous windows.
#[inline]
fn layer(t: &[f32], l: usize, len: usize) -> &[f32] {
    &t[l * len..l * len + len]
}

/// Apply the model's feed-forward activation to every element of `buf` in place.
///
/// Marian reads this from the checkpoint (`gelu` or `swish`); the engine supports
/// both and never hardcodes one.
fn activate(buf: &mut [f32], act: Activation) {
    match act {
        Activation::Gelu => {
            for v in buf.iter_mut() {
                *v = math::gelu(*v);
            }
        }
        Activation::Swish => {
            for v in buf.iter_mut() {
                *v = math::silu(*v);
            }
        }
    }
}

/// Run the Marian encoder over the source token ids, returning the encoder output.
///
/// Fills (and returns a view of) [`RunState::enc_x`] with the `src_len × d`
/// `last_hidden_state`. `tokens` holds the source token ids; `tokens.len()` is the
/// source length and must not exceed the `src_len` the state was sized for
/// ([`RunState::new`]). Allocates nothing — all scratch is the pre-carved state.
///
/// The matmul weights come through [`ModelWeights`] (fp32 or int8) and the biases /
/// LayerNorms through [`KeptWeights`]; `kernel` selects the matmul implementation. On the
/// int8 path `qs` must be `Some` (the W8A8 activation scratch); on the fp32 path pass
/// `None`. The representation and kernel change the output only by quantization /
/// fp32-rounding noise. Implements **post-norm** Marian; a pre-norm checkpoint
/// (`norm_before`, which no OPUS-MT model sets) trips a debug assertion.
///
/// # Panics
/// In debug builds, via the kernels' length assertions, if `tokens` is longer than the
/// state's `src_len`, the model is pre-norm, or `mw` is `Q8` and `qs` is `None`.
pub fn encode<'s>(
    c: &Config,
    mw: &ModelWeights,
    kept: &KeptWeights,
    s: &'s mut RunState,
    tokens: &[usize],
    kernel: Kernel,
    qs: &mut Option<QuantScratch>,
) -> &'s [f32] {
    debug_assert!(
        !c.norm_before,
        "seq2seq encode implements post-norm Marian only"
    );
    let d = c.d_model;
    let n = tokens.len();
    debug_assert!(
        n * d <= s.enc_x.len(),
        "source longer than the state's src_len"
    );

    // 1. Token embedding × scale, plus the sinusoidal positions.
    let scale = if c.scale_embedding {
        math::embed_scale(d)
    } else {
        1.0
    };
    for (t, &tok) in tokens.iter().enumerate() {
        let base = t * d;
        mw.embed(tok, d, &mut s.enc_x[base..base + d]);
        if c.scale_embedding {
            for v in &mut s.enc_x[base..base + d] {
                *v *= scale;
            }
        }
        math::sinusoidal_into(s.norm, t, d);
        math::accumulate(&mut s.enc_x[base..base + d], &s.norm[..]);
    }

    // 2. Encoder layers (post-norm).
    let heads = c.enc_heads;
    let hd = c.enc_head_dim();
    let attn_scale = 1.0 / libm::sqrtf(hd as f32);
    for l in 0..c.enc_layers {
        // --- self-attention sublayer ---
        // Project K and V for every position from the (raw) residual stream.
        let (bk, bv) = (layer(kept.enc_bk, l, d), layer(kept.enc_bv, l, d));
        for t in 0..n {
            let base = t * d;
            mw.matmul(
                Proj::EncWk,
                l,
                &mut s.enc_k[base..base + d],
                &s.enc_x[base..base + d],
                c,
                kernel,
                qs.as_mut(),
            );
            math::add_bias(&mut s.enc_k[base..base + d], bk);
            mw.matmul(
                Proj::EncWv,
                l,
                &mut s.enc_v[base..base + d],
                &s.enc_x[base..base + d],
                c,
                kernel,
                qs.as_mut(),
            );
            math::add_bias(&mut s.enc_v[base..base + d], bv);
        }

        // For each query position, attend over all keys (bidirectional), project,
        // and stash the attention output in `enc_sub`.
        let (bq, bo) = (layer(kept.enc_bq, l, d), layer(kept.enc_bo, l, d));
        for qi in 0..n {
            let base = qi * d;
            mw.matmul(Proj::EncWq, l, s.q, &s.enc_x[base..base + d], c, kernel, qs.as_mut());
            math::add_bias(s.q, bq);
            for h in 0..heads {
                let off = h * hd;
                let kv = KvHead {
                    keys: &s.enc_k[..],
                    values: &s.enc_v[..],
                    stride: d,
                    head_off: off,
                };
                math::attention_head(
                    &mut s.attn[off..off + hd],
                    &s.q[off..off + hd],
                    &kv,
                    n, // bidirectional: every source position is a valid key
                    attn_scale,
                    s.enc_scores,
                );
            }
            mw.matmul(
                Proj::EncWo,
                l,
                &mut s.enc_sub[base..base + d],
                &s.attn[..],
                c,
                kernel,
                qs.as_mut(),
            );
            math::add_bias(&mut s.enc_sub[base..base + d], bo);
        }

        // Residual + LayerNorm (post-norm): h = LN(h + attn_out). Folding the
        // residual into `enc_sub` lets LayerNorm read it and write `enc_x` without
        // aliasing (the two are distinct buffers).
        let (ln_w, ln_b) = (layer(kept.enc_ln_att_w, l, d), layer(kept.enc_ln_att_b, l, d));
        for t in 0..n {
            let base = t * d;
            math::accumulate(&mut s.enc_sub[base..base + d], &s.enc_x[base..base + d]);
            math::layernorm(
                &mut s.enc_x[base..base + d],
                &s.enc_sub[base..base + d],
                ln_w,
                ln_b,
            );
        }

        // --- feed-forward sublayer (position-wise) ---
        let (fc1_b, fc2_b) = (layer(kept.enc_fc1_b, l, c.enc_ffn), layer(kept.enc_fc2_b, l, d));
        let (lnf_w, lnf_b) = (layer(kept.enc_ln_ffn_w, l, d), layer(kept.enc_ln_ffn_b, l, d));
        for t in 0..n {
            let base = t * d;
            let hbuf = &mut s.ffn[..c.enc_ffn];
            mw.matmul(Proj::EncFc1, l, hbuf, &s.enc_x[base..base + d], c, kernel, qs.as_mut());
            math::add_bias(hbuf, fc1_b);
            activate(hbuf, c.activation);
            mw.matmul(
                Proj::EncFc2,
                l,
                &mut s.enc_sub[base..base + d],
                &s.ffn[..c.enc_ffn],
                c,
                kernel,
                qs.as_mut(),
            );
            math::add_bias(&mut s.enc_sub[base..base + d], fc2_b);
            math::accumulate(&mut s.enc_sub[base..base + d], &s.enc_x[base..base + d]);
            math::layernorm(
                &mut s.enc_x[base..base + d],
                &s.enc_sub[base..base + d],
                lnf_w,
                lnf_b,
            );
        }
    }

    &s.enc_x[..n * d]
}

/// Project the encoder output into the cross-attention K/V cache, once per decode.
///
/// The decoder's cross-attention keys and values come from the (fixed) encoder
/// output [`RunState::enc_x`], so they depend only on the source sentence, not
/// on the target position — project them a single time up front and every
/// [`decode_step`] reads them. Fills [`RunState::cross_k`] / `cross_v`, laid out
/// `[dec_layers, src_len, d]`. Run [`encode`] first (it fills `enc_x`), then this,
/// then the [`decode_step`] loop.
pub fn precompute_cross_kv(
    c: &Config,
    mw: &ModelWeights,
    kept: &KeptWeights,
    s: &mut RunState,
    src_len: usize,
    kernel: Kernel,
    qs: &mut Option<QuantScratch>,
) {
    let d = c.d_model;
    for l in 0..c.dec_layers {
        let (bk, bv) = (layer(kept.dec_cross_bk, l, d), layer(kept.dec_cross_bv, l, d));
        let lbase = l * src_len * d; // src_len is constant for the whole decode
        for t in 0..src_len {
            let off = lbase + t * d;
            let src = t * d;
            mw.matmul(
                Proj::DecCrossWk,
                l,
                &mut s.cross_k[off..off + d],
                &s.enc_x[src..src + d],
                c,
                kernel,
                qs.as_mut(),
            );
            math::add_bias(&mut s.cross_k[off..off + d], bk);
            mw.matmul(
                Proj::DecCrossWv,
                l,
                &mut s.cross_v[off..off + d],
                &s.enc_x[src..src + d],
                c,
                kernel,
                qs.as_mut(),
            );
            math::add_bias(&mut s.cross_v[off..off + d], bv);
        }
    }
}

/// Run one decoder step for target token `token` at position `pos`, returning the
/// vocabulary logits.
///
/// The autoregressive counterpart of [`encode`]: one token in, a full row of logits
/// out, appending this position's key/value to the self-attention cache. Each Marian
/// decoder layer is three post-norm sublayers — masked (causal) self-attention,
/// cross-attention over the encoder output, and the feed-forward — and the tied
/// lm_head plus `final_logits_bias` produce the logits. Greedy decoding feeds the
/// argmax back in as the next `token`.
///
/// Call [`encode`] then [`precompute_cross_kv`] once before the step loop; `pos` runs
/// `0, 1, …` from the decoder start token, and `src_len` is the encoder length (the
/// number of cross-attention keys). The self-KV cache stride is read from the state,
/// so the state may be sized for fewer than `max_tgt` steps. As in [`encode`], `mw`
/// carries the matmul weights (fp32 or int8), `kept` the fp32 biases / LayerNorms /
/// `final_logits_bias`, and `qs` the W8A8 scratch (`Some` only on the int8 path).
///
/// # Panics
/// In debug builds, via the kernels' length assertions, if the model is pre-norm, or if
/// `mw` is `Q8` and `qs` is `None`.
#[allow(clippy::too_many_arguments)]
pub fn decode_step<'s>(
    c: &Config,
    mw: &ModelWeights,
    kept: &KeptWeights,
    s: &'s mut RunState<'_>,
    token: usize,
    pos: usize,
    src_len: usize,
    kernel: Kernel,
    qs: &mut Option<QuantScratch>,
) -> &'s [f32] {
    debug_assert!(
        !c.norm_before,
        "seq2seq decode implements post-norm Marian only"
    );
    let d = c.d_model;
    let heads = c.dec_heads;
    let hd = c.dec_head_dim();
    let attn_scale = 1.0 / libm::sqrtf(hd as f32);
    // Self-KV cache stride: rows per layer = the target capacity the state was sized
    // for (derived from the buffer, so it need not equal `max_tgt`).
    let tgt_cap = s.self_k.len() / (c.dec_layers * d);

    // 1. Embed the current target token (× scale), add the sinusoidal position.
    //    No RoPE — Marian positions are additive, applied once, like the encoder.
    let scale = if c.scale_embedding {
        math::embed_scale(d)
    } else {
        1.0
    };
    mw.embed(token, d, &mut s.x[..d]);
    if c.scale_embedding {
        for v in &mut s.x[..d] {
            *v *= scale;
        }
    }
    math::sinusoidal_into(s.norm, pos, d);
    math::accumulate(&mut s.x[..d], &s.norm[..d]);

    for l in 0..c.dec_layers {
        // --- masked self-attention sublayer ---
        let (bq, bk) = (layer(kept.dec_bq, l, d), layer(kept.dec_bk, l, d));
        let (bv, bo) = (layer(kept.dec_bv, l, d), layer(kept.dec_bo, l, d));

        let self_base = l * tgt_cap * d; // base of layer l in the self caches
        let cur = self_base + pos * d; // this position's row
        {
            // K and V for this position go straight into the cache; Q into `q`.
            let krow = &mut s.self_k[cur..cur + d];
            let vrow = &mut s.self_v[cur..cur + d];
            mw.matmul(Proj::DecWq, l, s.q, &s.x[..d], c, kernel, qs.as_mut());
            math::add_bias(s.q, bq);
            mw.matmul(Proj::DecWk, l, krow, &s.x[..d], c, kernel, qs.as_mut());
            math::add_bias(krow, bk);
            mw.matmul(Proj::DecWv, l, vrow, &s.x[..d], c, kernel, qs.as_mut());
            math::add_bias(vrow, bv);
        }

        // Attend over the causal prefix 0..=pos.
        for h in 0..heads {
            let off = h * hd;
            let kv = KvHead {
                keys: &s.self_k[self_base..],
                values: &s.self_v[self_base..],
                stride: d,
                head_off: off,
            };
            math::attention_head(
                &mut s.attn[off..off + hd],
                &s.q[off..off + hd],
                &kv,
                pos + 1, // causal: keys 0..=pos are valid
                attn_scale,
                s.scores,
            );
        }

        // Output projection, residual + LayerNorm. `enc_sub[..d]` is free here
        // (encoder + cross-KV precompute are done) and serves as sublayer scratch.
        mw.matmul(Proj::DecWo, l, &mut s.enc_sub[..d], &s.attn[..], c, kernel, qs.as_mut());
        math::add_bias(&mut s.enc_sub[..d], bo);
        math::accumulate(&mut s.enc_sub[..d], &s.x[..d]);
        let (ln_w, ln_b) = (layer(kept.dec_ln_self_w, l, d), layer(kept.dec_ln_self_b, l, d));
        math::layernorm(&mut s.x[..d], &s.enc_sub[..d], ln_w, ln_b);

        // --- cross-attention sublayer ---
        // Q from the decoder state; K/V precomputed from the encoder output.
        let (bq, bo) = (layer(kept.dec_cross_bq, l, d), layer(kept.dec_cross_bo, l, d));
        let cross_base = l * src_len * d;

        mw.matmul(Proj::DecCrossWq, l, s.q, &s.x[..d], c, kernel, qs.as_mut());
        math::add_bias(s.q, bq);
        for h in 0..heads {
            let off = h * hd;
            let kv = KvHead {
                keys: &s.cross_k[cross_base..],
                values: &s.cross_v[cross_base..],
                stride: d,
                head_off: off,
            };
            math::attention_head(
                &mut s.attn[off..off + hd],
                &s.q[off..off + hd],
                &kv,
                src_len, // no mask: every source position is a valid key
                attn_scale,
                s.scores,
            );
        }

        mw.matmul(Proj::DecCrossWo, l, &mut s.enc_sub[..d], &s.attn[..], c, kernel, qs.as_mut());
        math::add_bias(&mut s.enc_sub[..d], bo);
        math::accumulate(&mut s.enc_sub[..d], &s.x[..d]);
        let (ln_w, ln_b) = (layer(kept.dec_ln_cross_w, l, d), layer(kept.dec_ln_cross_b, l, d));
        math::layernorm(&mut s.x[..d], &s.enc_sub[..d], ln_w, ln_b);

        // --- feed-forward sublayer (post-norm) ---
        let (fc1_b, fc2_b) = (layer(kept.dec_fc1_b, l, c.dec_ffn), layer(kept.dec_fc2_b, l, d));
        let (lnf_w, lnf_b) = (layer(kept.dec_ln_ffn_w, l, d), layer(kept.dec_ln_ffn_b, l, d));

        let hbuf = &mut s.ffn[..c.dec_ffn];
        mw.matmul(Proj::DecFc1, l, hbuf, &s.x[..d], c, kernel, qs.as_mut());
        math::add_bias(hbuf, fc1_b);
        activate(hbuf, c.activation);
        mw.matmul(
            Proj::DecFc2,
            l,
            &mut s.enc_sub[..d],
            &s.ffn[..c.dec_ffn],
            c,
            kernel,
            qs.as_mut(),
        );
        math::add_bias(&mut s.enc_sub[..d], fc2_b);
        math::accumulate(&mut s.enc_sub[..d], &s.x[..d]);
        math::layernorm(&mut s.x[..d], &s.enc_sub[..d], lnf_w, lnf_b);
    }

    // Output projection: TIED to the token embedding ([vocab, d] = [out, in]),
    // then add Marian's final_logits_bias. No embed_scale on the output side.
    mw.matmul(Proj::LmHead, 0, s.logits, &s.x[..d], c, kernel, qs.as_mut());
    math::add_bias(s.logits, kept.final_logits_bias);
    &s.logits[..]
}

/// Greedily translate `src_ids`, writing the target token ids into `out_ids` and
/// returning how many were written.
///
/// The whole pipeline end to end: [`encode`] the source, [`precompute_cross_kv`],
/// then loop [`decode_step`] feeding back the argmax — starting from the decoder-start
/// token (`pad_id`) at position 0 — until the model emits `eos_id` (not written) or
/// `out_ids` fills. Size `out_ids` to `max_tgt`, and create the state with
/// `src_len = src_ids.len()` and a `tgt_len` at least as large as `out_ids`. Allocates
/// nothing; the argmax tie-break (first maximum) matches Hugging Face greedy decoding.
///
/// For beam search or custom stopping, drive [`encode`] / [`precompute_cross_kv`] /
/// [`decode_step`] directly instead — this is the greedy convenience built on them.
#[allow(clippy::too_many_arguments)]
pub fn greedy_decode(
    c: &Config,
    mw: &ModelWeights,
    kept: &KeptWeights,
    s: &mut RunState,
    src_ids: &[usize],
    out_ids: &mut [usize],
    kernel: Kernel,
    qs: &mut Option<QuantScratch>,
) -> usize {
    encode(c, mw, kept, s, src_ids, kernel, qs);
    precompute_cross_kv(c, mw, kept, s, src_ids.len(), kernel, qs);

    let mut token = c.pad_id;
    let mut n = 0;
    for (pos, slot) in out_ids.iter_mut().enumerate() {
        let logits = decode_step(c, mw, kept, s, token, pos, src_ids.len(), kernel, qs);
        let next = math::argmax(logits);
        if next == c.eos_id {
            break;
        }
        *slot = next;
        token = next;
        n += 1;
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::Arena;
    use crate::seq2seq::config::Activation;
    use crate::seq2seq::memory::seq2seq_arena_floats;
    use crate::seq2seq::weights::weight_floats;

    extern crate std;
    use std::vec;
    use std::vec::Vec;

    fn tiny_config(activation: Activation) -> Config {
        Config {
            d_model: 8,
            enc_layers: 2,
            dec_layers: 1,
            enc_heads: 2,
            dec_heads: 2,
            enc_ffn: 16,
            dec_ffn: 16,
            vocab_size: 12,
            max_src: 6,
            max_tgt: 5,
            pad_id: 11,
            eos_id: 0,
            bos_id: 0,
            norm_before: false,
            activation,
            scale_embedding: true,
        }
    }

    /// Deterministic small pseudo-random weights, so the encoder produces finite,
    /// non-trivial output without overflow.
    fn fake_weights(c: &Config) -> Vec<f32> {
        let mut v = Vec::with_capacity(weight_floats(c));
        let mut state: u32 = 0x9e37_79b9;
        for _ in 0..weight_floats(c) {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            v.push((state as f32 / u32::MAX as f32 - 0.5) * 0.2);
        }
        v
    }

    fn run(c: &Config, wbuf: &[f32], tokens: &[usize]) -> Vec<f32> {
        let w = Weights::new(wbuf, c).unwrap();
        let kept = KeptWeights::from_weights(&w);
        let mw = ModelWeights::F32(w);
        let mut buf = vec![0.0f32; seq2seq_arena_floats(c, tokens.len(), 0)];
        let mut arena = Arena::new(&mut buf);
        let mut s = RunState::new(&mut arena, c, tokens.len(), 0).unwrap();
        encode(c, &mw, &kept, &mut s, tokens, Kernel::Simd, &mut None).to_vec()
    }

    #[test]
    fn encode_produces_finite_output_of_right_shape() {
        for act in [Activation::Swish, Activation::Gelu] {
            let c = tiny_config(act);
            let wbuf = fake_weights(&c);
            let tokens = [1usize, 5, 2, 7, 3];
            let out = run(&c, &wbuf, &tokens);
            assert_eq!(out.len(), tokens.len() * c.d_model);
            assert!(out.iter().all(|x| x.is_finite()));
        }
    }

    #[test]
    fn encode_is_deterministic() {
        let c = tiny_config(Activation::Swish);
        let wbuf = fake_weights(&c);
        let tokens = [4usize, 0, 9, 1];
        assert_eq!(run(&c, &wbuf, &tokens), run(&c, &wbuf, &tokens));
    }

    #[test]
    fn encode_is_bidirectional() {
        // Each position's encoder output depends on the *whole* sequence (all-to-all
        // attention), so changing a later token must perturb an earlier position's
        // output — the property that distinguishes the encoder from a causal decoder.
        let c = tiny_config(Activation::Swish);
        let wbuf = fake_weights(&c);
        let a = run(&c, &wbuf, &[3usize, 5, 2]);
        let b = run(&c, &wbuf, &[3usize, 5, 8]); // only the last token differs
        let d = c.d_model;
        // Position 0's output row differs because attention at pos 0 sees pos 2.
        assert!(a[..d]
            .iter()
            .zip(&b[..d])
            .any(|(x, y)| (x - y).abs() > 1e-6));
    }

    #[test]
    fn layernorm_output_is_normalized() {
        // After the final post-norm LayerNorm (unit-ish gains in these fake weights),
        // each output row should be roughly zero-mean — a cheap sanity check that the
        // norm actually ran last.
        let c = tiny_config(Activation::Swish);
        let wbuf = fake_weights(&c);
        let tokens = [1usize, 2, 3];
        let out = run(&c, &wbuf, &tokens);
        let d = c.d_model;
        for row in out.chunks(d) {
            let mean: f32 = row.iter().sum::<f32>() / d as f32;
            // The LN bias is tiny (fake weights ~±0.1), so the mean stays small.
            assert!(mean.abs() < 0.5, "row mean {mean} too large");
        }
    }

    fn decode_ids(c: &Config, wbuf: &[f32], src: &[usize], kernel: Kernel) -> Vec<usize> {
        let w = Weights::new(wbuf, c).unwrap();
        let kept = KeptWeights::from_weights(&w);
        let mw = ModelWeights::F32(w);
        let mut buf = vec![0.0f32; seq2seq_arena_floats(c, src.len(), c.max_tgt)];
        let mut arena = Arena::new(&mut buf);
        let mut s = RunState::new(&mut arena, c, src.len(), c.max_tgt).unwrap();
        let mut out = vec![0usize; c.max_tgt];
        let n = greedy_decode(c, &mw, &kept, &mut s, src, &mut out, kernel, &mut None);
        out.truncate(n);
        out
    }

    #[test]
    fn greedy_decode_is_deterministic_and_in_vocab() {
        for act in [Activation::Swish, Activation::Gelu] {
            let c = tiny_config(act);
            let wbuf = fake_weights(&c);
            let src = [1usize, 5, 2, 7];
            let a = decode_ids(&c, &wbuf, &src, Kernel::Simd);
            assert_eq!(a, decode_ids(&c, &wbuf, &src, Kernel::Simd)); // deterministic
            // The SIMD and scalar kernels pick the same greedy tokens.
            assert_eq!(a, decode_ids(&c, &wbuf, &src, Kernel::Scalar));
            assert!(a.len() <= c.max_tgt);
            assert!(a.iter().all(|&t| t < c.vocab_size));
            assert!(a.iter().all(|&t| t != c.eos_id)); // eos triggers the stop, unwritten
        }
    }

    #[test]
    fn decode_step_logits_are_finite_and_sized() {
        let c = tiny_config(Activation::Swish);
        let wbuf = fake_weights(&c);
        let w = Weights::new(&wbuf, &c).unwrap();
        let kept = KeptWeights::from_weights(&w);
        let mw = ModelWeights::F32(w);
        let src = [3usize, 1, 4];
        let mut buf = vec![0.0f32; seq2seq_arena_floats(&c, src.len(), c.max_tgt)];
        let mut arena = Arena::new(&mut buf);
        let mut s = RunState::new(&mut arena, &c, src.len(), c.max_tgt).unwrap();
        encode(&c, &mw, &kept, &mut s, &src, Kernel::Simd, &mut None);
        precompute_cross_kv(&c, &mw, &kept, &mut s, src.len(), Kernel::Simd, &mut None);
        let logits = decode_step(&c, &mw, &kept, &mut s, c.pad_id, 0, src.len(), Kernel::Simd, &mut None);
        assert_eq!(logits.len(), c.vocab_size);
        assert!(logits.iter().all(|x| x.is_finite()));
    }

    /// Build the int8 buffers + views for the int8 (Q8) path from fp32 weights.
    fn quantize(c: &Config, wbuf: &[f32], gs: usize) -> (Vec<i8>, Vec<f32>, Vec<f32>) {
        use crate::seq2seq::quantize::{
            kept_floats, pack_kept, quantize_weights, quantized_scale_count, quantized_weight_count,
        };
        let fp32 = Weights::new(wbuf, c).unwrap();
        let mut data = vec![0i8; quantized_weight_count(c)];
        let mut scales = vec![0.0f32; quantized_scale_count(c, gs)];
        quantize_weights(&fp32, &mut data, &mut scales, gs, c);
        let mut kept = vec![0.0f32; kept_floats(c)];
        pack_kept(&fp32, &mut kept, c);
        (data, scales, kept)
    }

    #[test]
    fn q8_decode_tracks_fp32_and_kernels_agree() {
        use crate::quant::QuantScratch;
        use crate::seq2seq::quantize::max_proj_d_in;

        let c = tiny_config(Activation::Swish);
        let wbuf = fake_weights(&c);
        let src = [1usize, 5, 2, 7];
        let gs = 4; // divides d_model (8) and the ffn dims (16)

        // One full pipeline (encode → cross-KV → first decode step), returning the logits.
        let decode = |mw: &ModelWeights, kept: &KeptWeights, kernel: Kernel| -> Vec<f32> {
            let n = max_proj_d_in(&c);
            let (mut xq, mut sc, mut gsum) = (vec![0i8; n], vec![0.0f32; n], vec![0.0f32; n]);
            let mut qs = matches!(mw, ModelWeights::Q8(_))
                .then(|| QuantScratch::new(&mut xq, &mut sc, &mut gsum, n).unwrap());
            let mut buf = vec![0.0f32; seq2seq_arena_floats(&c, src.len(), c.max_tgt)];
            let mut arena = Arena::new(&mut buf);
            let mut s = RunState::new(&mut arena, &c, src.len(), c.max_tgt).unwrap();
            encode(&c, mw, kept, &mut s, &src, kernel, &mut qs);
            precompute_cross_kv(&c, mw, kept, &mut s, src.len(), kernel, &mut qs);
            decode_step(&c, mw, kept, &mut s, c.pad_id, 0, src.len(), kernel, &mut qs).to_vec()
        };

        let fp32 = Weights::new(&wbuf, &c).unwrap();
        let fp_kept = KeptWeights::from_weights(&fp32);
        let mw_fp = ModelWeights::F32(fp32);
        let fp_logits = decode(&mw_fp, &fp_kept, Kernel::Scalar);

        let (data, scales, kept_buf) = quantize(&c, &wbuf, gs);
        let qw = QuantizedWeights::new(&data, &scales, gs, &c).unwrap();
        let q_kept = KeptWeights::carve(&kept_buf, &c).unwrap();
        let mw_q8 = ModelWeights::Q8(qw);

        let q_scalar = decode(&mw_q8, &q_kept, Kernel::Scalar);
        let q_simd = decode(&mw_q8, &q_kept, Kernel::Simd);

        // The int8 scalar and SIMD kernels accumulate in i32, so they agree *exactly*.
        assert_eq!(q_scalar, q_simd);
        assert!(q_scalar.iter().all(|x| x.is_finite()));
        // The int8 logits track the fp32 logits: quantization perturbs but does not change
        // the answer's scale.
        let max_diff = fp_logits
            .iter()
            .zip(&q_scalar)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let max_abs = fp_logits.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(
            max_diff < 0.1 * max_abs,
            "q8 logits drifted from fp32: max_diff={max_diff}, max_abs={max_abs}"
        );
    }
}
