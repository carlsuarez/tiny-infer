//! The seq2seq forward passes. Milestone 11 implements the **encoder**; the
//! decoder and greedy decode follow in a later milestone.
//!
//! [`encode`] runs the Marian encoder over a tokenized source sentence and leaves
//! the `src_len × d_model` encoder output in [`Seq2SeqState::enc_x`] — the
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

use crate::math::{self, KvHead};
use crate::seq2seq::config::{Activation, Seq2SeqConfig};
use crate::seq2seq::state::Seq2SeqState;
use crate::seq2seq::weights::Seq2SeqWeights;

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
/// Fills (and returns a view of) [`Seq2SeqState::enc_x`] with the `src_len × d`
/// `last_hidden_state`. `tokens` holds the source token ids; `tokens.len()` is the
/// source length and must not exceed the `src_len` the state was sized for
/// ([`Seq2SeqState::new`]). Allocates nothing — all scratch is the pre-carved state.
///
/// Uses the scalar [`math::matmul`] reference kernel (the milestone's bar is
/// correctness; the SIMD / int8 kernels join the seq2seq path in a later
/// milestone). Implements **post-norm** Marian; a pre-norm checkpoint
/// (`norm_before`, which no OPUS-MT model sets) trips a debug assertion.
///
/// # Panics
/// In debug builds, via the kernels' length assertions, if `tokens` is longer than
/// the state's `src_len` or the model is pre-norm.
pub fn encode<'s>(
    c: &Seq2SeqConfig,
    w: &Seq2SeqWeights,
    s: &'s mut Seq2SeqState,
    tokens: &[usize],
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
        s.enc_x[base..base + d].copy_from_slice(&w.token_embedding[tok * d..tok * d + d]);
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
        let (wk, bk) = (layer(w.enc_wk, l, d * d), layer(w.enc_bk, l, d));
        let (wv, bv) = (layer(w.enc_wv, l, d * d), layer(w.enc_bv, l, d));
        for t in 0..n {
            let base = t * d;
            math::matmul(
                &mut s.enc_k[base..base + d],
                &s.enc_x[base..base + d],
                wk,
                d,
                d,
            );
            math::add_bias(&mut s.enc_k[base..base + d], bk);
            math::matmul(
                &mut s.enc_v[base..base + d],
                &s.enc_x[base..base + d],
                wv,
                d,
                d,
            );
            math::add_bias(&mut s.enc_v[base..base + d], bv);
        }

        // For each query position, attend over all keys (bidirectional), project,
        // and stash the attention output in `enc_sub`.
        let (wq, bq) = (layer(w.enc_wq, l, d * d), layer(w.enc_bq, l, d));
        let (wo, bo) = (layer(w.enc_wo, l, d * d), layer(w.enc_bo, l, d));
        for qi in 0..n {
            let base = qi * d;
            math::matmul(s.q, &s.enc_x[base..base + d], wq, d, d);
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
            math::matmul(&mut s.enc_sub[base..base + d], &s.attn[..], wo, d, d);
            math::add_bias(&mut s.enc_sub[base..base + d], bo);
        }

        // Residual + LayerNorm (post-norm): h = LN(h + attn_out). Folding the
        // residual into `enc_sub` lets LayerNorm read it and write `enc_x` without
        // aliasing (the two are distinct buffers).
        let (ln_w, ln_b) = (layer(w.enc_ln_att_w, l, d), layer(w.enc_ln_att_b, l, d));
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
        let (fc1_w, fc1_b) = (
            layer(w.enc_fc1_w, l, c.enc_ffn * d),
            layer(w.enc_fc1_b, l, c.enc_ffn),
        );
        let (fc2_w, fc2_b) = (
            layer(w.enc_fc2_w, l, d * c.enc_ffn),
            layer(w.enc_fc2_b, l, d),
        );
        let (lnf_w, lnf_b) = (layer(w.enc_ln_ffn_w, l, d), layer(w.enc_ln_ffn_b, l, d));
        for t in 0..n {
            let base = t * d;
            let hbuf = &mut s.ffn[..c.enc_ffn];
            math::matmul(hbuf, &s.enc_x[base..base + d], fc1_w, d, c.enc_ffn);
            math::add_bias(hbuf, fc1_b);
            activate(hbuf, c.activation);
            math::matmul(
                &mut s.enc_sub[base..base + d],
                &s.ffn[..c.enc_ffn],
                fc2_w,
                c.enc_ffn,
                d,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::Arena;
    use crate::seq2seq::config::Activation;
    use crate::seq2seq::memory::seq2seq_arena_floats;
    use crate::seq2seq::weights::seq2seq_weight_floats;

    extern crate std;
    use std::vec;
    use std::vec::Vec;

    fn tiny_config(activation: Activation) -> Seq2SeqConfig {
        Seq2SeqConfig {
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
    fn fake_weights(c: &Seq2SeqConfig) -> Vec<f32> {
        let mut v = Vec::with_capacity(seq2seq_weight_floats(c));
        let mut state: u32 = 0x9e37_79b9;
        for _ in 0..seq2seq_weight_floats(c) {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            v.push((state as f32 / u32::MAX as f32 - 0.5) * 0.2);
        }
        v
    }

    fn run(c: &Seq2SeqConfig, wbuf: &[f32], tokens: &[usize]) -> Vec<f32> {
        let w = Seq2SeqWeights::new(wbuf, c).unwrap();
        let mut buf = vec![0.0f32; seq2seq_arena_floats(c, tokens.len(), 0)];
        let mut arena = Arena::new(&mut buf);
        let mut s = Seq2SeqState::new(&mut arena, c, tokens.len(), 0).unwrap();
        encode(c, &w, &mut s, tokens).to_vec()
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
}
