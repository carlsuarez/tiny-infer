//! The fp32 forward pass: one token in, a full row of logits out.
//!
//! [`forward`] runs a single decoder step for one position, reading and writing the
//! buffers in [`RunState`] and appending this position's key/value vectors to the
//! cache. It follows Karpathy's llama2.c `forward()` operation-for-operation so the
//! greedy (temperature-0) token stream matches bit-for-bit.
//!
//! Per layer the sequence is:
//! RMSNorm → Q/K/V projections → RoPE → append to KV cache → causal multi-head
//! attention (scale `1/√head_size`, softmax over positions `0..=pos`, grouped-query
//! head mapping `h / kv_mul`) → output projection + residual → RMSNorm → SwiGLU
//! feed-forward + residual. After the last layer: a final RMSNorm and the classifier
//! projection to vocabulary logits.

use crate::config::Config;
use crate::math;
use crate::quantize::{QuantizedTensor, QuantizedWeights};
use crate::state::RunState;
use crate::weights::Weights;

/// Which projection weight a [`ModelWeights::matmul`] call should apply.
///
/// The seven per-layer projections plus the `wcls` classifier — the only weights
/// that differ between the fp32 and int8 paths. Everything else in [`forward`] is
/// shared.
#[derive(Clone, Copy)]
enum Proj {
    Wq,
    Wk,
    Wv,
    Wo,
    W1,
    W2,
    W3,
    Wcls,
}

impl Proj {
    /// `(d_in, d_out)` for this projection — the dimensions [`math::matmul`] expects.
    /// `wcls` is a single `[vocab, dim]` matrix, addressed as "layer 0".
    fn shape(self, c: &Config) -> (usize, usize) {
        let att_dim = c.n_heads * c.head_size();
        let kv_dim = c.kv_dim();
        match self {
            Proj::Wq => (c.dim, att_dim),
            Proj::Wk => (c.dim, kv_dim),
            Proj::Wv => (c.dim, kv_dim),
            Proj::Wo => (att_dim, c.dim),
            Proj::W1 => (c.dim, c.hidden_dim),
            Proj::W2 => (c.hidden_dim, c.dim),
            Proj::W3 => (c.dim, c.hidden_dim),
            Proj::Wcls => (c.dim, c.vocab_size),
        }
    }
}

/// A model's weights in either representation, so a single [`forward`] serves both.
///
/// `forward` is byte-for-byte identical across the two arms except the projection
/// matmuls, which [`ModelWeights::matmul`] dispatches to the matching kernel
/// ([`math::matmul`] for `F32`, [`math::matmul_q8`] for `Q8`). The RMSNorm gains are
/// fp32 in both arms; the embedding is read through [`ModelWeights::embed`].
// Both variants are plain bundles of slice views (no heap); the enum is built once
// per run and only ever passed by reference, and the engine is no_std (so `Box` is
// not available) — the variant-size gap is harmless here.
#[allow(clippy::large_enum_variant)]
pub enum ModelWeights<'a> {
    /// Full-precision fp32 weights.
    F32(Weights<'a>),
    /// Group-wise int8-quantized weights.
    Q8(QuantizedWeights<'a>),
}

impl<'a> ModelWeights<'a> {
    /// Write token `token`'s embedding row (length `dim`) into `out`.
    ///
    /// fp32 copies the row straight from the table; int8 dequantizes just that one
    /// row (`vocab × dim` stays compressed — only `dim` values are expanded here).
    fn embed(&self, token: usize, dim: usize, out: &mut [f32]) {
        match self {
            ModelWeights::F32(w) => {
                out.copy_from_slice(&w.token_embedding[token * dim..token * dim + dim]);
            }
            ModelWeights::Q8(w) => {
                let row = w.token_embedding.layer(token, 1, dim);
                crate::quantize::dequantize(out, &row);
            }
        }
    }

    /// Per-layer attention RMSNorm gains (fp32 in both).
    fn rms_att(&self) -> &[f32] {
        match self {
            ModelWeights::F32(w) => w.rms_att,
            ModelWeights::Q8(w) => w.rms_att,
        }
    }

    /// Per-layer feed-forward RMSNorm gains (fp32 in both).
    fn rms_ffn(&self) -> &[f32] {
        match self {
            ModelWeights::F32(w) => w.rms_ffn,
            ModelWeights::Q8(w) => w.rms_ffn,
        }
    }

    /// Final RMSNorm gains (fp32 in both).
    fn rms_final(&self) -> &[f32] {
        match self {
            ModelWeights::F32(w) => w.rms_final,
            ModelWeights::Q8(w) => w.rms_final,
        }
    }

    /// Apply projection `p` of layer `l`: `out = W · x`, using whichever matmul
    /// kernel the representation calls for. For `wcls` (a single matrix) pass `l = 0`.
    fn matmul(&self, p: Proj, l: usize, out: &mut [f32], x: &[f32], c: &Config) {
        let (d_in, d_out) = p.shape(c);
        match self {
            ModelWeights::F32(w) => {
                let n = d_in * d_out;
                let wl = &f32_proj(w, p)[l * n..l * n + n];
                math::matmul(out, x, wl, d_in, d_out);
            }
            ModelWeights::Q8(w) => {
                let wl = q8_proj(w, p).layer(l, d_out, d_in);
                math::matmul_q8(out, x, &wl, d_in, d_out);
            }
        }
    }
}

/// The flat fp32 slice backing projection `p`.
fn f32_proj<'a>(w: &Weights<'a>, p: Proj) -> &'a [f32] {
    match p {
        Proj::Wq => w.wq,
        Proj::Wk => w.wk,
        Proj::Wv => w.wv,
        Proj::Wo => w.wo,
        Proj::W1 => w.w1,
        Proj::W2 => w.w2,
        Proj::W3 => w.w3,
        Proj::Wcls => w.wcls,
    }
}

/// The quantized tensor backing projection `p`.
fn q8_proj<'a>(w: &QuantizedWeights<'a>, p: Proj) -> QuantizedTensor<'a> {
    match p {
        Proj::Wq => w.wq,
        Proj::Wk => w.wk,
        Proj::Wv => w.wv,
        Proj::Wo => w.wo,
        Proj::W1 => w.w1,
        Proj::W2 => w.w2,
        Proj::W3 => w.w3,
        Proj::Wcls => w.wcls,
    }
}

/// Run one decoder step for `token` at sequence position `pos`.
///
/// Works over either fp32 or int8 weights via [`ModelWeights`]. Writes this step's
/// keys/values into `s.key_cache`/`s.value_cache` at `pos`, then attends over all
/// cached positions `0..=pos`. Returns the logits row (`s.logits`, length
/// `vocab_size`); the caller picks the next token (greedy = [`math::argmax`]).
///
/// `pos` must be `< config.seq_len` (the cache has exactly that many slots).
///
/// # Panics
/// In debug builds, via the kernels' length assertions, if `s` was not built from
/// this `config` or `pos` is out of range.
pub fn forward<'s>(
    c: &Config,
    w: &ModelWeights,
    s: &'s mut RunState,
    token: usize,
    pos: usize,
) -> &'s [f32] {
    let dim = c.dim;
    let hidden_dim = c.hidden_dim;
    let head_size = c.head_size();
    let kv_dim = c.kv_dim();
    let kv_mul = c.kv_mul();
    let n_heads = c.n_heads;
    let seq_len = c.seq_len;
    let scale = 1.0 / libm::sqrtf(head_size as f32);

    // Seed the residual stream with this token's embedding row.
    w.embed(token, dim, s.x);

    for l in 0..c.n_layers {
        // --- attention ---
        let rms_att_l = &w.rms_att()[l * dim..l * dim + dim];
        math::rmsnorm(s.xb, s.x, rms_att_l);

        // K and V for this position are written straight into the cache rows.
        let loff = l * seq_len * kv_dim; // base of layer l in the caches
        let kv_pos = loff + pos * kv_dim; // base of this position's row
        {
            let krow = &mut s.key_cache[kv_pos..kv_pos + kv_dim];
            let vrow = &mut s.value_cache[kv_pos..kv_pos + kv_dim];
            w.matmul(Proj::Wq, l, s.q, s.xb, c);
            w.matmul(Proj::Wk, l, krow, s.xb, c);
            w.matmul(Proj::Wv, l, vrow, s.xb, c);
            math::rope(s.q, krow, pos, head_size, dim, kv_dim);
        }

        // Causal multi-head attention, written into `xb` head by head.
        for h in 0..n_heads {
            let q_off = h * head_size;
            let q_head = &s.q[q_off..q_off + head_size];
            let kvh_off = (h / kv_mul) * head_size; // grouped-query: which KV head
            let att_off = h * seq_len;

            // Scaled dot-product scores against every cached key up to `pos`.
            for t in 0..=pos {
                let k_off = loff + t * kv_dim + kvh_off;
                let k_t = &s.key_cache[k_off..k_off + head_size];
                let mut score = 0.0f32;
                for i in 0..head_size {
                    score += q_head[i] * k_t[i];
                }
                s.att[att_off + t] = score * scale;
            }

            // Softmax over exactly the valid (causal) window `0..=pos`.
            math::softmax(&mut s.att[att_off..att_off + pos + 1]);

            // Weighted sum of values back into this head's slice of `xb`.
            let xb_head = &mut s.xb[q_off..q_off + head_size];
            xb_head.fill(0.0);
            for t in 0..=pos {
                let v_off = loff + t * kv_dim + kvh_off;
                let v_t = &s.value_cache[v_off..v_off + head_size];
                let a = s.att[att_off + t];
                for i in 0..head_size {
                    xb_head[i] += a * v_t[i];
                }
            }
        }

        // Output projection and the first residual add.
        w.matmul(Proj::Wo, l, s.xb2, s.xb, c);
        math::accumulate(s.x, s.xb2);

        // --- feed-forward (SwiGLU) ---
        let rms_ffn_l = &w.rms_ffn()[l * dim..l * dim + dim];
        math::rmsnorm(s.xb, s.x, rms_ffn_l);

        w.matmul(Proj::W1, l, s.hb, s.xb, c);
        w.matmul(Proj::W3, l, s.hb2, s.xb, c);
        for i in 0..hidden_dim {
            s.hb[i] = math::silu(s.hb[i]) * s.hb2[i];
        }
        w.matmul(Proj::W2, l, s.xb, s.hb, c);
        math::accumulate(s.x, s.xb);
    }

    // Final norm (into `xb` to avoid aliasing `x`), then classifier to logits.
    math::rmsnorm(s.xb, s.x, w.rms_final());
    w.matmul(Proj::Wcls, 0, s.logits, s.xb, c);
    &s.logits[..]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::Arena;
    use crate::weights::weight_floats;

    extern crate std;
    use std::vec;
    use std::vec::Vec;

    fn tiny_config() -> Config {
        // Grouped-query attention (n_kv_heads < n_heads) to exercise the kv_mul path.
        Config {
            dim: 8,
            hidden_dim: 16,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            vocab_size: 16,
            seq_len: 8,
            shared_weights: true,
        }
    }

    /// Deterministic pseudo-random weights in a small range, so the forward pass
    /// produces finite, non-trivial logits without overflow.
    fn fake_weights(c: &Config) -> Vec<f32> {
        let n = weight_floats(c);
        let mut v = Vec::with_capacity(n);
        let mut state: u32 = 0x1234_5678;
        for _ in 0..n {
            // xorshift -> value in roughly [-0.1, 0.1]
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let f = (state as f32 / u32::MAX as f32) - 0.5;
            v.push(f * 0.2);
        }
        v
    }

    #[test]
    fn forward_produces_finite_logits_of_right_length() {
        let c = tiny_config();
        let wbuf = fake_weights(&c);
        let w = ModelWeights::F32(Weights::new(&wbuf, &c).unwrap());

        let mut arena_buf = vec![0.0f32; crate::memory::arena_floats(&c)];
        let mut arena = Arena::new(&mut arena_buf);
        let mut s = RunState::new(&mut arena, &c).unwrap();

        let logits = forward(&c, &w, &mut s, 3, 0);
        assert_eq!(logits.len(), c.vocab_size);
        assert!(logits.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn forward_is_deterministic_across_positions() {
        let c = tiny_config();
        let wbuf = fake_weights(&c);
        let w = ModelWeights::F32(Weights::new(&wbuf, &c).unwrap());

        // Feed the same token sequence through two independent run states; the
        // logits at each step must match exactly (no hidden global state, cache
        // used consistently).
        let tokens = [1usize, 5, 2, 7];

        let run = |toks: &[usize]| -> Vec<Vec<f32>> {
            let mut arena_buf = vec![0.0f32; crate::memory::arena_floats(&c)];
            let mut arena = Arena::new(&mut arena_buf);
            let mut s = RunState::new(&mut arena, &c).unwrap();
            let mut out = Vec::new();
            for (pos, &tok) in toks.iter().enumerate() {
                out.push(forward(&c, &w, &mut s, tok, pos).to_vec());
            }
            out
        };

        assert_eq!(run(&tokens), run(&tokens));
    }

    #[test]
    fn attention_reads_history_from_the_cache() {
        // The logits for the *same* token at pos 1 must depend on which token
        // preceded it — that only happens if attention actually reads the cached
        // key/value of position 0. (Note: feeding the identical token at both
        // positions would NOT prove this, because values aren't position-rotated,
        // so attending over two identical value vectors returns that same vector.)
        let c = tiny_config();
        let wbuf = fake_weights(&c);
        let w = ModelWeights::F32(Weights::new(&wbuf, &c).unwrap());

        let run_pair = |first: usize| -> Vec<f32> {
            let mut arena_buf = vec![0.0f32; crate::memory::arena_floats(&c)];
            let mut arena = Arena::new(&mut arena_buf);
            let mut s = RunState::new(&mut arena, &c).unwrap();
            forward(&c, &w, &mut s, first, 0);
            forward(&c, &w, &mut s, 7, 1).to_vec()
        };

        assert_ne!(run_pair(4), run_pair(5));
    }

    #[test]
    fn forward_q8_tracks_fp32() {
        // Quantizing the eight matmul weights must not change the answer's scale: the
        // int8 logits track the fp32 logits closely for the same input.
        use crate::quantize::{quantize_weights, quantized_scale_count, quantized_weight_count};

        let c = tiny_config();
        let wbuf = fake_weights(&c);
        let fp32 = Weights::new(&wbuf, &c).unwrap();

        // group_size 4 divides every projection's d_in (8 and 16) in tiny_config.
        let group_size = 4;
        let mut data = vec![0i8; quantized_weight_count(&c)];
        let mut scales = vec![0.0f32; quantized_scale_count(&c, group_size)];
        quantize_weights(&fp32, &mut data, &mut scales, group_size, &c);
        let qw = QuantizedWeights::new(
            &data,
            &scales,
            fp32.rms_att,
            fp32.rms_ffn,
            fp32.rms_final,
            group_size,
            &c,
        )
        .unwrap();

        let f32_w = ModelWeights::F32(fp32);
        let q8_w = ModelWeights::Q8(qw);

        let logits = |w: &ModelWeights| -> Vec<f32> {
            let mut arena_buf = vec![0.0f32; crate::memory::arena_floats(&c)];
            let mut arena = Arena::new(&mut arena_buf);
            let mut s = RunState::new(&mut arena, &c).unwrap();
            forward(&c, w, &mut s, 3, 0).to_vec()
        };

        let a = logits(&f32_w);
        let b = logits(&q8_w);
        assert_eq!(b.len(), c.vocab_size);
        assert!(b.iter().all(|x| x.is_finite()));
        let max_diff = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        let max_abs = a.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        // The int8 logits stay within 2% of the largest fp32 logit (observed ~0.3%).
        assert!(
            max_diff < 0.02 * max_abs,
            "q8 logits drifted from fp32: max_diff={max_diff}, max_abs={max_abs}"
        );
    }
}
