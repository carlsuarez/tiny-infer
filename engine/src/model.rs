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
use crate::state::RunState;
use crate::weights::Weights;

/// Run one decoder step for `token` at sequence position `pos`.
///
/// Writes this step's keys/values into `s.key_cache`/`s.value_cache` at `pos`, then
/// attends over all cached positions `0..=pos`. Returns the logits row (`s.logits`,
/// length `vocab_size`); the caller picks the next token (greedy = [`math::argmax`]).
///
/// `pos` must be `< config.seq_len` (the cache has exactly that many slots).
///
/// # Panics
/// In debug builds, via the kernels' length assertions, if `s` was not built from
/// this `config` or `pos` is out of range.
pub fn forward<'s>(c: &Config, w: &Weights, s: &'s mut RunState, token: usize, pos: usize) -> &'s [f32] {
    let dim = c.dim;
    let hidden_dim = c.hidden_dim;
    let head_size = c.head_size();
    let att_dim = c.n_heads * head_size; // == dim, but named for clarity
    let kv_dim = c.kv_dim();
    let kv_mul = c.kv_mul();
    let n_heads = c.n_heads;
    let seq_len = c.seq_len;
    let scale = 1.0 / libm::sqrtf(head_size as f32);

    // Seed the residual stream with this token's embedding row.
    let emb = &w.token_embedding[token * dim..token * dim + dim];
    s.x.copy_from_slice(emb);

    for l in 0..c.n_layers {
        // --- attention ---
        let rms_att_l = &w.rms_att[l * dim..l * dim + dim];
        math::rmsnorm(s.xb, s.x, rms_att_l);

        // K and V for this position are written straight into the cache rows.
        let loff = l * seq_len * kv_dim; // base of layer l in the caches
        let kv_pos = loff + pos * kv_dim; // base of this position's row
        {
            let krow = &mut s.key_cache[kv_pos..kv_pos + kv_dim];
            let vrow = &mut s.value_cache[kv_pos..kv_pos + kv_dim];
            let wq_l = &w.wq[l * dim * att_dim..l * dim * att_dim + dim * att_dim];
            let wk_l = &w.wk[l * dim * kv_dim..l * dim * kv_dim + dim * kv_dim];
            let wv_l = &w.wv[l * dim * kv_dim..l * dim * kv_dim + dim * kv_dim];
            math::matmul(s.q, s.xb, wq_l, dim, att_dim);
            math::matmul(krow, s.xb, wk_l, dim, kv_dim);
            math::matmul(vrow, s.xb, wv_l, dim, kv_dim);
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
        let wo_l = &w.wo[l * att_dim * dim..l * att_dim * dim + att_dim * dim];
        math::matmul(s.xb2, s.xb, wo_l, att_dim, dim);
        math::accumulate(s.x, s.xb2);

        // --- feed-forward (SwiGLU) ---
        let rms_ffn_l = &w.rms_ffn[l * dim..l * dim + dim];
        math::rmsnorm(s.xb, s.x, rms_ffn_l);

        let w1_l = &w.w1[l * hidden_dim * dim..l * hidden_dim * dim + hidden_dim * dim];
        let w3_l = &w.w3[l * hidden_dim * dim..l * hidden_dim * dim + hidden_dim * dim];
        math::matmul(s.hb, s.xb, w1_l, dim, hidden_dim);
        math::matmul(s.hb2, s.xb, w3_l, dim, hidden_dim);
        for i in 0..hidden_dim {
            s.hb[i] = math::silu(s.hb[i]) * s.hb2[i];
        }
        let w2_l = &w.w2[l * dim * hidden_dim..l * dim * hidden_dim + dim * hidden_dim];
        math::matmul(s.xb, s.hb, w2_l, hidden_dim, dim);
        math::accumulate(s.x, s.xb);
    }

    // Final norm (into `xb` to avoid aliasing `x`), then classifier to logits.
    math::rmsnorm(s.xb, s.x, w.rms_final);
    math::matmul(s.logits, s.xb, w.wcls, dim, c.vocab_size);
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
        let w = Weights::new(&wbuf, &c).unwrap();

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
        let w = Weights::new(&wbuf, &c).unwrap();

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
        let w = Weights::new(&wbuf, &c).unwrap();

        let run_pair = |first: usize| -> Vec<f32> {
            let mut arena_buf = vec![0.0f32; crate::memory::arena_floats(&c)];
            let mut arena = Arena::new(&mut arena_buf);
            let mut s = RunState::new(&mut arena, &c).unwrap();
            forward(&c, &w, &mut s, first, 0);
            forward(&c, &w, &mut s, 7, 1).to_vec()
        };

        assert_ne!(run_pair(4), run_pair(5));
    }
}
