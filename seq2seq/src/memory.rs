//! Memory-budget arithmetic for the seq2seq working arena.
//!
//! Computes, from a [`Config`] and the source/target lengths alone, how
//! large the working arena must be — before any buffer is touched. This is the
//! seq2seq counterpart of `llama::memory`, and the contract is the same: the
//! seq2seq run state carves exactly this budget from the caller's arena, once,
//! and the forward passes then allocate nothing.
//!
//! A seq2seq pass needs more resident state than decoder-only generation, because
//! encoder self-attention is **all-to-all** — every source position attends to
//! every other — so the whole source-side residual stream stays live:
//!
//! * **Encoder buffers** ([`seq2seq_encoder_floats`]): the residual stream
//!   `enc_x` (`src_len × d_model`, which becomes the encoder output and persists
//!   through decoding), a same-sized sublayer-output buffer (post-norm layers
//!   may not update the stream in place while attention still reads it), one
//!   layer's K and V projections, and a single `src_len` score row — query
//!   positions loop over that one row, keeping scratch O(src_len) rather than
//!   O(src_len²).
//! * **Cross-KV cache** ([`seq2seq_cross_kv_floats`]): every decoder layer's K
//!   and V projections of the encoder output, computed once before decoding and
//!   reused at every step.
//! * **Self-KV cache** ([`seq2seq_self_kv_floats`]): the decoder's causal
//!   key/value cache, growing one target position per step — exactly the Llama
//!   path's `key_cache`/`value_cache`.
//! * **Per-step scratch** ([`seq2seq_step_floats`]): the single-position working
//!   set (residual, sublayer scratch, query, FFN buffer, attention scores,
//!   logits), shared between the encoder's per-position loops and the decoder
//!   step.

use crate::config::Config;

const F32: usize = core::mem::size_of::<f32>();

/// `f32` count of the encoder-side buffers for a `src_len`-token source.
///
/// `enc_x` + sublayer outputs (each `src_len × d_model`), one layer's K and V
/// (each `src_len × d_model`; overwritten layer by layer), and one `src_len`
/// score row.
pub const fn seq2seq_encoder_floats(c: &Config, src_len: usize) -> usize {
    4 * src_len * c.d_model + src_len
}

/// `f32` count of the cross-attention KV cache: K and V over the encoder
/// output, per decoder layer — projected once, read every decode step.
pub const fn seq2seq_cross_kv_floats(c: &Config, src_len: usize) -> usize {
    2 * c.dec_layers * src_len * c.d_model
}

/// `f32` count of the decoder's causal self-attention KV cache, sized for
/// `tgt_len` target positions.
pub const fn seq2seq_self_kv_floats(c: &Config, tgt_len: usize) -> usize {
    2 * c.dec_layers * tgt_len * c.d_model
}

/// `f32` count of the single-position working set: residual `x`, two sublayer
/// scratch vectors, the query (all `d_model`), the FFN buffer (the wider of the
/// two stacks), per-head attention scores over the longer of the two sequences,
/// and the output logits.
pub const fn seq2seq_step_floats(c: &Config, src_len: usize, tgt_len: usize) -> usize {
    // (`Ord::max` is not `const`, so pick the larger by hand.)
    let ffn = if c.enc_ffn > c.dec_ffn {
        c.enc_ffn
    } else {
        c.dec_ffn
    };
    let span = if src_len > tgt_len { src_len } else { tgt_len };
    4 * c.d_model + ffn + c.dec_heads * span + c.vocab_size
}

/// Total `f32` the seq2seq arena must hold for one run of up to
/// `src_len` source and `tgt_len` target tokens.
///
/// `const`, so an embedded host can size a `static`/stack arena — and
/// `const`-assert it fits a fixed RAM budget — at compile time, exactly like
/// `llama::memory::arena_floats` on the Llama path.
pub const fn seq2seq_arena_floats(c: &Config, src_len: usize, tgt_len: usize) -> usize {
    seq2seq_encoder_floats(c, src_len)
        + seq2seq_cross_kv_floats(c, src_len)
        + seq2seq_self_kv_floats(c, tgt_len)
        + seq2seq_step_floats(c, src_len, tgt_len)
}

/// A broken-down seq2seq memory budget for reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryBudget {
    /// `f32` count for the encoder-side buffers.
    pub encoder_floats: usize,
    /// `f32` count for the cross-attention KV cache.
    pub cross_kv_floats: usize,
    /// `f32` count for the decoder self-attention KV cache.
    pub self_kv_floats: usize,
    /// `f32` count for the per-step working set.
    pub step_floats: usize,
}

impl MemoryBudget {
    /// Compute the budget for a config at the given sequence lengths.
    pub const fn for_config(c: &Config, src_len: usize, tgt_len: usize) -> Self {
        MemoryBudget {
            encoder_floats: seq2seq_encoder_floats(c, src_len),
            cross_kv_floats: seq2seq_cross_kv_floats(c, src_len),
            self_kv_floats: seq2seq_self_kv_floats(c, tgt_len),
            step_floats: seq2seq_step_floats(c, src_len, tgt_len),
        }
    }

    /// Total `f32` across all four groups.
    #[inline]
    pub const fn total_floats(&self) -> usize {
        self.encoder_floats + self.cross_kv_floats + self.self_kv_floats + self.step_floats
    }

    /// Encoder-buffer bytes.
    #[inline]
    pub const fn encoder_bytes(&self) -> usize {
        self.encoder_floats * F32
    }

    /// Cross-KV cache bytes.
    #[inline]
    pub const fn cross_kv_bytes(&self) -> usize {
        self.cross_kv_floats * F32
    }

    /// Decoder self-KV cache bytes.
    #[inline]
    pub const fn self_kv_bytes(&self) -> usize {
        self.self_kv_floats * F32
    }

    /// Per-step working-set bytes.
    #[inline]
    pub const fn step_bytes(&self) -> usize {
        self.step_floats * F32
    }

    /// Total arena bytes.
    #[inline]
    pub const fn total_bytes(&self) -> usize {
        self.total_floats() * F32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Activation;

    fn tiny_config() -> Config {
        Config {
            d_model: 4,
            enc_layers: 1,
            dec_layers: 1,
            enc_heads: 2,
            dec_heads: 2,
            enc_ffn: 8,
            dec_ffn: 8,
            vocab_size: 10,
            max_src: 6,
            max_tgt: 5,
            pad_id: 9,
            eos_id: 0,
            bos_id: 0,
            norm_before: false,
            activation: Activation::Swish,
            scale_embedding: true,
        }
    }

    #[test]
    fn budget_matches_hand_count() {
        let c = tiny_config();
        let (src, tgt) = (6, 5);
        // encoder: 4*6*4 + 6 = 102; cross KV: 2*1*6*4 = 48; self KV: 2*1*5*4 = 40;
        // step: 4*4 + 8 + 2*6 + 10 = 46.
        assert_eq!(seq2seq_encoder_floats(&c, src), 102);
        assert_eq!(seq2seq_cross_kv_floats(&c, src), 48);
        assert_eq!(seq2seq_self_kv_floats(&c, tgt), 40);
        assert_eq!(seq2seq_step_floats(&c, src, tgt), 46);
        assert_eq!(seq2seq_arena_floats(&c, src, tgt), 102 + 48 + 40 + 46);

        let b = MemoryBudget::for_config(&c, src, tgt);
        assert_eq!(b.total_floats(), seq2seq_arena_floats(&c, src, tgt));
        assert_eq!(b.total_bytes(), b.total_floats() * 4);
    }

    #[test]
    fn opus_mt_en_fr_budget_is_tens_of_mib() {
        // The realistic figure the host will report for a ~75M-param OPUS-MT
        // pair model at its maximum sequence lengths.
        let c = Config {
            d_model: 512,
            enc_layers: 6,
            dec_layers: 6,
            enc_heads: 8,
            dec_heads: 8,
            enc_ffn: 2048,
            dec_ffn: 2048,
            vocab_size: 59514,
            max_src: 512,
            max_tgt: 512,
            pad_id: 59513,
            eos_id: 0,
            bos_id: 0,
            norm_before: false,
            activation: Activation::Swish,
            scale_embedding: true,
        };
        let b = MemoryBudget::for_config(&c, c.max_src, c.max_tgt);
        // The two KV caches dominate, like the Llama path's KV cache.
        assert_eq!(b.cross_kv_floats, 2 * 6 * 512 * 512);
        assert_eq!(b.self_kv_floats, 2 * 6 * 512 * 512);
        let mib = b.total_bytes() as f64 / (1024.0 * 1024.0);
        assert!((28.0..29.0).contains(&mib), "got {mib} MiB");
    }

    #[test]
    fn budget_is_const_evaluable() {
        // Forcing the evaluation in a const context guards the `const fn`-ness
        // against regressing (the bare-metal sizing story depends on it).
        const C: Config = Config {
            d_model: 4,
            enc_layers: 1,
            dec_layers: 1,
            enc_heads: 2,
            dec_heads: 2,
            enc_ffn: 8,
            dec_ffn: 8,
            vocab_size: 10,
            max_src: 6,
            max_tgt: 5,
            pad_id: 9,
            eos_id: 0,
            bos_id: 0,
            norm_before: false,
            activation: Activation::Swish,
            scale_embedding: true,
        };
        const FLOATS: usize = seq2seq_arena_floats(&C, 6, 5);
        const BUDGET: MemoryBudget = MemoryBudget::for_config(&C, 6, 5);
        const _: () = assert!(BUDGET.total_bytes() == FLOATS * F32);
        assert_eq!(FLOATS, 236);
        assert_eq!(BUDGET.total_floats(), FLOATS);
    }
}
