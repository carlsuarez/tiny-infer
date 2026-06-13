//! Working buffers for a seq2seq translation, carved from the [`Arena`] once.
//!
//! [`RunState`] is the encoder-decoder counterpart of the Llama path's
//! [`RunState`](crate::llama::state::RunState): every buffer the encoder and decoder
//! forward passes read or write is allocated up front from a single
//! caller-provided [`Arena`], so the passes themselves allocate nothing. The
//! total is exactly
//! [`seq2seq_arena_floats`](crate::seq2seq::memory::seq2seq_arena_floats), and the
//! four field groups below mirror that budget's four groups:
//!
//! * **encoder buffers** — the encoder's self-attention is all-to-all, so every
//!   source position stays resident: the residual stream (which becomes the
//!   encoder output), a sublayer-output buffer, one layer's keys and values, and a
//!   single score row reused across query positions.
//! * **cross-KV cache** — keys/values of the encoder output per decoder layer,
//!   projected once and read at every decode step (populated in a later milestone).
//! * **self-KV cache** — the decoder's causal key/value cache, one row per target
//!   position (populated in a later milestone).
//! * **step scratch** — the single-position working set shared by the encoder's
//!   per-position loops and the decoder step.
//!
//! [`Arena`]: crate::Arena

use crate::arena::Arena;
use crate::error::EngineError;
use crate::seq2seq::config::Config;

/// Every mutable buffer the encoder and decoder forward passes use.
///
/// Each field borrows a disjoint sub-slice of one arena block (`'buf`); as with
/// [`RunState`](crate::llama::state::RunState), [`Arena::alloc`] ties the slices to the
/// block's lifetime, not the arena borrow, so they coexist. Buffer sizes come
/// straight from [`Config`] and the source/target lengths.
#[derive(Debug)]
pub struct RunState<'buf> {
    // --- encoder buffers (all-to-all attention ⇒ all source positions resident) ---
    /// Encoder residual stream; after the last layer this *is* the encoder
    /// output (the decoder's cross-attention reads it), `[src_len * d]`.
    pub enc_x: &'buf mut [f32],
    /// Sublayer-output scratch — holds the attention outputs, then the FFN
    /// outputs, before each residual add, `[src_len * d]`.
    pub enc_sub: &'buf mut [f32],
    /// One encoder layer's keys (overwritten layer by layer), `[src_len * d]`.
    pub enc_k: &'buf mut [f32],
    /// One encoder layer's values (overwritten layer by layer), `[src_len * d]`.
    pub enc_v: &'buf mut [f32],
    /// Attention scores for one query position, reused across heads, `[src_len]`.
    pub enc_scores: &'buf mut [f32],

    // --- cross-attention KV cache (encoder output projected per decoder layer) ---
    /// Cross-attention keys, `[dec_layers * src_len * d]`.
    pub cross_k: &'buf mut [f32],
    /// Cross-attention values, `[dec_layers * src_len * d]`.
    pub cross_v: &'buf mut [f32],

    // --- decoder self-attention KV cache (grows one row per target position) ---
    /// Decoder self-attention keys, `[dec_layers * tgt_len * d]`.
    pub self_k: &'buf mut [f32],
    /// Decoder self-attention values, `[dec_layers * tgt_len * d]`.
    pub self_v: &'buf mut [f32],

    // --- per-step scratch (shared by the encoder loops and the decoder step) ---
    /// Decoder per-step residual stream, `[d]`.
    pub x: &'buf mut [f32],
    /// Query vector for the current position, `[d]`.
    pub q: &'buf mut [f32],
    /// Attention output (concatenated heads) before the output projection, `[d]`.
    pub attn: &'buf mut [f32],
    /// Norm / position scratch — a buffer LayerNorm or `sinusoidal_into` can write
    /// without aliasing its input, `[d]`.
    pub norm: &'buf mut [f32],
    /// Feed-forward hidden buffer, `[max(enc_ffn, dec_ffn)]`.
    pub ffn: &'buf mut [f32],
    /// Decoder attention scores, `[dec_heads * max(src_len, tgt_len)]`.
    pub scores: &'buf mut [f32],
    /// Output logits over the vocabulary, `[vocab]`.
    pub logits: &'buf mut [f32],
}

impl<'buf> RunState<'buf> {
    /// Carve every buffer out of `arena` for a translation of up to `src_len`
    /// source and `tgt_len` target tokens.
    ///
    /// Allocates in a fixed order summing to exactly
    /// [`seq2seq_arena_floats`](crate::seq2seq::memory::seq2seq_arena_floats);
    /// size the arena with that function. Returns [`EngineError::ArenaOverflow`]
    /// if it is too small. Pass `tgt_len = 0` to size an encode-only state (the
    /// decoder caches become empty slices).
    pub fn new(
        arena: &mut Arena<'buf>,
        c: &Config,
        src_len: usize,
        tgt_len: usize,
    ) -> Result<RunState<'buf>, EngineError> {
        let d = c.d_model;
        // `Ord::max` is not const-friendly elsewhere in the budget, so mirror its
        // hand-rolled form here for the two values that drive the step scratch.
        let ffn = if c.enc_ffn > c.dec_ffn {
            c.enc_ffn
        } else {
            c.dec_ffn
        };
        let span = if src_len > tgt_len { src_len } else { tgt_len };

        Ok(RunState {
            enc_x: arena.alloc(src_len * d)?,
            enc_sub: arena.alloc(src_len * d)?,
            enc_k: arena.alloc(src_len * d)?,
            enc_v: arena.alloc(src_len * d)?,
            enc_scores: arena.alloc(src_len)?,

            cross_k: arena.alloc(c.dec_layers * src_len * d)?,
            cross_v: arena.alloc(c.dec_layers * src_len * d)?,

            self_k: arena.alloc(c.dec_layers * tgt_len * d)?,
            self_v: arena.alloc(c.dec_layers * tgt_len * d)?,

            x: arena.alloc(d)?,
            q: arena.alloc(d)?,
            attn: arena.alloc(d)?,
            norm: arena.alloc(d)?,
            ffn: arena.alloc(ffn)?,
            scores: arena.alloc(c.dec_heads * span)?,
            logits: arena.alloc(c.vocab_size)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seq2seq::config::Activation;
    use crate::seq2seq::memory::seq2seq_arena_floats;

    extern crate std;
    use std::vec;

    fn tiny_config() -> Config {
        Config {
            d_model: 4,
            enc_layers: 1,
            dec_layers: 2,
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
    fn carves_all_buffers_with_correct_sizes() {
        let c = tiny_config();
        let (src, tgt) = (6, 5);
        let mut buf = vec![0.0f32; seq2seq_arena_floats(&c, src, tgt)];
        let mut arena = Arena::new(&mut buf);
        let s = RunState::new(&mut arena, &c, src, tgt).unwrap();

        let d = c.d_model;
        assert_eq!(s.enc_x.len(), src * d);
        assert_eq!(s.enc_sub.len(), src * d);
        assert_eq!(s.enc_k.len(), src * d);
        assert_eq!(s.enc_v.len(), src * d);
        assert_eq!(s.enc_scores.len(), src);
        assert_eq!(s.cross_k.len(), c.dec_layers * src * d);
        assert_eq!(s.cross_v.len(), c.dec_layers * src * d);
        assert_eq!(s.self_k.len(), c.dec_layers * tgt * d);
        assert_eq!(s.self_v.len(), c.dec_layers * tgt * d);
        assert_eq!(s.x.len(), d);
        assert_eq!(s.q.len(), d);
        assert_eq!(s.attn.len(), d);
        assert_eq!(s.norm.len(), d);
        assert_eq!(s.ffn.len(), c.enc_ffn.max(c.dec_ffn));
        assert_eq!(s.scores.len(), c.dec_heads * src.max(tgt));
        assert_eq!(s.logits.len(), c.vocab_size);
    }

    #[test]
    fn consumes_exactly_the_budget() {
        let c = tiny_config();
        let (src, tgt) = (6, 5);
        let total = seq2seq_arena_floats(&c, src, tgt);
        let mut buf = vec![0.0f32; total];
        let mut arena = Arena::new(&mut buf);
        let _s = RunState::new(&mut arena, &c, src, tgt).unwrap();
        // The budget is exact: nothing left over, nothing short.
        assert_eq!(arena.used(), total);
        assert_eq!(arena.remaining(), 0);
    }

    #[test]
    fn encode_only_state_zeroes_the_decoder_caches() {
        // tgt_len = 0 sizes an encode-only state; the decoder self-KV caches are
        // empty and the arena is just the encoder + cross-KV + step groups.
        let c = tiny_config();
        let src = 6;
        let mut buf = vec![0.0f32; seq2seq_arena_floats(&c, src, 0)];
        let mut arena = Arena::new(&mut buf);
        let s = RunState::new(&mut arena, &c, src, 0).unwrap();
        assert!(s.self_k.is_empty());
        assert!(s.self_v.is_empty());
        assert_eq!(s.cross_k.len(), c.dec_layers * src * c.d_model);
    }

    #[test]
    fn too_small_arena_overflows() {
        let c = tiny_config();
        let (src, tgt) = (6, 5);
        let mut buf = vec![0.0f32; seq2seq_arena_floats(&c, src, tgt) - 1];
        let mut arena = Arena::new(&mut buf);
        assert!(matches!(
            RunState::new(&mut arena, &c, src, tgt),
            Err(EngineError::ArenaOverflow { .. })
        ));
    }
}
