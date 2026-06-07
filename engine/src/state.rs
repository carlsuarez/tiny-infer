//! Per-step run state: every working buffer, carved from the [`Arena`] once.
//!
//! [`RunState`] mirrors llama2.c's `RunState`: a handful of activation scratch
//! buffers plus the key/value cache. All of it is allocated from a single
//! caller-provided [`Arena`] during [`RunState::new`] and then reused in place for
//! every token, so the steady-state forward pass allocates nothing.
//!
//! There is deliberately **no separate key/value scratch**: each step writes its
//! key and value vectors straight into the cache row for the current
//! `(layer, position)`, exactly as the reference does.
//!
//! [`Arena`]: crate::Arena

use crate::arena::Arena;
use crate::config::Config;
use crate::error::EngineError;

/// All mutable buffers a single forward pass reads and writes.
///
/// Every field borrows a disjoint sub-slice of the arena block (`'buf`); because
/// [`Arena::alloc`] hands back slices tied to the block's lifetime rather than to
/// the arena borrow, they can all be held at once. Buffer sizes come straight from
/// the [`Config`] and match [`crate::memory`]'s budget.
#[derive(Debug)]
pub struct RunState<'buf> {
    /// Residual stream, `[dim]`.
    pub x: &'buf mut [f32],
    /// Normed / attention-output scratch, `[dim]`.
    pub xb: &'buf mut [f32],
    /// Second residual scratch (attention output projection), `[dim]`.
    pub xb2: &'buf mut [f32],
    /// SwiGLU gate branch, `[hidden_dim]`.
    pub hb: &'buf mut [f32],
    /// SwiGLU up branch, `[hidden_dim]`.
    pub hb2: &'buf mut [f32],
    /// Query vector for the current position, `[n_heads * head_size]`.
    pub q: &'buf mut [f32],
    /// Attention scores, `[n_heads * seq_len]` (one row of `seq_len` per head).
    pub att: &'buf mut [f32],
    /// Output logits over the vocabulary, `[vocab_size]`.
    pub logits: &'buf mut [f32],
    /// Key cache, `[n_layers * seq_len * kv_dim]`.
    pub key_cache: &'buf mut [f32],
    /// Value cache, `[n_layers * seq_len * kv_dim]`.
    pub value_cache: &'buf mut [f32],
}

impl<'buf> RunState<'buf> {
    /// Carve every buffer out of `arena` for the given config.
    ///
    /// Allocates in a fixed order; the total is exactly
    /// [`crate::memory::arena_floats`]. Returns [`EngineError::ArenaOverflow`] if
    /// the arena was sized too small (size it with `arena_floats` to guarantee a
    /// fit).
    pub fn new(arena: &mut Arena<'buf>, c: &Config) -> Result<RunState<'buf>, EngineError> {
        let att_dim = c.n_heads * c.head_size();
        let kv_cache = c.n_layers * c.seq_len * c.kv_dim();

        Ok(RunState {
            x: arena.alloc(c.dim)?,
            xb: arena.alloc(c.dim)?,
            xb2: arena.alloc(c.dim)?,
            hb: arena.alloc(c.hidden_dim)?,
            hb2: arena.alloc(c.hidden_dim)?,
            q: arena.alloc(att_dim)?,
            att: arena.alloc(c.n_heads * c.seq_len)?,
            logits: arena.alloc(c.vocab_size)?,
            key_cache: arena.alloc(kv_cache)?,
            value_cache: arena.alloc(kv_cache)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::arena_floats;

    extern crate std;
    use std::vec;

    fn tiny_config() -> Config {
        Config {
            dim: 8,
            hidden_dim: 16,
            n_layers: 2,
            n_heads: 2,
            n_kv_heads: 1,
            vocab_size: 32,
            seq_len: 4,
            shared_weights: true,
        }
    }

    #[test]
    fn carves_all_buffers_with_correct_sizes() {
        let c = tiny_config();
        let mut buf = vec![0.0f32; arena_floats(&c)];
        let mut arena = Arena::new(&mut buf);
        let s = RunState::new(&mut arena, &c).unwrap();

        let att_dim = c.n_heads * c.head_size();
        let kv_cache = c.n_layers * c.seq_len * c.kv_dim();
        assert_eq!(s.x.len(), c.dim);
        assert_eq!(s.xb.len(), c.dim);
        assert_eq!(s.xb2.len(), c.dim);
        assert_eq!(s.hb.len(), c.hidden_dim);
        assert_eq!(s.hb2.len(), c.hidden_dim);
        assert_eq!(s.q.len(), att_dim);
        assert_eq!(s.att.len(), c.n_heads * c.seq_len);
        assert_eq!(s.logits.len(), c.vocab_size);
        assert_eq!(s.key_cache.len(), kv_cache);
        assert_eq!(s.value_cache.len(), kv_cache);
    }

    #[test]
    fn consumes_exactly_the_budget() {
        let c = tiny_config();
        let total = arena_floats(&c);
        let mut buf = vec![0.0f32; total];
        let mut arena = Arena::new(&mut buf);
        let _s = RunState::new(&mut arena, &c).unwrap();
        // The budget is exact: nothing left over, nothing short.
        assert_eq!(arena.used(), total);
        assert_eq!(arena.remaining(), 0);
    }

    #[test]
    fn too_small_arena_overflows() {
        let c = tiny_config();
        let mut buf = vec![0.0f32; arena_floats(&c) - 1];
        let mut arena = Arena::new(&mut buf);
        assert!(matches!(
            RunState::new(&mut arena, &c),
            Err(EngineError::ArenaOverflow { .. })
        ));
    }
}
