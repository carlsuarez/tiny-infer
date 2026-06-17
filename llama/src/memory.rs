//! Memory-budget arithmetic.
//!
//! Computes, from a [`Config`] alone, exactly how large the working arena must be
//! — before any buffer is touched. This is what lets the host pre-allocate one
//! block and lets the CLI report the budget up front.
//!
//! The layout mirrors llama2.c's `RunState`: a set of activation scratch buffers
//! plus the KV cache. The KV cache dominates and grows with `n_layers * seq_len`.

use crate::config::Config;

const F32: usize = core::mem::size_of::<f32>();

/// Number of `f32` in the per-step activation scratch buffers.
///
/// | buffer  | size        | role                                   |
/// |---------|-------------|----------------------------------------|
/// | `x`     | `dim`       | residual stream                        |
/// | `xb`    | `dim`       | normed / attention output scratch      |
/// | `xb2`   | `dim`       | second residual scratch                |
/// | `hb`    | `hidden`    | SwiGLU branch 1                        |
/// | `hb2`   | `hidden`    | SwiGLU branch 2                        |
/// | `q`     | `att_dim`   | query for the current position         |
/// | `att`   | `heads*seq` | attention scores per head              |
/// | `logits`| `vocab`     | output distribution                    |
///
/// The int8 (W8A8) path's activation scratch is *not* part of this arena — it lives in a
/// separate, caller-owned [`QuantScratch`](engine::QuantScratch), so an fp32 run budgets
/// none of it.
pub const fn activation_floats(c: &Config) -> usize {
    let att_dim = c.n_heads * c.head_size();
    3 * c.dim                 // x, xb, xb2
        + 2 * c.hidden_dim    // hb, hb2
        + att_dim             // q
        + c.n_heads * c.seq_len // att
        + c.vocab_size // logits
}

/// The largest matmul input dimension `d_in` across all projections — the size each
/// buffer of the W8A8 activation scratch ([`QuantScratch`](engine::QuantScratch)) must
/// cover.
pub const fn max_proj_d_in(c: &Config) -> usize {
    let att_dim = c.n_heads * c.head_size();
    // Wq/Wk/Wv/W1/W3/Wcls take `dim`; Wo takes `att_dim`; W2 takes `hidden_dim`.
    // (`Ord::max` is not `const`, so pick the larger of each pair by hand.)
    let a = if c.dim > att_dim { c.dim } else { att_dim };
    if a > c.hidden_dim {
        a
    } else {
        c.hidden_dim
    }
}

/// Number of `f32` in the KV cache: key and value caches, each
/// `n_layers * seq_len * kv_dim`.
pub const fn kv_cache_floats(c: &Config) -> usize {
    2 * c.n_layers * c.seq_len * c.kv_dim()
}

/// Total `f32` the arena must hold: activations plus KV cache.
///
/// `const`, so an embedded host can size a `static`/stack arena at compile time and
/// even `const`-assert it fits a fixed memory budget — see `examples/baremetal.rs`.
pub const fn arena_floats(c: &Config) -> usize {
    activation_floats(c) + kv_cache_floats(c)
}

/// A broken-down memory budget for reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryBudget {
    /// `f32` count for activation scratch.
    pub activation_floats: usize,
    /// `f32` count for the KV cache.
    pub kv_cache_floats: usize,
}

impl MemoryBudget {
    /// Compute the budget for a config.
    pub const fn for_config(c: &Config) -> Self {
        MemoryBudget {
            activation_floats: activation_floats(c),
            kv_cache_floats: kv_cache_floats(c),
        }
    }

    /// Total `f32` (activations + KV cache).
    #[inline]
    pub const fn total_floats(&self) -> usize {
        self.activation_floats + self.kv_cache_floats
    }

    /// Activation bytes.
    #[inline]
    pub const fn activation_bytes(&self) -> usize {
        self.activation_floats * F32
    }

    /// KV-cache bytes.
    #[inline]
    pub const fn kv_cache_bytes(&self) -> usize {
        self.kv_cache_floats * F32
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

    fn stories15m() -> Config {
        Config {
            dim: 288,
            hidden_dim: 768,
            n_layers: 6,
            n_heads: 6,
            n_kv_heads: 6,
            vocab_size: 32000,
            seq_len: 256,
            shared_weights: true,
        }
    }

    #[test]
    fn activation_budget_matches_hand_count() {
        let c = stories15m();
        // 3*288 + 2*768 + 288 + 6*256 + 32000  (fp32 working set; int8 scratch is separate)
        let expect = 864 + 1536 + 288 + 1536 + 32000;
        assert_eq!(activation_floats(&c), expect);
        // max_d_in = max(dim=288, att_dim=288, hidden=768) = 768 sizes the QuantScratch.
        assert_eq!(max_proj_d_in(&c), 768);
    }

    #[test]
    fn kv_cache_budget_matches_hand_count() {
        let c = stories15m();
        // 2 * 6 * 256 * 288
        assert_eq!(kv_cache_floats(&c), 2 * 6 * 256 * 288);
        assert_eq!(kv_cache_floats(&c), 884_736);
    }

    #[test]
    fn total_budget_is_about_3_5_mib() {
        let c = stories15m();
        let b = MemoryBudget::for_config(&c);
        assert_eq!(b.total_floats(), 36_224 + 884_736);
        // The KV cache is the overwhelming majority of the arena.
        assert!(b.kv_cache_floats > 20 * b.activation_floats);
        // 920,960 f32 -> 3,683,840 bytes -> ~3.51 MiB (~3.68 MB).
        let mib = b.total_bytes() as f64 / (1024.0 * 1024.0);
        assert!((3.4..3.6).contains(&mib), "got {mib} MiB");
    }

    #[test]
    fn budget_is_const_evaluable() {
        // The whole budget is `const fn`, so an embedded host can size a `static`/stack
        // arena and `const`-assert it fits a fixed RAM budget at compile time (exactly as
        // `examples/baremetal.rs` does). These `const`s force that evaluation in a const
        // context: if any budget function stopped being `const fn`, this would fail to
        // compile rather than silently regress the bare-metal story.
        const C: Config = Config {
            dim: 288,
            hidden_dim: 768,
            n_layers: 6,
            n_heads: 6,
            n_kv_heads: 6,
            vocab_size: 32000,
            seq_len: 256,
            shared_weights: true,
        };
        const FLOATS: usize = arena_floats(&C);
        const BUDGET: MemoryBudget = MemoryBudget::for_config(&C);
        const _: () = assert!(BUDGET.total_bytes() == FLOATS * F32);
        assert_eq!(FLOATS, 36_224 + 884_736);
        assert_eq!(BUDGET.total_floats(), FLOATS);
        assert_eq!(max_proj_d_in(&C), 768);
    }
}
