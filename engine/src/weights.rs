//! Zero-copy views into a checkpoint's `f32` weight region.
//!
//! After the [`Config`] header, a legacy llama2.c checkpoint stores every weight
//! tensor as raw little-endian `f32`, back to back, in a fixed order. We never
//! copy or reshape them: [`Weights`] holds borrowed sub-slices that point straight
//! into the (host-owned, memory-mapped or read) file buffer.
//!
//! The on-disk order, with `att_dim = n_heads * head_size` and `kv_dim`:
//!
//! | # | tensor              | length (`f32`)               |
//! |---|---------------------|------------------------------|
//! | 1 | `token_embedding`   | `vocab * dim`                |
//! | 2 | `rms_att`           | `n_layers * dim`             |
//! | 3 | `wq`                | `n_layers * dim * att_dim`   |
//! | 4 | `wk`                | `n_layers * dim * kv_dim`    |
//! | 5 | `wv`                | `n_layers * dim * kv_dim`    |
//! | 6 | `wo`                | `n_layers * att_dim * dim`   |
//! | 7 | `rms_ffn`           | `n_layers * dim`             |
//! | 8 | `w1`                | `n_layers * hidden * dim`    |
//! | 9 | `w2`                | `n_layers * dim * hidden`    |
//! |10 | `w3`                | `n_layers * hidden * dim`    |
//! |11 | `rms_final`         | `dim`                        |
//! |12 | `freq_cis_real`     | `seq_len * head_size/2` *(skipped — RoPE is computed live)* |
//! |13 | `freq_cis_imag`     | `seq_len * head_size/2` *(skipped)* |
//! |14 | `wcls`              | `vocab * dim` *(only if not shared)* |

use crate::config::{Config, HEADER_BYTES};
use crate::error::EngineError;

/// Number of `f32` elements the checkpoint stores after the header for a given
/// config — including the two skipped `freq_cis` tables, and the separate
/// classifier matrix only when weights are not shared.
pub fn weight_floats(c: &Config) -> usize {
    let att_dim = c.n_heads * c.head_size();
    let kv_dim = c.kv_dim();
    let l = c.n_layers;
    let freq = c.seq_len * (c.head_size() / 2);

    let mut n = 0;
    n += c.vocab_size * c.dim; // token_embedding
    n += l * c.dim; // rms_att
    n += l * c.dim * att_dim; // wq
    n += l * c.dim * kv_dim; // wk
    n += l * c.dim * kv_dim; // wv
    n += l * att_dim * c.dim; // wo
    n += l * c.dim; // rms_ffn
    n += l * c.hidden_dim * c.dim; // w1
    n += l * c.dim * c.hidden_dim; // w2
    n += l * c.hidden_dim * c.dim; // w3
    n += c.dim; // rms_final
    n += freq; // freq_cis_real (skipped at runtime)
    n += freq; // freq_cis_imag (skipped at runtime)
    if !c.shared_weights {
        n += c.vocab_size * c.dim; // wcls
    }
    n
}

/// Total on-disk size of a checkpoint with this config, in bytes:
/// the 28-byte header plus every weight `f32`.
pub fn expected_file_bytes(c: &Config) -> usize {
    HEADER_BYTES + weight_floats(c) * core::mem::size_of::<f32>()
}

/// Borrowed, zero-copy views of every weight tensor.
///
/// Each field is a flat slice; per-layer indexing (e.g. row `l` of `wq`) is the
/// caller's responsibility in later milestones. `wcls` aliases `token_embedding`
/// when the checkpoint shares weights.
#[derive(Debug, Clone, Copy)]
pub struct Weights<'a> {
    /// Token embedding table, `[vocab, dim]`.
    pub token_embedding: &'a [f32],
    /// Per-layer attention RMSNorm gains, `[n_layers, dim]`.
    pub rms_att: &'a [f32],
    /// Query projection, `[n_layers, dim, att_dim]`.
    pub wq: &'a [f32],
    /// Key projection, `[n_layers, dim, kv_dim]`.
    pub wk: &'a [f32],
    /// Value projection, `[n_layers, dim, kv_dim]`.
    pub wv: &'a [f32],
    /// Attention output projection, `[n_layers, att_dim, dim]`.
    pub wo: &'a [f32],
    /// Per-layer feed-forward RMSNorm gains, `[n_layers, dim]`.
    pub rms_ffn: &'a [f32],
    /// SwiGLU gate projection, `[n_layers, hidden, dim]`.
    pub w1: &'a [f32],
    /// SwiGLU down projection, `[n_layers, dim, hidden]`.
    pub w2: &'a [f32],
    /// SwiGLU up projection, `[n_layers, hidden, dim]`.
    pub w3: &'a [f32],
    /// Final RMSNorm gains, `[dim]`.
    pub rms_final: &'a [f32],
    /// Output classifier, `[vocab, dim]` (aliases `token_embedding` if shared).
    pub wcls: &'a [f32],
}

impl<'a> Weights<'a> {
    /// Carve the weight tensors out of the checkpoint's `f32` region.
    ///
    /// `floats` must be the file's contents *after* the header, reinterpreted as
    /// `f32` (the host performs that aligned cast). Returns
    /// [`EngineError::SizeMismatch`] if `floats` is shorter than the config
    /// requires.
    pub fn new(floats: &'a [f32], c: &Config) -> Result<Weights<'a>, EngineError> {
        let needed = weight_floats(c);
        if floats.len() < needed {
            return Err(EngineError::SizeMismatch {
                expected: needed * core::mem::size_of::<f32>(),
                actual: core::mem::size_of_val(floats),
            });
        }

        let att_dim = c.n_heads * c.head_size();
        let kv_dim = c.kv_dim();
        let l = c.n_layers;
        let freq = c.seq_len * (c.head_size() / 2);

        // Bump a cursor through the slice, taking each tensor in file order.
        let mut rest = floats;
        let mut take = |n: usize| -> &'a [f32] {
            let (head, tail) = rest.split_at(n);
            rest = tail;
            head
        };

        let token_embedding = take(c.vocab_size * c.dim);
        let rms_att = take(l * c.dim);
        let wq = take(l * c.dim * att_dim);
        let wk = take(l * c.dim * kv_dim);
        let wv = take(l * c.dim * kv_dim);
        let wo = take(l * att_dim * c.dim);
        let rms_ffn = take(l * c.dim);
        let w1 = take(l * c.hidden_dim * c.dim);
        let w2 = take(l * c.dim * c.hidden_dim);
        let w3 = take(l * c.hidden_dim * c.dim);
        let rms_final = take(c.dim);
        let _freq_cis_real = take(freq); // skipped: RoPE is computed on the fly
        let _freq_cis_imag = take(freq);
        let wcls = if c.shared_weights {
            token_embedding
        } else {
            take(c.vocab_size * c.dim)
        };

        Ok(Weights {
            token_embedding,
            rms_att,
            wq,
            wk,
            wv,
            wo,
            rms_ffn,
            w1,
            w2,
            w3,
            rms_final,
            wcls,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config(shared: bool) -> Config {
        Config {
            dim: 4,
            hidden_dim: 8,
            n_layers: 2,
            n_heads: 2,
            n_kv_heads: 1,
            vocab_size: 5,
            seq_len: 6,
            shared_weights: shared,
        }
    }

    #[test]
    fn weight_floats_matches_hand_sum() {
        let c = tiny_config(true);
        // att_dim = 2*2 = 4, kv_dim = 1*2 = 2, freq = 6 * (2/2) = 6.
        // emb 20 | rms_att 8 | wq 2*4*4=32 | wk 2*4*2=16 | wv 16 | wo 2*4*4=32
        // rms_ffn 8 | w1 2*8*4=64 | w2 64 | w3 64 | rms_final 4 | freq*2 = 12
        // shared -> no wcls
        let expect = 20 + 8 + 32 + 16 + 16 + 32 + 8 + 64 + 64 + 64 + 4 + 12;
        assert_eq!(weight_floats(&c), expect);
        assert_eq!(expected_file_bytes(&c), HEADER_BYTES + expect * 4);
    }

    #[test]
    fn unshared_adds_classifier() {
        let shared = weight_floats(&tiny_config(true));
        let unshared = weight_floats(&tiny_config(false));
        assert_eq!(unshared - shared, 5 * 4); // vocab * dim
    }

    #[test]
    fn views_partition_the_slice_and_alias_wcls() {
        let c = tiny_config(true);
        let buf = alloc_floats(weight_floats(&c));
        let w = Weights::new(&buf, &c).unwrap();
        assert_eq!(w.token_embedding.len(), c.vocab_size * c.dim);
        assert_eq!(w.wq.len(), c.n_layers * c.dim * c.dim);
        assert_eq!(w.rms_final.len(), c.dim);
        // Shared weights: wcls must be the very same slice as the embedding.
        assert_eq!(w.wcls.as_ptr(), w.token_embedding.as_ptr());
    }

    #[test]
    fn unshared_wcls_is_distinct_tail() {
        let c = tiny_config(false);
        let buf = alloc_floats(weight_floats(&c));
        let w = Weights::new(&buf, &c).unwrap();
        assert_ne!(w.wcls.as_ptr(), w.token_embedding.as_ptr());
        assert_eq!(w.wcls.len(), c.vocab_size * c.dim);
    }

    #[test]
    fn short_slice_is_rejected() {
        let c = tiny_config(true);
        let buf = alloc_floats(weight_floats(&c) - 1);
        assert!(matches!(
            Weights::new(&buf, &c),
            Err(EngineError::SizeMismatch { .. })
        ));
    }

    // Tiny test-only heap helper; the engine itself never allocates.
    extern crate std;
    fn alloc_floats(n: usize) -> std::vec::Vec<f32> {
        std::vec![0.0f32; n]
    }
}
