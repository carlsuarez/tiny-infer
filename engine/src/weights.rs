//! Zero-copy views into a checkpoint's `f32` weight region.
//!
//! After the [`Config`] header, an fp32 llama2.c checkpoint stores every weight
//! tensor as raw little-endian `f32`, back to back, in a fixed order. We never
//! copy or reshape them: [`Weights`] holds borrowed sub-slices that point straight
//! into the (host-owned, memory-mapped or read) file buffer.
//!
//! The two fp32 formats store the **same tensors in different orders**
//! ([`Weights::new`] parses legacy, [`Weights::new_v1`] parses v1); the int8 v2
//! format is handled by [`crate::quantize`] instead. With
//! `att_dim = n_heads * head_size` and `kv_dim`:
//!
//! | legacy (v0)         | v1                  | length (`f32`)               |
//! |---------------------|---------------------|------------------------------|
//! | `token_embedding`   | `rms_att`           | (see own column)             |
//! | `rms_att`           | `rms_ffn`           | `n_layers * dim`             |
//! | `wq`                | `rms_final`         | `dim`                        |
//! | `wk`                | `token_embedding`   | `vocab * dim`                |
//! | `wv`                | `wq`                | `n_layers * dim * att_dim`   |
//! | `wo`                | `wk`                | `n_layers * dim * kv_dim`    |
//! | `rms_ffn`           | `wv`                | `n_layers * dim * kv_dim`    |
//! | `w1`                | `wo`                | `n_layers * att_dim * dim`   |
//! | `w2`                | `w1`                | `n_layers * hidden * dim`    |
//! | `w3`                | `w2`                | `n_layers * dim * hidden`    |
//! | `rms_final`         | `w3`                | `n_layers * hidden * dim`    |
//! | `freq_cis_real`/`imag` *(legacy only, skipped — RoPE is computed live)* | — | `seq_len * head_size/2` × 2 |
//! | `wcls` *(only if not shared)* | `wcls` *(ditto)* | `vocab * dim`     |

use crate::config::{Config, ModelFormat};
use crate::error::EngineError;

/// Number of `f32` elements a **legacy** checkpoint stores after the header —
/// including the two skipped `freq_cis` tables, and the separate classifier
/// matrix only when weights are not shared.
pub fn weight_floats(c: &Config) -> usize {
    let freq = c.seq_len * (c.head_size() / 2);
    // Same tensors as v1 plus the two legacy-only freq_cis tables.
    weight_floats_v1(c) + 2 * freq
}

/// Number of `f32` elements a **v1** checkpoint stores after its 256-byte header:
/// every legacy tensor except the `freq_cis` tables (v1 dropped them).
pub fn weight_floats_v1(c: &Config) -> usize {
    let att_dim = c.n_heads * c.head_size();
    let kv_dim = c.kv_dim();
    let l = c.n_layers;

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
    if !c.shared_weights {
        n += c.vocab_size * c.dim; // wcls
    }
    n
}

/// Total on-disk size of a checkpoint with this config and format, in bytes:
/// the format's header plus its weight payload.
///
/// For [`ModelFormat::V2`] the payload is the fp32 RMSNorm gains followed by the
/// int8 weights and their `f32` scales (see [`crate::quantize`]).
pub fn expected_file_bytes(c: &Config, format: ModelFormat) -> usize {
    const F32: usize = core::mem::size_of::<f32>();
    format.header_bytes()
        + match format {
            ModelFormat::Legacy => weight_floats(c) * F32,
            ModelFormat::V1 => weight_floats_v1(c) * F32,
            ModelFormat::V2 { group_size } => {
                let norms = 2 * c.n_layers * c.dim + c.dim; // rms_att + rms_ffn + rms_final
                let data = crate::quantize::quantized_weight_count(c);
                let scales = crate::quantize::quantized_scale_count(c, group_size);
                norms * F32 + data + scales * F32
            }
        }
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

    /// Carve the weight tensors out of a **v1** checkpoint's `f32` region.
    ///
    /// Same contract as [`Weights::new`], but in the v1 on-disk order: the three
    /// RMSNorm gain groups first, then the embedding and the seven projections,
    /// with no `freq_cis` tables. The resulting view is indistinguishable from a
    /// legacy one — only the carving order differs.
    pub fn new_v1(floats: &'a [f32], c: &Config) -> Result<Weights<'a>, EngineError> {
        let needed = weight_floats_v1(c);
        if floats.len() < needed {
            return Err(EngineError::SizeMismatch {
                expected: needed * core::mem::size_of::<f32>(),
                actual: core::mem::size_of_val(floats),
            });
        }

        let att_dim = c.n_heads * c.head_size();
        let kv_dim = c.kv_dim();
        let l = c.n_layers;

        let mut rest = floats;
        let mut take = |n: usize| -> &'a [f32] {
            let (head, tail) = rest.split_at(n);
            rest = tail;
            head
        };

        let rms_att = take(l * c.dim);
        let rms_ffn = take(l * c.dim);
        let rms_final = take(c.dim);
        let token_embedding = take(c.vocab_size * c.dim);
        let wq = take(l * c.dim * att_dim);
        let wk = take(l * c.dim * kv_dim);
        let wv = take(l * c.dim * kv_dim);
        let wo = take(l * att_dim * c.dim);
        let w1 = take(l * c.hidden_dim * c.dim);
        let w2 = take(l * c.dim * c.hidden_dim);
        let w3 = take(l * c.hidden_dim * c.dim);
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
        assert_eq!(
            expected_file_bytes(&c, ModelFormat::Legacy),
            crate::config::HEADER_BYTES + expect * 4
        );
        // v1 drops the two freq_cis tables (12 floats here) and pays the bigger header.
        assert_eq!(weight_floats_v1(&c), expect - 12);
        assert_eq!(
            expected_file_bytes(&c, ModelFormat::V1),
            crate::config::VERSIONED_HEADER_BYTES + (expect - 12) * 4
        );
    }

    #[test]
    fn expected_v2_bytes_counts_norms_data_and_scales() {
        let c = tiny_config(true);
        let gs = 2;
        // fp32 norms: rms_att 8 + rms_ffn 8 + rms_final 4 = 20 floats.
        // int8 data: every quantized weight, 1 byte each; scales: one f32 per group.
        let data = crate::quantize::quantized_weight_count(&c);
        let expect =
            crate::config::VERSIONED_HEADER_BYTES + 20 * 4 + data + (data / gs) * 4;
        assert_eq!(
            expected_file_bytes(&c, ModelFormat::V2 { group_size: gs }),
            expect
        );
    }

    #[test]
    fn v1_views_carve_the_same_tensors_at_their_v1_offsets() {
        // Fill the buffer with its own indices so each slice's first element
        // reveals the offset it was carved from — proving the v1 order.
        let c = tiny_config(true);
        let buf: std::vec::Vec<f32> = (0..weight_floats_v1(&c)).map(|i| i as f32).collect();
        let w = Weights::new_v1(&buf, &c).unwrap();

        // v1 order: rms_att(8) | rms_ffn(8) | rms_final(4) | emb(20) | wq(32) | ...
        assert_eq!(w.rms_att[0], 0.0);
        assert_eq!(w.rms_ffn[0], 8.0);
        assert_eq!(w.rms_final[0], 16.0);
        assert_eq!(w.token_embedding[0], 20.0);
        assert_eq!(w.wq[0], 40.0);
        assert_eq!(w.wk[0], 72.0);
        assert_eq!(w.wv[0], 88.0);
        assert_eq!(w.wo[0], 104.0);
        assert_eq!(w.w1[0], 136.0);
        assert_eq!(w.w2[0], 200.0);
        assert_eq!(w.w3[0], 264.0);
        // Shared: wcls aliases the embedding.
        assert_eq!(w.wcls.as_ptr(), w.token_embedding.as_ptr());
    }

    #[test]
    fn v1_unshared_wcls_is_the_tail() {
        let c = tiny_config(false);
        let buf: std::vec::Vec<f32> = (0..weight_floats_v1(&c)).map(|i| i as f32).collect();
        let w = Weights::new_v1(&buf, &c).unwrap();
        assert_eq!(w.wcls.len(), c.vocab_size * c.dim);
        assert_eq!(w.wcls[0], (weight_floats_v1(&c) - c.vocab_size * c.dim) as f32);
    }

    #[test]
    fn v1_short_slice_is_rejected() {
        let c = tiny_config(true);
        let buf = alloc_floats(weight_floats_v1(&c) - 1);
        assert!(matches!(
            Weights::new_v1(&buf, &c),
            Err(EngineError::SizeMismatch { .. })
        ));
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
