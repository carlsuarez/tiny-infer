//! Int8 quantization of the Llama weight set.
//!
//! Builds on the shared [`engine::quant`] primitives ([`QuantizedTensor`],
//! [`quantize`]) to quantize the matmul-heavy matrices of a decoder-only
//! checkpoint, cutting their memory footprint ~4×. Only the RMSNorm gains
//! (`rms_att`, `rms_ffn`, `rms_final`) stay `f32` — they are 1-D and tiny.
//! Everything else is quantized, including the token-embedding / classifier table:
//!
//! * the seven layer projections `wq, wk, wv, wo, w1, w2, w3`, and
//! * the `token_embedding` table, which doubles as the `wcls` classifier when the
//!   checkpoint shares weights. It is stored **once** as int8: the classifier reads
//!   it as a matmul ([`matmul_q8`](engine::math::matmul_q8)), and the embedding lookup
//!   dequantizes just the one row it needs ([`QuantizedTensor::layer`] +
//!   [`dequantize`](engine::quant::dequantize)). When weights are not shared, `wcls`
//!   is a second quantized table.
//!
//! Storing the embedding once as int8 — instead of keeping a full fp32 copy just for
//! the lookup — is the bulk of the memory win on these embedding-dominated models.
//!
//! ## Where the buffers come from
//!
//! The engine never allocates. The host quantizes the fp32 checkpoint **once** at
//! load time into two host-owned buffers — an `i8` data buffer
//! ([`quantized_weight_count`] long) and an `f32` scales buffer
//! ([`quantized_scale_count`] long) — using [`quantize_weights`], then hands them to
//! [`QuantizedWeights::new`] along with the three fp32 RMSNorm gains.
//! [`quantize_weights`] and [`QuantizedWeights::new`] walk the tensors in the **same
//! order** (`token_embedding`, the seven projections, then `wcls` only when weights
//! are not shared), so the fill and the view always agree. Because `new` borrows
//! nothing from the original fp32 file, the host can free that file once the int8
//! buffers and the gains are in hand.

use engine::error::EngineError;
use crate::config::Config;
use crate::weights::Weights;
use engine::quant::{quantize, QuantizedTensor};

/// Lengths (in weights) of the always-quantized tensors, in storage order: the
/// `token_embedding` table first (it doubles as `wcls` when weights are shared), then
/// the seven layer projections.
///
/// This is the single source of truth for *which* tensors are quantized and in what
/// order; [`quantize_weights`] and [`QuantizedWeights::new`] both iterate it so the
/// fill and the view can never drift apart. A non-shared `wcls` is a separate
/// `[vocab, dim]` block appended after these (see [`quantized_weight_count`]).
fn quantized_lens(c: &Config) -> [usize; 8] {
    let att_dim = c.n_heads * c.head_size();
    let kv_dim = c.kv_dim();
    let l = c.n_layers;
    [
        c.vocab_size * c.dim,     // token_embedding (also wcls when shared)
        l * c.dim * att_dim,      // wq
        l * c.dim * kv_dim,       // wk
        l * c.dim * kv_dim,       // wv
        l * att_dim * c.dim,      // wo
        l * c.hidden_dim * c.dim, // w1
        l * c.dim * c.hidden_dim, // w2
        l * c.hidden_dim * c.dim, // w3
    ]
}

/// Total number of `i8` weights a checkpoint of this config needs once quantized.
/// Use this to size the host's data buffer. Includes a separate `wcls` table only
/// when the checkpoint does not share weights.
#[inline(always)]
pub fn quantized_weight_count(c: &Config) -> usize {
    let mut n: usize = quantized_lens(c).iter().sum();
    if !c.shared_weights {
        n += c.vocab_size * c.dim; // separate classifier
    }
    n
}

/// Total number of `f32` scales the quantized weights need for `group_size`. Use
/// this to size the host's scales buffer.
#[inline(always)]
pub fn quantized_scale_count(c: &Config, group_size: usize) -> usize {
    quantized_weight_count(c) / group_size
}

/// The sizes (in weights) of the quantized tensors as a **v2 checkpoint** stores
/// them, in file order.
///
/// A v2 file (the quantized int8 format) serializes each tensor as its int8 data
/// immediately followed by its `f32` scales — and, unlike [`quantize_weights`]'
/// two flat buffers, every *layer* of a projection is its own tensor. The order is
/// the embedding, then for each projection all `n_layers` layer matrices
/// (`wq` × L, `wk` × L, `wv` × L, `wo` × L, `w1` × L, `w2` × L, `w3` × L), then the
/// classifier when weights are not shared. The host walks this sequence to
/// de-interleave the file into the flat layout [`QuantizedWeights::new`] expects;
/// the per-tensor concatenation of data (and of scales) in this order is exactly
/// that flat layout. The sizes sum to [`quantized_weight_count`].
pub fn v2_tensor_sizes(c: &Config) -> impl Iterator<Item = usize> {
    let att_dim = c.n_heads * c.head_size();
    let kv_dim = c.kv_dim();
    // One layer of each projection, in v2 file order.
    let per_layer: [usize; 7] = [
        c.dim * att_dim,      // wq
        c.dim * kv_dim,       // wk
        c.dim * kv_dim,       // wv
        att_dim * c.dim,      // wo
        c.hidden_dim * c.dim, // w1
        c.dim * c.hidden_dim, // w2
        c.hidden_dim * c.dim, // w3
    ];
    let emb = c.vocab_size * c.dim;
    let n_layers = c.n_layers;
    let wcls = (!c.shared_weights).then_some(emb);

    core::iter::once(emb)
        .chain(
            per_layer
                .into_iter()
                .flat_map(move |n| core::iter::repeat_n(n, n_layers)),
        )
        .chain(wcls)
}

/// Quantize the matmul matrices of `fp32` into `data` + `scales`.
///
/// Fills the caller-provided buffers in storage order — `token_embedding`, then
/// `wq, wk, wv, wo, w1, w2, w3`, then `wcls` only when weights are not shared (when
/// they are, `wcls` *is* `token_embedding`, already written). This is the same order
/// [`QuantizedWeights::new`] reads them back. `data` must be
/// [`quantized_weight_count`] long and `scales` [`quantized_scale_count`] long; both
/// are filled completely. Allocates nothing, so the host can call it over buffers it
/// owns.
///
/// # Panics
/// In debug builds, if `data` / `scales` are not the exact required lengths, or if
/// `group_size` does not divide every tensor.
pub fn quantize_weights(
    fp32: &Weights,
    data: &mut [i8],
    scales: &mut [f32],
    group_size: usize,
    c: &Config,
) {
    debug_assert_eq!(data.len(), quantized_weight_count(c));
    debug_assert_eq!(scales.len(), quantized_scale_count(c, group_size));

    // Advance disjoint cursors through both output buffers in lockstep.
    let mut data = data;
    let mut scales = scales;
    let mut emit = |src: &[f32]| {
        let n = src.len();
        let g = n / group_size;
        let (qd, dtail) = core::mem::take(&mut data).split_at_mut(n);
        let (qs, stail) = core::mem::take(&mut scales).split_at_mut(g);
        quantize(qd, qs, src, group_size);
        data = dtail;
        scales = stail;
    };

    emit(fp32.token_embedding);
    emit(fp32.wq);
    emit(fp32.wk);
    emit(fp32.wv);
    emit(fp32.wo);
    emit(fp32.w1);
    emit(fp32.w2);
    emit(fp32.w3);
    if !c.shared_weights {
        emit(fp32.wcls);
    }
}

/// Borrowed, zero-copy views of every Llama weight tensor, with the projections
/// quantized.
///
/// The quantized analogue of [`Weights`]: the seven projection matrices are
/// [`QuantizedTensor`]s viewing the host's `i8`/`f32` buffers, while the RMSNorm
/// gains, the token embedding, and the classifier stay full-precision `&[f32]`.
/// `wcls` aliases `token_embedding` when the checkpoint shares weights.
#[derive(Debug, Clone, Copy)]
pub struct QuantizedWeights<'a> {
    /// Token embedding table, `[vocab, dim]`, quantized. The lookup dequantizes one
    /// row at a time; it also serves as `wcls` when the checkpoint shares weights.
    pub token_embedding: QuantizedTensor<'a>,
    /// Per-layer attention RMSNorm gains, `[n_layers, dim]` (kept fp32).
    pub rms_att: &'a [f32],
    /// Query projection, `[n_layers, dim, att_dim]`.
    pub wq: QuantizedTensor<'a>,
    /// Key projection, `[n_layers, dim, kv_dim]`.
    pub wk: QuantizedTensor<'a>,
    /// Value projection, `[n_layers, dim, kv_dim]`.
    pub wv: QuantizedTensor<'a>,
    /// Attention output projection, `[n_layers, att_dim, dim]`.
    pub wo: QuantizedTensor<'a>,
    /// Per-layer feed-forward RMSNorm gains, `[n_layers, dim]` (kept fp32).
    pub rms_ffn: &'a [f32],
    /// SwiGLU gate projection, `[n_layers, hidden, dim]`.
    pub w1: QuantizedTensor<'a>,
    /// SwiGLU down projection, `[n_layers, dim, hidden]`.
    pub w2: QuantizedTensor<'a>,
    /// SwiGLU up projection, `[n_layers, hidden, dim]`.
    pub w3: QuantizedTensor<'a>,
    /// Final RMSNorm gains, `[dim]` (kept fp32).
    pub rms_final: &'a [f32],
    /// Output classifier, `[vocab, dim]`, quantized. Aliases `token_embedding` when
    /// the checkpoint shares weights, otherwise its own quantized table.
    pub wcls: QuantizedTensor<'a>,
}

impl<'a> QuantizedWeights<'a> {
    /// Assemble quantized weight views from the host's quantized buffers and the
    /// three fp32 RMSNorm gains.
    ///
    /// Carves `token_embedding` and the seven projections from `data` / `scales`
    /// (filled by [`quantize_weights`] with the same config and `group_size`); `wcls`
    /// aliases `token_embedding` when weights are shared, else it is carved as a
    /// further table. The `rms_*` slices are stored as-is. This borrows **nothing**
    /// from the original fp32 checkpoint, so the host may free that file first.
    /// Returns [`EngineError::SizeMismatch`] if a buffer is shorter than required.
    pub fn new(
        data: &'a [i8],
        scales: &'a [f32],
        rms_att: &'a [f32],
        rms_ffn: &'a [f32],
        rms_final: &'a [f32],
        group_size: usize,
        c: &Config,
    ) -> Result<QuantizedWeights<'a>, EngineError> {
        let need_data = quantized_weight_count(c);
        let need_scales = quantized_scale_count(c, group_size);
        if data.len() < need_data || scales.len() < need_scales {
            return Err(EngineError::SizeMismatch {
                // Report the i8 shortfall (1 byte per weight); a wrong group_size is
                // the usual cause of a scales-length mismatch.
                expected: need_data,
                actual: data.len(),
            });
        }

        // Two cursors advancing in lockstep, mirroring `quantize_weights`' order.
        let mut drest = data;
        let mut srest = scales;
        let mut take = |n: usize| -> QuantizedTensor<'a> {
            let g = n / group_size;
            let (qd, dtail) = drest.split_at(n);
            let (qs, stail) = srest.split_at(g);
            drest = dtail;
            srest = stail;
            QuantizedTensor {
                data: qd,
                scales: qs,
                group_size,
            }
        };

        let lens = quantized_lens(c);
        let token_embedding = take(lens[0]);
        let wq = take(lens[1]);
        let wk = take(lens[2]);
        let wv = take(lens[3]);
        let wo = take(lens[4]);
        let w1 = take(lens[5]);
        let w2 = take(lens[6]);
        let w3 = take(lens[7]);
        // Shared weights reuse the embedding table as the classifier; otherwise it is
        // a separate quantized block right after the projections.
        let wcls = if c.shared_weights {
            token_embedding
        } else {
            take(c.vocab_size * c.dim)
        };

        Ok(QuantizedWeights {
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

    extern crate std;
    use std::vec;
    use std::vec::Vec;

    fn tiny_config(shared: bool) -> Config {
        Config {
            dim: 4,
            hidden_dim: 8,
            n_layers: 1,
            n_heads: 2,
            n_kv_heads: 2,
            vocab_size: 6,
            seq_len: 4,
            shared_weights: shared,
        }
    }

    fn quantized(c: &Config, gs: usize) -> (Vec<i8>, Vec<f32>) {
        // Distinct per-element values so the embedding and a separate wcls differ.
        let wbuf: Vec<f32> = (0..crate::weights::weight_floats(c))
            .map(|i| (i as f32).sin() * 0.1)
            .collect();
        let fp32 = Weights::new(&wbuf, c).unwrap();
        let mut data = vec![0i8; quantized_weight_count(c)];
        let mut scales = vec![0.0f32; quantized_scale_count(c, gs)];
        quantize_weights(&fp32, &mut data, &mut scales, gs, c);
        (data, scales)
    }

    #[test]
    fn shared_weights_alias_wcls_to_the_embedding() {
        let c = tiny_config(true);
        let gs = 2;
        let (data, scales) = quantized(&c, gs);
        let rms = vec![1.0f32; c.dim]; // any fp32 gains; not under test here
        let qw = QuantizedWeights::new(&data, &scales, &rms, &rms, &rms, gs, &c).unwrap();
        // The classifier reuses the exact same int8 table as the embedding.
        assert_eq!(qw.wcls.data.as_ptr(), qw.token_embedding.data.as_ptr());
        assert_eq!(qw.token_embedding.data.len(), c.vocab_size * c.dim);
    }

    #[test]
    fn v2_tensor_sizes_sum_to_the_quantized_weight_count() {
        for shared in [true, false] {
            let c = tiny_config(shared);
            let sizes: Vec<usize> = v2_tensor_sizes(&c).collect();
            // emb + 7 projections × n_layers (+ wcls when unshared).
            let expect_count = 1 + 7 * c.n_layers + usize::from(!shared);
            assert_eq!(sizes.len(), expect_count);
            assert_eq!(sizes.iter().sum::<usize>(), quantized_weight_count(&c));
            // First is the embedding; the unshared classifier is the same size last.
            assert_eq!(sizes[0], c.vocab_size * c.dim);
            if !shared {
                assert_eq!(*sizes.last().unwrap(), c.vocab_size * c.dim);
            }
        }
    }

    #[test]
    fn unshared_weights_give_wcls_its_own_table() {
        let c = tiny_config(false);
        let gs = 2;
        let (data, scales) = quantized(&c, gs);
        let rms = vec![1.0f32; c.dim];
        let qw = QuantizedWeights::new(&data, &scales, &rms, &rms, &rms, gs, &c).unwrap();
        assert_ne!(qw.wcls.data.as_ptr(), qw.token_embedding.data.as_ptr());
        assert_eq!(qw.wcls.data.len(), c.vocab_size * c.dim);
    }
}
