//! Int8 quantization of the seq2seq (Marian / OPUS-MT) weight set.
//!
//! The encoder-decoder counterpart of [`crate::llama::quantize`], built on the same
//! shared [`crate::quant`] primitives ([`QuantizedTensor`], [`quantize`]). It splits a
//! Marian checkpoint's tensors into two groups:
//!
//! * **Quantized** — the 17 matmul-heavy matrices, the ones whose `W · x` dominates the
//!   compute and the memory: the shared `token_embedding` (which also serves as the tied
//!   lm_head), the encoder's `wq/wk/wv/wo` and `fc1/fc2`, and the decoder's self-,
//!   cross-attention `wq/wk/wv/wo` and `fc1/fc2`. These become [`QuantizedTensor`]s over
//!   the host's `i8`/`f32` buffers.
//! * **Kept fp32** — everything 1-D and tiny: every projection **bias**, every
//!   **LayerNorm** weight and bias, and the `final_logits_bias`. Quantizing these would
//!   save almost nothing and cost accuracy, so they stay full precision in a
//!   [`KeptWeights`] bundle. Unlike the Llama path's three RMSNorm gains, a Marian model
//!   keeps ~27 such tensors, so they travel together as one packed buffer rather than as
//!   loose arguments.
//!
//! ## Where the buffers come from
//!
//! The engine never allocates. The host quantizes the fp32 checkpoint **once** at load
//! time into an `i8` data buffer ([`quantized_weight_count`] long) plus an `f32` scales
//! buffer ([`quantized_scale_count`] long) with [`quantize_weights`], and copies the kept
//! fp32 tensors into a [`kept_floats`]-long buffer with [`pack_kept`]. It then hands those
//! to [`QuantizedWeights::new`] and [`KeptWeights::carve`]. Both walk their tensors in the
//! **same order** the fillers wrote them, so the views always agree with the buffers, and
//! because neither borrows the original fp32 file the host can free it once the int8 data,
//! the scales, and the kept buffer are in hand — the whole point of quantizing at load.

use crate::error::EngineError;
use crate::quant::{quantize, QuantizedTensor};
use crate::seq2seq::config::Config;
use crate::seq2seq::weights::Weights;

/// Lengths (in weights) of the 17 always-quantized matmul tensors, in storage order:
/// the shared `token_embedding`, the encoder's six projections, then the decoder's ten
/// (self + cross attention, then the two FFN matrices).
///
/// This is the single source of truth for *which* tensors are quantized and in what
/// order; [`quantize_weights`] and [`QuantizedWeights::new`] both iterate it so the fill
/// and the view can never drift apart.
fn quantized_lens(c: &Config) -> [usize; 17] {
    let d = c.d_model;
    let (le, ld) = (c.enc_layers, c.dec_layers);
    [
        c.vocab_size * d, // token_embedding (also the tied lm_head)
        le * d * d,       // enc_wq
        le * d * d,       // enc_wk
        le * d * d,       // enc_wv
        le * d * d,       // enc_wo
        le * c.enc_ffn * d, // enc_fc1_w
        le * d * c.enc_ffn, // enc_fc2_w
        ld * d * d,       // dec_wq
        ld * d * d,       // dec_wk
        ld * d * d,       // dec_wv
        ld * d * d,       // dec_wo
        ld * d * d,       // dec_cross_wq
        ld * d * d,       // dec_cross_wk
        ld * d * d,       // dec_cross_wv
        ld * d * d,       // dec_cross_wo
        ld * c.dec_ffn * d, // dec_fc1_w
        ld * d * c.dec_ffn, // dec_fc2_w
    ]
}

/// Total number of `i8` weights a seq2seq checkpoint needs once its matmul matrices are
/// quantized. Use this to size the host's data buffer.
#[inline(always)]
pub fn quantized_weight_count(c: &Config) -> usize {
    quantized_lens(c).iter().sum()
}

/// Total number of `f32` scales the quantized weights need for `group_size`. Use this to
/// size the host's scales buffer.
#[inline(always)]
pub fn quantized_scale_count(c: &Config, group_size: usize) -> usize {
    quantized_weight_count(c) / group_size
}

/// Number of `f32` the kept (un-quantized) tensors occupy: every projection bias, every
/// LayerNorm weight and bias, and the `final_logits_bias`. Use this to size the host's
/// packed kept buffer for [`pack_kept`] / [`KeptWeights::carve`].
///
/// Per layer that is `9·d + enc_ffn` for the encoder (four attention biases + two
/// LayerNorms = 6·d, the FFN's `fc1`/`fc2` biases = `enc_ffn + d`, its LayerNorm = 2·d)
/// and `15·d + dec_ffn` for the decoder (self- and cross-attention each contribute 6·d,
/// the FFN `dec_ffn + 3·d`), plus the `vocab`-long `final_logits_bias`.
#[inline(always)]
pub fn kept_floats(c: &Config) -> usize {
    let d = c.d_model;
    c.enc_layers * (9 * d + c.enc_ffn) + c.dec_layers * (15 * d + c.dec_ffn) + c.vocab_size
}

/// The largest matmul input dimension `d_in` across the quantized projections — the size
/// each buffer of the W8A8 activation scratch
/// ([`QuantScratch`](crate::QuantScratch)) must cover.
///
/// Attention and `fc1` projections take `d_model`; `fc2` takes the (wider) ffn dimension,
/// so the maximum is `max(d_model, enc_ffn, dec_ffn)`.
#[inline(always)]
pub fn max_proj_d_in(c: &Config) -> usize {
    c.d_model.max(c.enc_ffn).max(c.dec_ffn)
}

/// Quantize the 17 matmul matrices of `fp32` into `data` + `scales`.
///
/// Fills the caller-provided buffers in `quantized_lens` order — the same order
/// [`QuantizedWeights::new`] reads them back. `data` must be [`quantized_weight_count`]
/// long and `scales` [`quantized_scale_count`] long; both are filled completely.
/// Allocates nothing.
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
    emit(fp32.enc_wq);
    emit(fp32.enc_wk);
    emit(fp32.enc_wv);
    emit(fp32.enc_wo);
    emit(fp32.enc_fc1_w);
    emit(fp32.enc_fc2_w);
    emit(fp32.dec_wq);
    emit(fp32.dec_wk);
    emit(fp32.dec_wv);
    emit(fp32.dec_wo);
    emit(fp32.dec_cross_wq);
    emit(fp32.dec_cross_wk);
    emit(fp32.dec_cross_wv);
    emit(fp32.dec_cross_wo);
    emit(fp32.dec_fc1_w);
    emit(fp32.dec_fc2_w);
}

/// Copy the kept (un-quantized) tensors of `fp32` into `out`, in carve order.
///
/// The fp32 companion of [`quantize_weights`]: it packs every projection bias, LayerNorm
/// weight/bias, and the `final_logits_bias` into one contiguous buffer that
/// [`KeptWeights::carve`] reads back. `out` must be [`kept_floats`] long and is filled
/// completely.
///
/// # Panics
/// In debug builds, if `out` is not exactly [`kept_floats`] long.
pub fn pack_kept(fp32: &Weights, out: &mut [f32], c: &Config) {
    debug_assert_eq!(out.len(), kept_floats(c));
    let mut out = out;
    let mut put = |src: &[f32]| {
        let (head, tail) = core::mem::take(&mut out).split_at_mut(src.len());
        head.copy_from_slice(src);
        out = tail;
    };
    for src in kept_tensors(fp32) {
        put(src);
    }
}

/// The kept tensors of an fp32 weight view, in the fixed carve order shared by
/// [`pack_kept`], [`KeptWeights::carve`], and [`KeptWeights::from_weights`].
fn kept_tensors<'a>(w: &Weights<'a>) -> [&'a [f32]; 27] {
    [
        w.enc_bq,
        w.enc_bk,
        w.enc_bv,
        w.enc_bo,
        w.enc_ln_att_w,
        w.enc_ln_att_b,
        w.enc_fc1_b,
        w.enc_fc2_b,
        w.enc_ln_ffn_w,
        w.enc_ln_ffn_b,
        w.dec_bq,
        w.dec_bk,
        w.dec_bv,
        w.dec_bo,
        w.dec_ln_self_w,
        w.dec_ln_self_b,
        w.dec_cross_bq,
        w.dec_cross_bk,
        w.dec_cross_bv,
        w.dec_cross_bo,
        w.dec_ln_cross_w,
        w.dec_ln_cross_b,
        w.dec_fc1_b,
        w.dec_fc2_b,
        w.dec_ln_ffn_w,
        w.dec_ln_ffn_b,
        w.final_logits_bias,
    ]
}

/// Borrowed, zero-copy views of the 17 quantized seq2seq matmul matrices.
///
/// The quantized analogue of the matmul half of [`Weights`]: each field is a
/// [`QuantizedTensor`] spanning every layer of one projection (slice a single layer with
/// [`QuantizedTensor::layer`]). The kept fp32 tensors live separately in [`KeptWeights`].
#[derive(Debug, Clone, Copy)]
pub struct QuantizedWeights<'a> {
    /// Shared token embedding `[vocab, d]`, quantized — encoder/decoder input lookup
    /// (dequantized one row at a time) and the tied lm_head matmul.
    pub token_embedding: QuantizedTensor<'a>,
    /// Encoder query projection, `[enc_layers, d, d]`.
    pub enc_wq: QuantizedTensor<'a>,
    /// Encoder key projection, `[enc_layers, d, d]`.
    pub enc_wk: QuantizedTensor<'a>,
    /// Encoder value projection, `[enc_layers, d, d]`.
    pub enc_wv: QuantizedTensor<'a>,
    /// Encoder attention output projection, `[enc_layers, d, d]`.
    pub enc_wo: QuantizedTensor<'a>,
    /// Encoder FFN up projection, `[enc_layers, enc_ffn, d]`.
    pub enc_fc1_w: QuantizedTensor<'a>,
    /// Encoder FFN down projection, `[enc_layers, d, enc_ffn]`.
    pub enc_fc2_w: QuantizedTensor<'a>,
    /// Decoder self-attention query projection, `[dec_layers, d, d]`.
    pub dec_wq: QuantizedTensor<'a>,
    /// Decoder self-attention key projection, `[dec_layers, d, d]`.
    pub dec_wk: QuantizedTensor<'a>,
    /// Decoder self-attention value projection, `[dec_layers, d, d]`.
    pub dec_wv: QuantizedTensor<'a>,
    /// Decoder self-attention output projection, `[dec_layers, d, d]`.
    pub dec_wo: QuantizedTensor<'a>,
    /// Cross-attention query projection, `[dec_layers, d, d]`.
    pub dec_cross_wq: QuantizedTensor<'a>,
    /// Cross-attention key projection, `[dec_layers, d, d]`.
    pub dec_cross_wk: QuantizedTensor<'a>,
    /// Cross-attention value projection, `[dec_layers, d, d]`.
    pub dec_cross_wv: QuantizedTensor<'a>,
    /// Cross-attention output projection, `[dec_layers, d, d]`.
    pub dec_cross_wo: QuantizedTensor<'a>,
    /// Decoder FFN up projection, `[dec_layers, dec_ffn, d]`.
    pub dec_fc1_w: QuantizedTensor<'a>,
    /// Decoder FFN down projection, `[dec_layers, d, dec_ffn]`.
    pub dec_fc2_w: QuantizedTensor<'a>,
}

impl<'a> QuantizedWeights<'a> {
    /// Assemble the 17 quantized matmul views from the host's `data` / `scales` buffers.
    ///
    /// Carves the tensors in `quantized_lens` order (the order [`quantize_weights`]
    /// filled them, with the same config and `group_size`). Borrows **nothing** from the
    /// original fp32 checkpoint, so the host may free that file first. Returns
    /// [`EngineError::SizeMismatch`] if a buffer is shorter than required.
    pub fn new(
        data: &'a [i8],
        scales: &'a [f32],
        group_size: usize,
        c: &Config,
    ) -> Result<QuantizedWeights<'a>, EngineError> {
        let need_data = quantized_weight_count(c);
        let need_scales = quantized_scale_count(c, group_size);
        if data.len() < need_data || scales.len() < need_scales {
            return Err(EngineError::SizeMismatch {
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
        Ok(QuantizedWeights {
            token_embedding: take(lens[0]),
            enc_wq: take(lens[1]),
            enc_wk: take(lens[2]),
            enc_wv: take(lens[3]),
            enc_wo: take(lens[4]),
            enc_fc1_w: take(lens[5]),
            enc_fc2_w: take(lens[6]),
            dec_wq: take(lens[7]),
            dec_wk: take(lens[8]),
            dec_wv: take(lens[9]),
            dec_wo: take(lens[10]),
            dec_cross_wq: take(lens[11]),
            dec_cross_wk: take(lens[12]),
            dec_cross_wv: take(lens[13]),
            dec_cross_wo: take(lens[14]),
            dec_fc1_w: take(lens[15]),
            dec_fc2_w: take(lens[16]),
        })
    }
}

/// The kept (un-quantized) fp32 tensors: every projection bias, LayerNorm weight and bias,
/// and the `final_logits_bias`.
///
/// Used identically on both the fp32 and the int8 paths — these tensors stay full
/// precision regardless — so the forward pass reads them through this one bundle. On the
/// fp32 path it is built straight from the file with [`KeptWeights::from_weights`]; on the
/// int8 path it is carved from the host's packed buffer with [`KeptWeights::carve`]. Each
/// field is a flat slice over all layers; per-layer indexing is the forward pass's job.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KeptWeights<'a> {
    /// Encoder query bias, `[enc_layers, d]`.
    pub enc_bq: &'a [f32],
    /// Encoder key bias, `[enc_layers, d]`.
    pub enc_bk: &'a [f32],
    /// Encoder value bias, `[enc_layers, d]`.
    pub enc_bv: &'a [f32],
    /// Encoder attention output bias, `[enc_layers, d]`.
    pub enc_bo: &'a [f32],
    /// Encoder self-attention LayerNorm weight, `[enc_layers, d]`.
    pub enc_ln_att_w: &'a [f32],
    /// Encoder self-attention LayerNorm bias, `[enc_layers, d]`.
    pub enc_ln_att_b: &'a [f32],
    /// Encoder FFN up bias, `[enc_layers, enc_ffn]`.
    pub enc_fc1_b: &'a [f32],
    /// Encoder FFN down bias, `[enc_layers, d]`.
    pub enc_fc2_b: &'a [f32],
    /// Encoder FFN LayerNorm weight, `[enc_layers, d]`.
    pub enc_ln_ffn_w: &'a [f32],
    /// Encoder FFN LayerNorm bias, `[enc_layers, d]`.
    pub enc_ln_ffn_b: &'a [f32],
    /// Decoder self-attention query bias, `[dec_layers, d]`.
    pub dec_bq: &'a [f32],
    /// Decoder self-attention key bias, `[dec_layers, d]`.
    pub dec_bk: &'a [f32],
    /// Decoder self-attention value bias, `[dec_layers, d]`.
    pub dec_bv: &'a [f32],
    /// Decoder self-attention output bias, `[dec_layers, d]`.
    pub dec_bo: &'a [f32],
    /// Decoder self-attention LayerNorm weight, `[dec_layers, d]`.
    pub dec_ln_self_w: &'a [f32],
    /// Decoder self-attention LayerNorm bias, `[dec_layers, d]`.
    pub dec_ln_self_b: &'a [f32],
    /// Cross-attention query bias, `[dec_layers, d]`.
    pub dec_cross_bq: &'a [f32],
    /// Cross-attention key bias, `[dec_layers, d]`.
    pub dec_cross_bk: &'a [f32],
    /// Cross-attention value bias, `[dec_layers, d]`.
    pub dec_cross_bv: &'a [f32],
    /// Cross-attention output bias, `[dec_layers, d]`.
    pub dec_cross_bo: &'a [f32],
    /// Cross-attention LayerNorm weight, `[dec_layers, d]`.
    pub dec_ln_cross_w: &'a [f32],
    /// Cross-attention LayerNorm bias, `[dec_layers, d]`.
    pub dec_ln_cross_b: &'a [f32],
    /// Decoder FFN up bias, `[dec_layers, dec_ffn]`.
    pub dec_fc1_b: &'a [f32],
    /// Decoder FFN down bias, `[dec_layers, d]`.
    pub dec_fc2_b: &'a [f32],
    /// Decoder FFN LayerNorm weight, `[dec_layers, d]`.
    pub dec_ln_ffn_w: &'a [f32],
    /// Decoder FFN LayerNorm bias, `[dec_layers, d]`.
    pub dec_ln_ffn_b: &'a [f32],
    /// Bias added to the tied lm_head's logits, `[vocab]`.
    pub final_logits_bias: &'a [f32],
}

impl<'a> KeptWeights<'a> {
    /// Borrow the kept tensors straight from an fp32 weight view (the fp32 path).
    ///
    /// Copies the 27 slice references out of `w`; the result borrows the same underlying
    /// file (`'a`), not `w` itself, so the caller can move `w` into a
    /// [`ModelWeights`](crate::seq2seq::ModelWeights) afterward.
    pub fn from_weights(w: &Weights<'a>) -> KeptWeights<'a> {
        let t = kept_tensors(w);
        KeptWeights {
            enc_bq: t[0],
            enc_bk: t[1],
            enc_bv: t[2],
            enc_bo: t[3],
            enc_ln_att_w: t[4],
            enc_ln_att_b: t[5],
            enc_fc1_b: t[6],
            enc_fc2_b: t[7],
            enc_ln_ffn_w: t[8],
            enc_ln_ffn_b: t[9],
            dec_bq: t[10],
            dec_bk: t[11],
            dec_bv: t[12],
            dec_bo: t[13],
            dec_ln_self_w: t[14],
            dec_ln_self_b: t[15],
            dec_cross_bq: t[16],
            dec_cross_bk: t[17],
            dec_cross_bv: t[18],
            dec_cross_bo: t[19],
            dec_ln_cross_w: t[20],
            dec_ln_cross_b: t[21],
            dec_fc1_b: t[22],
            dec_fc2_b: t[23],
            dec_ln_ffn_w: t[24],
            dec_ln_ffn_b: t[25],
            final_logits_bias: t[26],
        }
    }

    /// Carve the kept tensors from a host buffer packed by [`pack_kept`] (the int8 path).
    ///
    /// `packed` must be [`kept_floats`] long; tensors are taken in carve order. Borrows
    /// nothing from the original fp32 file. Returns [`EngineError::SizeMismatch`] if
    /// `packed` is too short.
    pub fn carve(packed: &'a [f32], c: &Config) -> Result<KeptWeights<'a>, EngineError> {
        let need = kept_floats(c);
        if packed.len() < need {
            return Err(EngineError::SizeMismatch {
                expected: need,
                actual: packed.len(),
            });
        }
        let d = c.d_model;
        let (le, ld) = (c.enc_layers, c.dec_layers);
        let mut rest = packed;
        let mut take = |n: usize| -> &'a [f32] {
            let (head, tail) = rest.split_at(n);
            rest = tail;
            head
        };
        Ok(KeptWeights {
            enc_bq: take(le * d),
            enc_bk: take(le * d),
            enc_bv: take(le * d),
            enc_bo: take(le * d),
            enc_ln_att_w: take(le * d),
            enc_ln_att_b: take(le * d),
            enc_fc1_b: take(le * c.enc_ffn),
            enc_fc2_b: take(le * d),
            enc_ln_ffn_w: take(le * d),
            enc_ln_ffn_b: take(le * d),
            dec_bq: take(ld * d),
            dec_bk: take(ld * d),
            dec_bv: take(ld * d),
            dec_bo: take(ld * d),
            dec_ln_self_w: take(ld * d),
            dec_ln_self_b: take(ld * d),
            dec_cross_bq: take(ld * d),
            dec_cross_bk: take(ld * d),
            dec_cross_bv: take(ld * d),
            dec_cross_bo: take(ld * d),
            dec_ln_cross_w: take(ld * d),
            dec_ln_cross_b: take(ld * d),
            dec_fc1_b: take(ld * c.dec_ffn),
            dec_fc2_b: take(ld * d),
            dec_ln_ffn_w: take(ld * d),
            dec_ln_ffn_b: take(ld * d),
            final_logits_bias: take(c.vocab_size),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seq2seq::config::Activation;
    use crate::seq2seq::weights::weight_floats;

    extern crate std;
    use std::vec;
    use std::vec::Vec;

    fn tiny_config() -> Config {
        Config {
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
            activation: Activation::Swish,
            scale_embedding: true,
        }
    }

    fn fake_weights(c: &Config) -> Vec<f32> {
        (0..weight_floats(c)).map(|i| (i as f32).sin() * 0.1).collect()
    }

    #[test]
    fn counts_partition_the_weight_payload() {
        // Every weight float is either quantized or kept, exactly once: the two counts
        // must sum to the full fp32 payload.
        let c = tiny_config();
        assert_eq!(
            quantized_weight_count(&c) + kept_floats(&c),
            weight_floats(&c)
        );
        // max d_in is the wider FFN dimension here (16 > d_model 8).
        assert_eq!(max_proj_d_in(&c), 16);
    }

    #[test]
    fn quantized_then_carved_views_have_the_right_shapes() {
        let c = tiny_config();
        let gs = 4; // divides d_model (8) and the ffn dims (16)
        let wbuf = fake_weights(&c);
        let fp32 = Weights::new(&wbuf, &c).unwrap();

        let mut data = vec![0i8; quantized_weight_count(&c)];
        let mut scales = vec![0.0f32; quantized_scale_count(&c, gs)];
        quantize_weights(&fp32, &mut data, &mut scales, gs, &c);
        let qw = QuantizedWeights::new(&data, &scales, gs, &c).unwrap();

        let d = c.d_model;
        assert_eq!(qw.token_embedding.data.len(), c.vocab_size * d);
        assert_eq!(qw.enc_wq.data.len(), c.enc_layers * d * d);
        assert_eq!(qw.enc_fc1_w.data.len(), c.enc_layers * c.enc_ffn * d);
        assert_eq!(qw.dec_cross_wo.data.len(), c.dec_layers * d * d);
        assert_eq!(qw.dec_fc2_w.data.len(), c.dec_layers * d * c.dec_ffn);
        // Per-layer slice of a multi-layer projection lands on the right block.
        let l1 = qw.enc_wq.layer(1, d, d);
        assert_eq!(l1.data, &qw.enc_wq.data[d * d..2 * d * d]);
    }

    #[test]
    fn packed_kept_roundtrips_to_the_borrowed_view() {
        // pack_kept + carve must reproduce exactly what from_weights borrows from the file,
        // proving the pack order, carve order, and struct field order all agree.
        let c = tiny_config();
        let wbuf = fake_weights(&c);
        let fp32 = Weights::new(&wbuf, &c).unwrap();

        let mut packed = vec![0.0f32; kept_floats(&c)];
        pack_kept(&fp32, &mut packed, &c);
        let carved = KeptWeights::carve(&packed, &c).unwrap();
        let borrowed = KeptWeights::from_weights(&fp32);
        assert_eq!(carved, borrowed);
        // Spot-check a couple of fields against the original tensors directly.
        assert_eq!(carved.enc_bq, fp32.enc_bq);
        assert_eq!(carved.final_logits_bias, fp32.final_logits_bias);
    }

    #[test]
    fn new_rejects_short_buffers() {
        let c = tiny_config();
        let gs = 4;
        let data = vec![0i8; quantized_weight_count(&c) - 1];
        let scales = vec![0.0f32; quantized_scale_count(&c, gs)];
        assert!(matches!(
            QuantizedWeights::new(&data, &scales, gs, &c),
            Err(EngineError::SizeMismatch { .. })
        ));
        let short = vec![0.0f32; kept_floats(&c) - 1];
        assert!(matches!(
            KeptWeights::carve(&short, &c),
            Err(EngineError::SizeMismatch { .. })
        ));
    }
}
