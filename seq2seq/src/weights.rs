//! Zero-copy views into a seq2seq checkpoint's `f32` weight region.
//!
//! After the [`Config`] header, a seq2seq checkpoint stores every tensor
//! as raw little-endian `f32`, back to back, in the fixed order below —
//! `scripts/export_marian.py` writes it and [`Weights::new`] carves it.
//! As on the Llama path, each named tensor is stored **flat across layers**
//! (layer `l`'s rows begin at `l * rows * cols`), and nothing is copied or
//! reshaped: every field borrows straight out of the host's file buffer.
//!
//! Matrices are row-major `[d_out, d_in]`, matching both the engine's
//! [`matmul`](engine::math::matmul) convention and PyTorch `nn.Linear` storage,
//! so the export script dumps Hugging Face tensors unmodified. Sinusoidal
//! position embeddings are **not stored** — they are a pure function of
//! `(pos, d_model)` and are computed at run time, exactly as the Llama path
//! computes RoPE.
//!
//! On-disk order (`d` = `d_model`, `Le`/`Ld` = encoder/decoder layers,
//! `Fe`/`Fd` = encoder/decoder ffn widths, `V` = vocab):
//!
//! | tensor | shape | role |
//! |--------|-------|------|
//! | `token_embedding` | `[V, d]` | shared: encoder input, decoder input, tied lm_head |
//! | `enc_wq`,`enc_bq` … `enc_wo`,`enc_bo` | `[Le,d,d]` + `[Le,d]` each | encoder self-attention projections |
//! | `enc_ln_att_w`,`enc_ln_att_b` | `[Le,d]` each | LayerNorm of the self-attention sublayer |
//! | `enc_fc1_w`,`enc_fc1_b` | `[Le,Fe,d]`, `[Le,Fe]` | encoder FFN up projection |
//! | `enc_fc2_w`,`enc_fc2_b` | `[Le,d,Fe]`, `[Le,d]` | encoder FFN down projection |
//! | `enc_ln_ffn_w`,`enc_ln_ffn_b` | `[Le,d]` each | LayerNorm of the FFN sublayer |
//! | `dec_wq`,`dec_bq` … `dec_wo`,`dec_bo` | `[Ld,d,d]` + `[Ld,d]` each | decoder (causal) self-attention |
//! | `dec_ln_self_w`,`dec_ln_self_b` | `[Ld,d]` each | LayerNorm of the self-attention sublayer |
//! | `dec_cross_wq`,`dec_cross_bq` … `dec_cross_wo`,`dec_cross_bo` | `[Ld,d,d]` + `[Ld,d]` each | cross-attention (Q from decoder, K/V from encoder output) |
//! | `dec_ln_cross_w`,`dec_ln_cross_b` | `[Ld,d]` each | LayerNorm of the cross-attention sublayer |
//! | `dec_fc1_w`,`dec_fc1_b` | `[Ld,Fd,d]`, `[Ld,Fd]` | decoder FFN up projection |
//! | `dec_fc2_w`,`dec_fc2_b` | `[Ld,d,Fd]`, `[Ld,d]` | decoder FFN down projection |
//! | `dec_ln_ffn_w`,`dec_ln_ffn_b` | `[Ld,d]` each | LayerNorm of the FFN sublayer |
//! | `final_logits_bias` | `[V]` | added to the tied lm_head's logits |
//!
//! (Whether each LayerNorm runs before or after its sublayer is the
//! [`norm_before`](Config::norm_before) flag's business, not the file
//! layout's; the storage order above is fixed either way.)

use engine::error::EngineError;
use crate::config::{Config, SEQ2SEQ_HEADER_BYTES};

/// Number of `f32` elements a seq2seq checkpoint stores after its header.
pub const fn weight_floats(c: &Config) -> usize {
    let d = c.d_model;
    // One attention block: four d×d projections with their four d-vector biases,
    // plus the sublayer's LayerNorm weight and bias.
    let att_block = 4 * d * d + 4 * d + 2 * d;
    // One FFN block: fc1 [ffn,d]+[ffn], fc2 [d,ffn]+[d], plus its LayerNorm.
    let enc_ffn_block = c.enc_ffn * d + c.enc_ffn + d * c.enc_ffn + d + 2 * d;
    let dec_ffn_block = c.dec_ffn * d + c.dec_ffn + d * c.dec_ffn + d + 2 * d;

    c.vocab_size * d // shared token embedding (encoder, decoder, tied lm_head)
        + c.enc_layers * (att_block + enc_ffn_block)
        + c.dec_layers * (2 * att_block + dec_ffn_block) // self-attn + cross-attn
        + c.vocab_size // final_logits_bias
}

/// Total on-disk size of a seq2seq checkpoint with this config, in bytes:
/// the fixed header plus the fp32 weight payload.
pub const fn expected_file_bytes(c: &Config) -> usize {
    SEQ2SEQ_HEADER_BYTES + weight_floats(c) * core::mem::size_of::<f32>()
}

/// Borrowed, zero-copy views of every seq2seq weight tensor.
///
/// Each field is a flat slice over all layers; per-layer indexing is the forward
/// pass's responsibility. Shapes are given as `[layers, rows, cols]` /
/// `[layers, len]` in the field docs, with `d = d_model`.
#[derive(Debug, Clone, Copy)]
pub struct Weights<'a> {
    /// Shared token embedding table, `[vocab, d]` — encoder input, decoder
    /// input, and (tied) lm_head all read this one matrix.
    pub token_embedding: &'a [f32],

    // --- encoder self-attention ---
    /// Encoder query projection, `[enc_layers, d, d]`.
    pub enc_wq: &'a [f32],
    /// Encoder query bias, `[enc_layers, d]`.
    pub enc_bq: &'a [f32],
    /// Encoder key projection, `[enc_layers, d, d]`.
    pub enc_wk: &'a [f32],
    /// Encoder key bias, `[enc_layers, d]`.
    pub enc_bk: &'a [f32],
    /// Encoder value projection, `[enc_layers, d, d]`.
    pub enc_wv: &'a [f32],
    /// Encoder value bias, `[enc_layers, d]`.
    pub enc_bv: &'a [f32],
    /// Encoder attention output projection, `[enc_layers, d, d]`.
    pub enc_wo: &'a [f32],
    /// Encoder attention output bias, `[enc_layers, d]`.
    pub enc_bo: &'a [f32],
    /// Encoder self-attention LayerNorm weight, `[enc_layers, d]`.
    pub enc_ln_att_w: &'a [f32],
    /// Encoder self-attention LayerNorm bias, `[enc_layers, d]`.
    pub enc_ln_att_b: &'a [f32],

    // --- encoder feed-forward ---
    /// Encoder FFN up projection, `[enc_layers, enc_ffn, d]`.
    pub enc_fc1_w: &'a [f32],
    /// Encoder FFN up bias, `[enc_layers, enc_ffn]`.
    pub enc_fc1_b: &'a [f32],
    /// Encoder FFN down projection, `[enc_layers, d, enc_ffn]`.
    pub enc_fc2_w: &'a [f32],
    /// Encoder FFN down bias, `[enc_layers, d]`.
    pub enc_fc2_b: &'a [f32],
    /// Encoder FFN LayerNorm weight, `[enc_layers, d]`.
    pub enc_ln_ffn_w: &'a [f32],
    /// Encoder FFN LayerNorm bias, `[enc_layers, d]`.
    pub enc_ln_ffn_b: &'a [f32],

    // --- decoder (causal) self-attention ---
    /// Decoder self-attention query projection, `[dec_layers, d, d]`.
    pub dec_wq: &'a [f32],
    /// Decoder self-attention query bias, `[dec_layers, d]`.
    pub dec_bq: &'a [f32],
    /// Decoder self-attention key projection, `[dec_layers, d, d]`.
    pub dec_wk: &'a [f32],
    /// Decoder self-attention key bias, `[dec_layers, d]`.
    pub dec_bk: &'a [f32],
    /// Decoder self-attention value projection, `[dec_layers, d, d]`.
    pub dec_wv: &'a [f32],
    /// Decoder self-attention value bias, `[dec_layers, d]`.
    pub dec_bv: &'a [f32],
    /// Decoder self-attention output projection, `[dec_layers, d, d]`.
    pub dec_wo: &'a [f32],
    /// Decoder self-attention output bias, `[dec_layers, d]`.
    pub dec_bo: &'a [f32],
    /// Decoder self-attention LayerNorm weight, `[dec_layers, d]`.
    pub dec_ln_self_w: &'a [f32],
    /// Decoder self-attention LayerNorm bias, `[dec_layers, d]`.
    pub dec_ln_self_b: &'a [f32],

    // --- decoder cross-attention (Q from decoder, K/V from encoder output) ---
    /// Cross-attention query projection, `[dec_layers, d, d]`.
    pub dec_cross_wq: &'a [f32],
    /// Cross-attention query bias, `[dec_layers, d]`.
    pub dec_cross_bq: &'a [f32],
    /// Cross-attention key projection, `[dec_layers, d, d]`.
    pub dec_cross_wk: &'a [f32],
    /// Cross-attention key bias, `[dec_layers, d]`.
    pub dec_cross_bk: &'a [f32],
    /// Cross-attention value projection, `[dec_layers, d, d]`.
    pub dec_cross_wv: &'a [f32],
    /// Cross-attention value bias, `[dec_layers, d]`.
    pub dec_cross_bv: &'a [f32],
    /// Cross-attention output projection, `[dec_layers, d, d]`.
    pub dec_cross_wo: &'a [f32],
    /// Cross-attention output bias, `[dec_layers, d]`.
    pub dec_cross_bo: &'a [f32],
    /// Cross-attention LayerNorm weight, `[dec_layers, d]`.
    pub dec_ln_cross_w: &'a [f32],
    /// Cross-attention LayerNorm bias, `[dec_layers, d]`.
    pub dec_ln_cross_b: &'a [f32],

    // --- decoder feed-forward ---
    /// Decoder FFN up projection, `[dec_layers, dec_ffn, d]`.
    pub dec_fc1_w: &'a [f32],
    /// Decoder FFN up bias, `[dec_layers, dec_ffn]`.
    pub dec_fc1_b: &'a [f32],
    /// Decoder FFN down projection, `[dec_layers, d, dec_ffn]`.
    pub dec_fc2_w: &'a [f32],
    /// Decoder FFN down bias, `[dec_layers, d]`.
    pub dec_fc2_b: &'a [f32],
    /// Decoder FFN LayerNorm weight, `[dec_layers, d]`.
    pub dec_ln_ffn_w: &'a [f32],
    /// Decoder FFN LayerNorm bias, `[dec_layers, d]`.
    pub dec_ln_ffn_b: &'a [f32],

    /// Bias added to the tied lm_head's logits, `[vocab]`.
    pub final_logits_bias: &'a [f32],
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

        let d = c.d_model;
        let (le, ld) = (c.enc_layers, c.dec_layers);

        // Bump a cursor through the slice, taking each tensor in file order.
        let mut rest = floats;
        let mut take = |n: usize| -> &'a [f32] {
            let (head, tail) = rest.split_at(n);
            rest = tail;
            head
        };

        Ok(Weights {
            token_embedding: take(c.vocab_size * d),

            enc_wq: take(le * d * d),
            enc_bq: take(le * d),
            enc_wk: take(le * d * d),
            enc_bk: take(le * d),
            enc_wv: take(le * d * d),
            enc_bv: take(le * d),
            enc_wo: take(le * d * d),
            enc_bo: take(le * d),
            enc_ln_att_w: take(le * d),
            enc_ln_att_b: take(le * d),
            enc_fc1_w: take(le * c.enc_ffn * d),
            enc_fc1_b: take(le * c.enc_ffn),
            enc_fc2_w: take(le * d * c.enc_ffn),
            enc_fc2_b: take(le * d),
            enc_ln_ffn_w: take(le * d),
            enc_ln_ffn_b: take(le * d),

            dec_wq: take(ld * d * d),
            dec_bq: take(ld * d),
            dec_wk: take(ld * d * d),
            dec_bk: take(ld * d),
            dec_wv: take(ld * d * d),
            dec_bv: take(ld * d),
            dec_wo: take(ld * d * d),
            dec_bo: take(ld * d),
            dec_ln_self_w: take(ld * d),
            dec_ln_self_b: take(ld * d),

            dec_cross_wq: take(ld * d * d),
            dec_cross_bq: take(ld * d),
            dec_cross_wk: take(ld * d * d),
            dec_cross_bk: take(ld * d),
            dec_cross_wv: take(ld * d * d),
            dec_cross_bv: take(ld * d),
            dec_cross_wo: take(ld * d * d),
            dec_cross_bo: take(ld * d),
            dec_ln_cross_w: take(ld * d),
            dec_ln_cross_b: take(ld * d),

            dec_fc1_w: take(ld * c.dec_ffn * d),
            dec_fc1_b: take(ld * c.dec_ffn),
            dec_fc2_w: take(ld * d * c.dec_ffn),
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
    fn weight_floats_matches_hand_sum() {
        let c = tiny_config();
        // d=4: att block = 4*16 + 4*4 + 2*4 = 88.
        // ffn block (ffn=8) = 8*4 + 8 + 4*8 + 4 + 2*4 = 84.
        // enc layer = 88 + 84 = 172; dec layer = 2*88 + 84 = 260.
        // emb 10*4=40, final_logits_bias 10.
        let expect = 40 + 172 + 260 + 10;
        assert_eq!(weight_floats(&c), expect);
        assert_eq!(expected_file_bytes(&c), SEQ2SEQ_HEADER_BYTES + expect * 4);
    }

    #[test]
    fn views_carve_tensors_at_their_disk_offsets() {
        // Fill the buffer with its own indices so each slice's first element
        // reveals the offset it was carved from — proving the disk order.
        let c = tiny_config();
        let buf: std::vec::Vec<f32> = (0..weight_floats(&c)).map(|i| i as f32).collect();
        let w = Weights::new(&buf, &c).unwrap();

        assert_eq!(w.token_embedding[0], 0.0);
        assert_eq!(w.token_embedding.len(), 40);
        // Encoder: emb(40) | wq 16 | bq 4 | wk 16 | bk 4 | wv 16 | bv 4 | wo 16 |
        // bo 4 | ln_att 4+4 | fc1 32+8 | fc2 32+4 | ln_ffn 4+4.
        assert_eq!(w.enc_wq[0], 40.0);
        assert_eq!(w.enc_bq[0], 56.0);
        assert_eq!(w.enc_wk[0], 60.0);
        assert_eq!(w.enc_wo[0], 100.0);
        assert_eq!(w.enc_ln_att_w[0], 120.0);
        assert_eq!(w.enc_fc1_w[0], 128.0);
        assert_eq!(w.enc_fc1_b[0], 160.0);
        assert_eq!(w.enc_fc2_w[0], 168.0);
        assert_eq!(w.enc_ln_ffn_b[0], 208.0);
        // Decoder starts right after the encoder block (offset 212).
        assert_eq!(w.dec_wq[0], 212.0);
        assert_eq!(w.dec_ln_self_w[0], 292.0);
        assert_eq!(w.dec_cross_wq[0], 300.0);
        assert_eq!(w.dec_ln_cross_b[0], 384.0);
        assert_eq!(w.dec_fc1_w[0], 388.0);
        assert_eq!(w.dec_ln_ffn_b[0], 468.0);
        // The logit bias is the file's tail; the carve must land exactly at EOF.
        assert_eq!(w.final_logits_bias[0], 472.0);
        assert_eq!(w.final_logits_bias.len(), c.vocab_size);
        assert_eq!(472 + c.vocab_size, weight_floats(&c));
    }

    #[test]
    fn short_slice_is_rejected() {
        let c = tiny_config();
        let buf = std::vec![0.0f32; weight_floats(&c) - 1];
        assert!(matches!(
            Weights::new(&buf, &c),
            Err(EngineError::SizeMismatch { .. })
        ));
    }

    #[test]
    fn weight_floats_is_const_evaluable() {
        // Like the Llama path's `weight_floats`, the size math is `const fn` so a
        // bare-metal host can size a static weight buffer at compile time; forcing
        // the evaluation in a const context guards that against regressing.
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
        const FLOATS: usize = weight_floats(&C);
        const BYTES: usize = expected_file_bytes(&C);
        const _: () = assert!(BYTES == SEQ2SEQ_HEADER_BYTES + FLOATS * 4);
        assert_eq!(FLOATS, 482);
    }

    // Tiny test-only heap helper; the engine itself never allocates.
    extern crate std;
}
