//! Seq2seq model hyperparameters, parsed from a tiny-infer seq2seq checkpoint
//! header.
//!
//! The format is tiny-infer's own; `scripts/export_marian.py` writes it from a
//! Hugging Face `MarianMTModel`. It deliberately mirrors the Llama **versioned**
//! header style — a magic, a version, fixed little-endian `i32` fields, flag bytes,
//! zero padding to a fixed size — but with its own magic so the two families can
//! never be confused:
//!
//! | offset | size | field |
//! |--------|------|-------|
//! | 0      | 4    | magic `"tis2"` ([`SEQ2SEQ_MAGIC`]) |
//! | 4      | 4    | version (`i32`, currently 1) |
//! | 8      | 4    | `d_model` |
//! | 12     | 4    | `enc_layers` |
//! | 16     | 4    | `dec_layers` |
//! | 20     | 4    | `enc_heads` |
//! | 24     | 4    | `dec_heads` |
//! | 28     | 4    | `enc_ffn` |
//! | 32     | 4    | `dec_ffn` |
//! | 36     | 4    | `vocab_size` |
//! | 40     | 4    | `max_src` |
//! | 44     | 4    | `max_tgt` |
//! | 48     | 4    | `pad_id` |
//! | 52     | 4    | `eos_id` |
//! | 56     | 4    | `bos_id` |
//! | 60     | 1    | `norm_before` (0 = post-norm, Marian's default) |
//! | 61     | 1    | activation (0 = GELU, 1 = swish/SiLU) |
//! | 62     | 1    | `scale_embedding` (1 = embeddings scaled by `√d_model`) |
//! | 63     | 193  | zero padding up to [`SEQ2SEQ_HEADER_BYTES`] |
//!
//! All `i32` fields must be positive except the three token ids, which must lie
//! in `0..vocab_size`. The fp32 weight payload that follows the header is
//! documented in [`crate::weights`].

use engine::error::{EngineError, Seq2SeqField};

/// Size of the seq2seq on-disk header: magic, version, fields, flags, padding.
pub const SEQ2SEQ_HEADER_BYTES: usize = 256;

/// The `u32` magic that opens a seq2seq checkpoint: the ASCII bytes `"tis2"`
/// (**t**iny-**i**nfer **s**eq**2**seq), read little-endian. Distinct from the
/// Llama family's `"ak42"` magic, so [`is_seq2seq`] can route a file to the
/// right loader from its first four bytes.
pub const SEQ2SEQ_MAGIC: u32 = u32::from_le_bytes(*b"tis2");

/// The header version this engine reads and `export_marian.py` writes.
pub const SEQ2SEQ_VERSION: i32 = 1;

/// Whether `bytes` opens with the seq2seq magic (cheap format sniff; does not
/// validate anything past the first four bytes).
pub fn is_seq2seq(bytes: &[u8]) -> bool {
    bytes.len() >= 4
        && u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) == SEQ2SEQ_MAGIC
}

/// The feed-forward activation a checkpoint was trained with.
///
/// Marian reads this from the Hugging Face config's `activation_function`;
/// OPUS-MT models typically use swish (SiLU), while other Marian exports use
/// GELU. The engine supports both and never hardcodes one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// Exact-erf GELU (`x · Φ(x)`).
    Gelu,
    /// Swish / SiLU (`x · σ(x)`), the same function the Llama path's SwiGLU uses.
    Swish,
}

/// Parsed, validated seq2seq (Marian-style encoder-decoder) hyperparameters.
///
/// The encoder and decoder stacks share `d_model` and the vocabulary (one
/// embedding table serves the encoder input, the decoder input, and the tied
/// lm_head) but may differ in depth, head count, and feed-forward width.
/// Construct via [`Config::parse`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Config {
    /// Embedding/residual-stream width, shared by both stacks.
    pub d_model: usize,
    /// Number of encoder layers.
    pub enc_layers: usize,
    /// Number of decoder layers.
    pub dec_layers: usize,
    /// Number of encoder self-attention heads (must divide `d_model`).
    pub enc_heads: usize,
    /// Number of decoder self- and cross-attention heads (must divide `d_model`).
    pub dec_heads: usize,
    /// Inner width of the encoder feed-forward block.
    pub enc_ffn: usize,
    /// Inner width of the decoder feed-forward block.
    pub dec_ffn: usize,
    /// Vocabulary size (source and target share one SentencePiece vocabulary).
    pub vocab_size: usize,
    /// Maximum source-sequence length the checkpoint supports.
    pub max_src: usize,
    /// Maximum target-sequence length the checkpoint supports.
    pub max_tgt: usize,
    /// Padding token id. Marian also uses it as the decoder start token
    /// (see [`Config::decoder_start_id`]).
    pub pad_id: usize,
    /// End-of-sequence token id (stops greedy decoding).
    pub eos_id: usize,
    /// Beginning-of-sequence token id (unused by Marian decoding, but recorded).
    pub bos_id: usize,
    /// `true` = pre-norm (`LN` before each sublayer); `false` = post-norm
    /// (`LN` after the residual add), Marian's default.
    pub norm_before: bool,
    /// Feed-forward activation function.
    pub activation: Activation,
    /// Whether token embeddings are multiplied by `√d_model` before positions
    /// are added (Marian's `scale_embedding`).
    pub scale_embedding: bool,
}

impl Config {
    /// Parse and validate a seq2seq header from the start of a checkpoint's
    /// bytes.
    ///
    /// Only the first [`SEQ2SEQ_HEADER_BYTES`] are read; the weight payload is
    /// carved separately (see [`crate::weights`]). Returns
    /// [`EngineError::NotSeq2Seq`] if the magic is absent (so a caller can fall
    /// back to other formats), [`EngineError::HeaderTooShort`] /
    /// [`EngineError::UnsupportedVersion`] for a damaged or newer file, and
    /// [`EngineError::InvalidSeq2SeqConfig`] if a field is unusable.
    pub fn parse(bytes: &[u8]) -> Result<Config, EngineError> {
        if bytes.len() < 4 {
            return Err(EngineError::HeaderTooShort);
        }
        if !is_seq2seq(bytes) {
            return Err(EngineError::NotSeq2Seq);
        }
        if bytes.len() < SEQ2SEQ_HEADER_BYTES {
            return Err(EngineError::HeaderTooShort);
        }
        let int =
            |o: usize| i32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);

        let version = int(4);
        if version != SEQ2SEQ_VERSION {
            return Err(EngineError::UnsupportedVersion { version });
        }

        // Every count must be a positive integer; the token ids are validated
        // against the vocabulary below.
        let pos = |v: i32, field: Seq2SeqField| -> Result<usize, EngineError> {
            if v > 0 {
                Ok(v as usize)
            } else {
                Err(EngineError::InvalidSeq2SeqConfig(field))
            }
        };

        let d_model = pos(int(8), Seq2SeqField::DModel)?;
        let enc_layers = pos(int(12), Seq2SeqField::EncLayers)?;
        let dec_layers = pos(int(16), Seq2SeqField::DecLayers)?;
        let enc_heads = pos(int(20), Seq2SeqField::EncHeads)?;
        let dec_heads = pos(int(24), Seq2SeqField::DecHeads)?;
        let enc_ffn = pos(int(28), Seq2SeqField::EncFfn)?;
        let dec_ffn = pos(int(32), Seq2SeqField::DecFfn)?;
        let vocab_size = pos(int(36), Seq2SeqField::VocabSize)?;
        let max_src = pos(int(40), Seq2SeqField::MaxSrc)?;
        let max_tgt = pos(int(44), Seq2SeqField::MaxTgt)?;

        // Token ids may be zero but must index into the vocabulary.
        let id = |v: i32, field: Seq2SeqField| -> Result<usize, EngineError> {
            if v >= 0 && (v as usize) < vocab_size {
                Ok(v as usize)
            } else {
                Err(EngineError::InvalidSeq2SeqConfig(field))
            }
        };
        let pad_id = id(int(48), Seq2SeqField::PadId)?;
        let eos_id = id(int(52), Seq2SeqField::EosId)?;
        let bos_id = id(int(56), Seq2SeqField::BosId)?;

        // Structural invariants the attention layout relies on.
        if !d_model.is_multiple_of(enc_heads) {
            return Err(EngineError::InvalidSeq2SeqConfig(Seq2SeqField::EncHeads));
        }
        if !d_model.is_multiple_of(dec_heads) {
            return Err(EngineError::InvalidSeq2SeqConfig(Seq2SeqField::DecHeads));
        }

        let norm_before = bytes[60] != 0;
        let activation = match bytes[61] {
            0 => Activation::Gelu,
            1 => Activation::Swish,
            _ => return Err(EngineError::InvalidSeq2SeqConfig(Seq2SeqField::Activation)),
        };
        let scale_embedding = bytes[62] != 0;

        Ok(Config {
            d_model,
            enc_layers,
            dec_layers,
            enc_heads,
            dec_heads,
            enc_ffn,
            dec_ffn,
            vocab_size,
            max_src,
            max_tgt,
            pad_id,
            eos_id,
            bos_id,
            norm_before,
            activation,
            scale_embedding,
        })
    }

    /// Size of a single encoder attention head: `d_model / enc_heads`.
    #[inline]
    pub const fn enc_head_dim(&self) -> usize {
        self.d_model / self.enc_heads
    }

    /// Size of a single decoder attention head: `d_model / dec_heads`.
    #[inline]
    pub const fn dec_head_dim(&self) -> usize {
        self.d_model / self.dec_heads
    }

    /// The token that seeds the decoder. Marian starts decoding from the padding
    /// token (Hugging Face's `decoder_start_token_id == pad_token_id`), which the
    /// export script asserts at conversion time.
    #[inline]
    pub const fn decoder_start_id(&self) -> usize {
        self.pad_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a 256-byte seq2seq header from raw fields and flag bytes.
    fn header(fields: [i32; 13], flags: [u8; 3]) -> [u8; SEQ2SEQ_HEADER_BYTES] {
        let mut out = [0u8; SEQ2SEQ_HEADER_BYTES];
        out[0..4].copy_from_slice(b"tis2");
        out[4..8].copy_from_slice(&SEQ2SEQ_VERSION.to_le_bytes());
        for (i, v) in fields.iter().enumerate() {
            out[8 + i * 4..8 + i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        out[60..63].copy_from_slice(&flags);
        out
    }

    /// The shape of Helsinki-NLP/opus-mt-en-fr: d_model, enc/dec layers, heads,
    /// ffn dims, vocab, max_src, max_tgt, pad, eos, bos.
    const OPUS_EN_FR: [i32; 13] = [512, 6, 6, 8, 8, 2048, 2048, 59514, 512, 512, 59513, 0, 0];
    /// post-norm, swish, scaled embeddings — OPUS-MT's flags.
    const OPUS_FLAGS: [u8; 3] = [0, 1, 1];

    #[test]
    fn parses_opus_mt_shaped_header() {
        let bytes = header(OPUS_EN_FR, OPUS_FLAGS);
        let c = Config::parse(&bytes).unwrap();
        assert_eq!(c.d_model, 512);
        assert_eq!(c.enc_layers, 6);
        assert_eq!(c.dec_layers, 6);
        assert_eq!(c.enc_heads, 8);
        assert_eq!(c.dec_heads, 8);
        assert_eq!(c.enc_ffn, 2048);
        assert_eq!(c.dec_ffn, 2048);
        assert_eq!(c.vocab_size, 59514);
        assert_eq!(c.max_src, 512);
        assert_eq!(c.max_tgt, 512);
        assert_eq!(c.pad_id, 59513);
        assert_eq!(c.eos_id, 0);
        assert_eq!(c.bos_id, 0);
        assert!(!c.norm_before);
        assert_eq!(c.activation, Activation::Swish);
        assert!(c.scale_embedding);
        assert_eq!(c.enc_head_dim(), 64);
        assert_eq!(c.dec_head_dim(), 64);
        // Marian seeds the decoder with the pad token.
        assert_eq!(c.decoder_start_id(), 59513);
    }

    #[test]
    fn parses_pre_norm_gelu_flags() {
        let bytes = header(OPUS_EN_FR, [1, 0, 0]);
        let c = Config::parse(&bytes).unwrap();
        assert!(c.norm_before);
        assert_eq!(c.activation, Activation::Gelu);
        assert!(!c.scale_embedding);
    }

    #[test]
    fn rejects_wrong_magic() {
        // A Llama versioned header must not parse as seq2seq.
        let mut bytes = header(OPUS_EN_FR, OPUS_FLAGS);
        bytes[0..4].copy_from_slice(&llama::config::MAGIC.to_le_bytes());
        assert_eq!(Config::parse(&bytes).unwrap_err(), EngineError::NotSeq2Seq);
        assert!(!is_seq2seq(&bytes));
        assert!(is_seq2seq(&header(OPUS_EN_FR, OPUS_FLAGS)));
    }

    #[test]
    fn rejects_short_header() {
        // Too short to even check the magic.
        assert_eq!(
            Config::parse(&[0u8; 3]).unwrap_err(),
            EngineError::HeaderTooShort
        );
        // Magic present but the header is truncated.
        let bytes = header(OPUS_EN_FR, OPUS_FLAGS);
        assert_eq!(
            Config::parse(&bytes[..200]).unwrap_err(),
            EngineError::HeaderTooShort
        );
    }

    #[test]
    fn rejects_unknown_version() {
        let mut bytes = header(OPUS_EN_FR, OPUS_FLAGS);
        bytes[4..8].copy_from_slice(&2i32.to_le_bytes());
        assert_eq!(
            Config::parse(&bytes).unwrap_err(),
            EngineError::UnsupportedVersion { version: 2 }
        );
    }

    #[test]
    fn rejects_zero_fields() {
        let mut fields = OPUS_EN_FR;
        fields[0] = 0; // d_model
        assert_eq!(
            Config::parse(&header(fields, OPUS_FLAGS)).unwrap_err(),
            EngineError::InvalidSeq2SeqConfig(Seq2SeqField::DModel)
        );
        let mut fields = OPUS_EN_FR;
        fields[6] = -1; // dec_ffn
        assert_eq!(
            Config::parse(&header(fields, OPUS_FLAGS)).unwrap_err(),
            EngineError::InvalidSeq2SeqConfig(Seq2SeqField::DecFfn)
        );
    }

    #[test]
    fn rejects_heads_not_dividing_d_model() {
        let mut fields = OPUS_EN_FR;
        fields[3] = 7; // enc_heads: 512 % 7 != 0
        assert_eq!(
            Config::parse(&header(fields, OPUS_FLAGS)).unwrap_err(),
            EngineError::InvalidSeq2SeqConfig(Seq2SeqField::EncHeads)
        );
        let mut fields = OPUS_EN_FR;
        fields[4] = 6; // dec_heads: 512 % 6 != 0
        assert_eq!(
            Config::parse(&header(fields, OPUS_FLAGS)).unwrap_err(),
            EngineError::InvalidSeq2SeqConfig(Seq2SeqField::DecHeads)
        );
    }

    #[test]
    fn rejects_token_ids_outside_vocab() {
        let mut fields = OPUS_EN_FR;
        fields[10] = 59514; // pad_id == vocab_size: one past the end
        assert_eq!(
            Config::parse(&header(fields, OPUS_FLAGS)).unwrap_err(),
            EngineError::InvalidSeq2SeqConfig(Seq2SeqField::PadId)
        );
        let mut fields = OPUS_EN_FR;
        fields[12] = -1; // bos_id negative
        assert_eq!(
            Config::parse(&header(fields, OPUS_FLAGS)).unwrap_err(),
            EngineError::InvalidSeq2SeqConfig(Seq2SeqField::BosId)
        );
    }

    #[test]
    fn rejects_unknown_activation_byte() {
        let bytes = header(OPUS_EN_FR, [0, 2, 1]);
        assert_eq!(
            Config::parse(&bytes).unwrap_err(),
            EngineError::InvalidSeq2SeqConfig(Seq2SeqField::Activation)
        );
    }
}
