//! Model hyperparameters, parsed from a llama2.c checkpoint header.
//!
//! The legacy ("version 0") llama2.c checkpoint format that the TinyStories
//! checkpoints use begins with seven little-endian `i32` fields:
//!
//! ```text
//! dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
//! ```
//!
//! A *negative* `vocab_size` is llama2.c's in-band flag meaning "the classifier
//! matrix is stored separately" (unshared weights); a positive value means the
//! output projection shares the token-embedding table. We decode that sign into
//! [`Config::shared_weights`] and store the magnitude.

use crate::error::{ConfigField, EngineError};

/// Number of bytes in the on-disk config header: seven `i32` fields.
pub const HEADER_BYTES: usize = 7 * 4;

/// Parsed, validated model hyperparameters.
///
/// All counts are stored as `usize` so they compose directly into buffer-size
/// arithmetic without casts. Construct via [`Config::parse`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Config {
    /// Transformer embedding/residual-stream width.
    pub dim: usize,
    /// Inner width of the SwiGLU feed-forward block.
    pub hidden_dim: usize,
    /// Number of transformer blocks.
    pub n_layers: usize,
    /// Number of query heads.
    pub n_heads: usize,
    /// Number of key/value heads (`<= n_heads` for grouped-query attention).
    pub n_kv_heads: usize,
    /// Vocabulary size (magnitude; sign is decoded into `shared_weights`).
    pub vocab_size: usize,
    /// Maximum sequence length the checkpoint was exported for.
    pub seq_len: usize,
    /// Whether the output projection reuses the token-embedding table.
    pub shared_weights: bool,
}

impl Config {
    /// Parse and validate the config from the start of a checkpoint's bytes.
    ///
    /// Only the first [`HEADER_BYTES`] are read; trailing weight data is ignored
    /// here (see [`crate::weights`]). Returns [`EngineError::HeaderTooShort`] if
    /// `bytes` is too small, or [`EngineError::InvalidConfig`] if a field is
    /// unusable.
    pub fn parse(bytes: &[u8]) -> Result<Config, EngineError> {
        if bytes.len() < HEADER_BYTES {
            return Err(EngineError::HeaderTooShort);
        }

        // Read the seven fields as signed i32 first; only `vocab_size` is
        // legitimately allowed to be negative (the shared-weights flag).
        let raw = |i: usize| -> i32 {
            let o = i * 4;
            i32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]])
        };

        let dim = raw(0);
        let hidden_dim = raw(1);
        let n_layers = raw(2);
        let n_heads = raw(3);
        let n_kv_heads = raw(4);
        let vocab_raw = raw(5);
        let seq_len = raw(6);

        let shared_weights = vocab_raw > 0;
        let vocab_size = vocab_raw.unsigned_abs() as usize;

        // Every count must be a positive integer; coerce and check.
        let pos = |v: i32, field: ConfigField| -> Result<usize, EngineError> {
            if v > 0 {
                Ok(v as usize)
            } else {
                Err(EngineError::InvalidConfig(field))
            }
        };

        let dim = pos(dim, ConfigField::Dim)?;
        let hidden_dim = pos(hidden_dim, ConfigField::HiddenDim)?;
        let n_layers = pos(n_layers, ConfigField::NLayers)?;
        let n_heads = pos(n_heads, ConfigField::NHeads)?;
        let n_kv_heads = pos(n_kv_heads, ConfigField::NKvHeads)?;
        let seq_len = pos(seq_len, ConfigField::SeqLen)?;
        if vocab_size == 0 {
            return Err(EngineError::InvalidConfig(ConfigField::VocabSize));
        }

        // Structural invariants the attention layout relies on.
        if !dim.is_multiple_of(n_heads) {
            return Err(EngineError::InvalidConfig(ConfigField::Dim));
        }
        if !n_heads.is_multiple_of(n_kv_heads) {
            return Err(EngineError::InvalidConfig(ConfigField::NKvHeads));
        }

        Ok(Config {
            dim,
            hidden_dim,
            n_layers,
            n_heads,
            n_kv_heads,
            vocab_size,
            seq_len,
            shared_weights,
        })
    }

    /// Size of a single attention head: `dim / n_heads`.
    #[inline]
    pub const fn head_size(&self) -> usize {
        self.dim / self.n_heads
    }

    /// Combined width of the key/value projections: `n_kv_heads * head_size`.
    #[inline]
    pub const fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_size()
    }

    /// How many query heads share each key/value head (grouped-query attention).
    #[inline]
    pub const fn kv_mul(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a 28-byte header from seven raw i32 values.
    fn header(fields: [i32; 7]) -> [u8; HEADER_BYTES] {
        let mut out = [0u8; HEADER_BYTES];
        for (i, v) in fields.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        out
    }

    #[test]
    fn parses_stories15m_header() {
        // dim, hidden, layers, heads, kv_heads, vocab, seq_len
        let bytes = header([288, 768, 6, 6, 6, 32000, 256]);
        let c = Config::parse(&bytes).unwrap();
        assert_eq!(c.dim, 288);
        assert_eq!(c.hidden_dim, 768);
        assert_eq!(c.n_layers, 6);
        assert_eq!(c.n_heads, 6);
        assert_eq!(c.n_kv_heads, 6);
        assert_eq!(c.vocab_size, 32000);
        assert_eq!(c.seq_len, 256);
        assert!(c.shared_weights);
        assert_eq!(c.head_size(), 48);
        assert_eq!(c.kv_dim(), 288);
        assert_eq!(c.kv_mul(), 1);
    }

    #[test]
    fn negative_vocab_means_unshared_weights() {
        let bytes = header([288, 768, 6, 6, 6, -32000, 256]);
        let c = Config::parse(&bytes).unwrap();
        assert_eq!(c.vocab_size, 32000);
        assert!(!c.shared_weights);
    }

    #[test]
    fn grouped_query_attention_dims() {
        // 8 query heads, 2 kv heads -> kv_mul = 4.
        let bytes = header([512, 1376, 8, 8, 2, 32000, 1024]);
        let c = Config::parse(&bytes).unwrap();
        assert_eq!(c.head_size(), 64);
        assert_eq!(c.kv_dim(), 128);
        assert_eq!(c.kv_mul(), 4);
    }

    #[test]
    fn rejects_short_header() {
        assert_eq!(
            Config::parse(&[0u8; 12]).unwrap_err(),
            EngineError::HeaderTooShort
        );
    }

    #[test]
    fn rejects_zero_fields() {
        let bytes = header([0, 768, 6, 6, 6, 32000, 256]);
        assert_eq!(
            Config::parse(&bytes).unwrap_err(),
            EngineError::InvalidConfig(ConfigField::Dim)
        );
    }

    #[test]
    fn rejects_dim_not_divisible_by_heads() {
        let bytes = header([100, 768, 6, 7, 7, 32000, 256]);
        assert_eq!(
            Config::parse(&bytes).unwrap_err(),
            EngineError::InvalidConfig(ConfigField::Dim)
        );
    }

    #[test]
    fn rejects_kv_heads_not_dividing_heads() {
        let bytes = header([512, 1376, 8, 8, 3, 32000, 1024]);
        assert_eq!(
            Config::parse(&bytes).unwrap_err(),
            EngineError::InvalidConfig(ConfigField::NKvHeads)
        );
    }
}
