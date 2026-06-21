//! Model hyperparameters, parsed from a checkpoint header.
//!
//! There are three supported checkpoint formats; [`parse_header`] detects which one a
//! file uses and parses all of them:
//!
//! * **Legacy ("version 0")** — what the TinyStories checkpoints use. Seven
//!   little-endian `i32` fields and nothing else:
//!
//!   ```text
//!   dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
//!   ```
//!
//!   A *negative* `vocab_size` is the in-band flag meaning "the classifier matrix
//!   is stored separately" (unshared weights); positive means the output
//!   projection shares the token-embedding table. The sign is decoded into
//!   [`Config::shared_weights`] and the magnitude stored.
//!
//! * **v1 / v2 (versioned)** — a 256-byte header: the [`MAGIC`] `u32` (`"ak42"`),
//!   an `i32` version, the same seven `i32` fields (with `vocab_size` always
//!   positive), a `u8` shared-classifier flag, and — v2 only — the `i32`
//!   quantization group size; the rest is zero padding. v1 stores fp32 weights
//!   (in a different order from legacy, see [`crate::weights`]); v2 stores the
//!   matmul weights pre-quantized to int8 (group-wise symmetric Q8_0 format).
//!
//! The detected variant is reported as a [`ModelFormat`], which the loader uses to
//! pick the weight layout and validate the file size.

use engine::error::{ConfigField, EngineError};

/// Number of bytes in the legacy on-disk config header: seven `i32` fields.
pub const HEADER_BYTES: usize = 7 * 4;

/// Number of bytes in the versioned (v1/v2) header: magic, version, the seven
/// config fields, the flags, then zero padding up to a fixed 256.
pub const VERSIONED_HEADER_BYTES: usize = 256;

/// The `u32` magic ("ak42" in ASCII) that opens a versioned (v1/v2) checkpoint.
/// Legacy files start directly with `dim`, which is never remotely this large.
pub const MAGIC: u32 = 0x616b_3432;

/// Whether `bytes` look like a Llama checkpoint of any supported format — the
/// counterpart of `seq2seq::config::is_seq2seq` for routing a file to this
/// architecture.
///
/// Two positive signals:
/// * a versioned (v1/v2) file opens with the [`MAGIC`] `"ak42"`;
/// * a **legacy** ("version 0") file carries *no* magic, so the only signal is that
///   its seven-`i32` header parses as a valid [`Config`].
///
/// Because the legacy format is magic-less, this is a **fallback** predicate: it
/// returns `true` for any file whose first [`HEADER_BYTES`] happen to form a plausible
/// llama header — *including* another architecture's checkpoint (a seq2seq `"tis2"`
/// file, for instance, parses as a nonsense-but-structurally-valid legacy config). A
/// dispatcher must therefore test the magic-bearing formats (seq2seq, and any future
/// architecture) **before** falling back to `is_llama`; only the magic-less legacy
/// format legitimately has nothing else to match on.
pub fn is_llama(bytes: &[u8]) -> bool {
    // Versioned formats are positively identified by the magic alone (the loader
    // validates the rest); the magic is also far larger than any real `dim`, so it
    // can never be confused with a legacy header.
    if bytes.len() >= 4 && u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) == MAGIC {
        return true;
    }
    // Legacy: no magic to check, so accept it iff the raw header is a valid config.
    Config::parse(bytes).is_ok()
}

/// Which on-disk checkpoint format a file uses, as detected by [`parse_header`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    /// Legacy "version 0": 28-byte header, fp32 weights with the `freq_cis`
    /// tables in the middle, sharing signaled by a negative `vocab_size`.
    Legacy,
    /// Version 1: 256-byte header, fp32 weights with the norms stored first and
    /// no `freq_cis` tables.
    V1,
    /// Version 2: 256-byte header, RMSNorm gains in fp32 followed by every matmul
    /// weight pre-quantized to group-wise int8 (Q8_0).
    V2 {
        /// Quantization group size from the header (one `f32` scale per group).
        group_size: usize,
    },
}

impl ModelFormat {
    /// Size of this format's header — the file offset where weight data begins.
    pub const fn header_bytes(self) -> usize {
        match self {
            ModelFormat::Legacy => HEADER_BYTES,
            ModelFormat::V1 | ModelFormat::V2 { .. } => VERSIONED_HEADER_BYTES,
        }
    }
}

/// Parse any supported checkpoint header: legacy, v1, or v2.
///
/// Files opening with [`MAGIC`] are parsed as versioned (the `i32` version after
/// the magic selects v1 or v2; anything else is
/// [`EngineError::UnsupportedVersion`]); everything else is treated as legacy and
/// handed to [`Config::parse`]. Returns the validated config plus the detected
/// [`ModelFormat`] so the caller knows the weight layout and header size.
pub fn parse_header(bytes: &[u8]) -> Result<(Config, ModelFormat), EngineError> {
    if bytes.len() >= 4 && u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) == MAGIC {
        return parse_versioned(bytes);
    }
    Config::parse(bytes).map(|c| (c, ModelFormat::Legacy))
}

/// Parse a versioned (v1/v2) 256-byte header (the magic has already matched).
fn parse_versioned(bytes: &[u8]) -> Result<(Config, ModelFormat), EngineError> {
    if bytes.len() < VERSIONED_HEADER_BYTES {
        return Err(EngineError::HeaderTooShort);
    }
    let int = |o: usize| i32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);

    let version = int(4);
    if version != 1 && version != 2 {
        return Err(EngineError::UnsupportedVersion { version });
    }

    // The seven config fields follow the magic + version. Unlike legacy,
    // `vocab_size` carries no sign trick here (a negative value is just invalid);
    // sharing comes from the explicit flag byte after the fields.
    let mut fields = [0i32; 7];
    for (i, f) in fields.iter_mut().enumerate() {
        *f = int(8 + i * 4);
    }
    let shared_weights = bytes[36] != 0;
    let config = Config::from_fields(fields, shared_weights)?;

    if version == 1 {
        return Ok((config, ModelFormat::V1));
    }

    // v2 appends the quantization group size. It must be positive and divide both
    // matmul input widths (`dim` and `hidden_dim`) so a group never straddles a
    // row — the same constraint the runtime quantizer enforces.
    let group_size = int(37);
    if group_size <= 0 {
        return Err(EngineError::InvalidConfig(ConfigField::GroupSize));
    }
    let group_size = group_size as usize;
    if !config.dim.is_multiple_of(group_size) || !config.hidden_dim.is_multiple_of(group_size) {
        return Err(EngineError::InvalidConfig(ConfigField::GroupSize));
    }

    Ok((config, ModelFormat::V2 { group_size }))
}

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
    /// Parse and validate a **legacy** (version 0) config from the start of a
    /// checkpoint's bytes. Format-agnostic callers should use [`parse_header`],
    /// which detects the versioned formats and falls back to this.
    ///
    /// Only the first [`HEADER_BYTES`] are read; trailing weight data is ignored
    /// here (see [`crate::weights`]). Returns [`EngineError::HeaderTooShort`] if
    /// `bytes` is too small, or [`EngineError::InvalidConfig`] if a field is
    /// unusable.
    pub fn parse(bytes: &[u8]) -> Result<Config, EngineError> {
        if bytes.len() < HEADER_BYTES {
            return Err(EngineError::HeaderTooShort);
        }

        // Read the seven fields as signed i32; only `vocab_size` is legitimately
        // allowed to be negative (the legacy shared-weights flag), so decode its
        // sign here and validate the magnitude with everything else.
        let mut fields = [0i32; 7];
        for (i, f) in fields.iter_mut().enumerate() {
            let o = i * 4;
            *f = i32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
        }
        let shared_weights = fields[5] > 0;
        fields[5] = fields[5].saturating_abs();
        Config::from_fields(fields, shared_weights)
    }

    /// Validate seven raw header fields (in on-disk order) into a [`Config`].
    ///
    /// Shared by the legacy and versioned parsers, which differ only in where the
    /// fields sit and how `shared_weights` is signaled.
    fn from_fields(fields: [i32; 7], shared_weights: bool) -> Result<Config, EngineError> {
        let [dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len] = fields;

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
        let vocab_size = pos(vocab_size, ConfigField::VocabSize)?;
        let seq_len = pos(seq_len, ConfigField::SeqLen)?;

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

    /// Build a 256-byte versioned (v1/v2) header.
    fn versioned_header(
        version: i32,
        fields: [i32; 7],
        shared: u8,
        group_size: Option<i32>,
    ) -> [u8; VERSIONED_HEADER_BYTES] {
        let mut out = [0u8; VERSIONED_HEADER_BYTES];
        out[0..4].copy_from_slice(&MAGIC.to_le_bytes());
        out[4..8].copy_from_slice(&version.to_le_bytes());
        for (i, v) in fields.iter().enumerate() {
            out[8 + i * 4..8 + i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        out[36] = shared;
        if let Some(gs) = group_size {
            out[37..41].copy_from_slice(&gs.to_le_bytes());
        }
        out
    }

    const STORIES15M: [i32; 7] = [288, 768, 6, 6, 6, 32000, 256];

    #[test]
    fn parse_header_detects_legacy() {
        let bytes = header(STORIES15M);
        let (c, fmt) = parse_header(&bytes).unwrap();
        assert_eq!(fmt, ModelFormat::Legacy);
        assert_eq!(fmt.header_bytes(), HEADER_BYTES);
        assert_eq!(c, Config::parse(&bytes).unwrap());
    }

    #[test]
    fn parse_header_reads_v1() {
        let bytes = versioned_header(1, STORIES15M, 1, None);
        let (c, fmt) = parse_header(&bytes).unwrap();
        assert_eq!(fmt, ModelFormat::V1);
        assert_eq!(fmt.header_bytes(), VERSIONED_HEADER_BYTES);
        assert_eq!(c.dim, 288);
        assert_eq!(c.vocab_size, 32000);
        assert!(c.shared_weights);

        // Flag byte 0 → unshared, even though vocab_size is positive (no legacy
        // sign trick in the versioned formats).
        let bytes = versioned_header(1, STORIES15M, 0, None);
        let (c, _) = parse_header(&bytes).unwrap();
        assert!(!c.shared_weights);
    }

    #[test]
    fn parse_header_reads_v2_group_size() {
        let bytes = versioned_header(2, STORIES15M, 1, Some(32));
        let (c, fmt) = parse_header(&bytes).unwrap();
        assert_eq!(fmt, ModelFormat::V2 { group_size: 32 });
        assert!(c.shared_weights);
    }

    #[test]
    fn parse_header_rejects_unknown_version() {
        let bytes = versioned_header(3, STORIES15M, 1, None);
        assert_eq!(
            parse_header(&bytes).unwrap_err(),
            EngineError::UnsupportedVersion { version: 3 }
        );
    }

    #[test]
    fn parse_header_rejects_truncated_versioned_header() {
        let bytes = versioned_header(1, STORIES15M, 1, None);
        assert_eq!(
            parse_header(&bytes[..200]).unwrap_err(),
            EngineError::HeaderTooShort
        );
    }

    #[test]
    fn parse_header_rejects_negative_vocab_in_versioned() {
        // No sign trick in v1/v2: a negative vocab_size is simply invalid.
        let bytes = versioned_header(1, [288, 768, 6, 6, 6, -32000, 256], 1, None);
        assert_eq!(
            parse_header(&bytes).unwrap_err(),
            EngineError::InvalidConfig(ConfigField::VocabSize)
        );
    }

    #[test]
    fn is_llama_accepts_legacy_and_versioned() {
        // Legacy (no magic), v1, and v2 are all recognized.
        assert!(is_llama(&header(STORIES15M)));
        assert!(is_llama(&versioned_header(1, STORIES15M, 1, None)));
        assert!(is_llama(&versioned_header(2, STORIES15M, 1, Some(32))));
    }

    #[test]
    fn is_llama_rejects_nonsense_and_short_headers() {
        // All-zero fields: dim = 0 fails the legacy parse.
        assert!(!is_llama(&[0u8; HEADER_BYTES]));
        // Too short to even hold the seven-int legacy header.
        assert!(!is_llama(&[0u8; 12]));
    }

    #[test]
    fn is_llama_versioned_needs_the_exact_ak42_magic() {
        // A foreign magic is not a versioned llama file. With the rest of the header
        // zeroed it also fails the legacy fallback (hidden_dim = 0), so it is rejected
        // outright — but note a *fully formed* foreign header (e.g. a real seq2seq one)
        // would pass the magic-less legacy parse, which is why magic-bearing formats
        // must be routed first. See [`is_llama`].
        let mut bytes = [0u8; HEADER_BYTES];
        bytes[..4].copy_from_slice(b"tis2");
        assert!(!is_llama(&bytes));
    }

    #[test]
    fn parse_header_rejects_bad_v2_group_size() {
        // 0 and negative are invalid outright.
        for gs in [0, -32] {
            let bytes = versioned_header(2, STORIES15M, 1, Some(gs));
            assert_eq!(
                parse_header(&bytes).unwrap_err(),
                EngineError::InvalidConfig(ConfigField::GroupSize)
            );
        }
        // 64 divides hidden_dim (768) but not dim (288): groups would straddle rows.
        let bytes = versioned_header(2, STORIES15M, 1, Some(64));
        assert_eq!(
            parse_header(&bytes).unwrap_err(),
            EngineError::InvalidConfig(ConfigField::GroupSize)
        );
    }
}
