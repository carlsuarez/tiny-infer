//! Engine error type.
//!
//! Small, `Copy`, `no_std` enum. The host maps these into richer, user-facing
//! diagnostics; the engine itself never formats strings or allocates.

use core::fmt;

/// Errors produced by the engine core.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum EngineError {
    /// The checkpoint header was shorter than its config block (28 bytes for the
    /// legacy format, 256 for the versioned v1/v2 formats).
    HeaderTooShort,
    /// The checkpoint carries the versioned magic but declares a format version
    /// this engine does not support (supported: 1 and 2).
    UnsupportedVersion {
        /// The version field found after the magic.
        version: i32,
    },
    /// A config field held a value the engine cannot use (e.g. zero or
    /// non-divisible `dim`/`n_heads`).
    InvalidConfig(ConfigField),
    /// The checkpoint's actual byte length did not match the length computed
    /// from its config. Carries `(expected, actual)` in bytes.
    SizeMismatch {
        /// Length the config says the file should be.
        expected: usize,
        /// Length actually provided.
        actual: usize,
    },
    /// The provided arena did not have room for a requested allocation.
    ArenaOverflow {
        /// Bytes requested (after alignment padding).
        requested: usize,
        /// Bytes still free in the arena.
        available: usize,
    },
    /// The file does not begin with the seq2seq checkpoint magic (`"tis2"`), so
    /// it is not a tiny-infer seq2seq checkpoint.
    #[cfg(feature = "seq2seq")]
    NotSeq2Seq,
    /// A seq2seq config field held a value the engine cannot use (e.g. zero, a
    /// special-token id outside the vocabulary, or a head count that does not
    /// divide `d_model`).
    #[cfg(feature = "seq2seq")]
    InvalidSeq2SeqConfig(Seq2SeqField),
}

/// Identifies which [`Config`](crate::Config) field failed validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ConfigField {
    /// `dim` was zero, or not divisible by `n_heads`.
    Dim,
    /// `hidden_dim` was zero.
    HiddenDim,
    /// `n_layers` was zero.
    NLayers,
    /// `n_heads` was zero.
    NHeads,
    /// `n_kv_heads` was zero, or did not divide `n_heads`.
    NKvHeads,
    /// `vocab_size` was zero (or negative in a v1/v2 header, where the sign
    /// carries no meaning).
    VocabSize,
    /// `seq_len` was zero.
    SeqLen,
    /// A v2 header's quantization `group_size` was non-positive, or did not divide
    /// `dim` and `hidden_dim` (a group must never straddle a matrix row).
    GroupSize,
}

/// Identifies which [`Seq2SeqConfig`](crate::seq2seq::Seq2SeqConfig) field
/// failed validation.
#[cfg(feature = "seq2seq")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Seq2SeqField {
    /// `d_model` was zero.
    DModel,
    /// `enc_layers` was zero.
    EncLayers,
    /// `dec_layers` was zero.
    DecLayers,
    /// `enc_heads` was zero, or did not divide `d_model`.
    EncHeads,
    /// `dec_heads` was zero, or did not divide `d_model`.
    DecHeads,
    /// `enc_ffn` was zero.
    EncFfn,
    /// `dec_ffn` was zero.
    DecFfn,
    /// `vocab_size` was zero.
    VocabSize,
    /// `max_src` was zero.
    MaxSrc,
    /// `max_tgt` was zero.
    MaxTgt,
    /// `pad_id` was negative or not below `vocab_size`.
    PadId,
    /// `eos_id` was negative or not below `vocab_size`.
    EosId,
    /// `bos_id` was negative or not below `vocab_size`.
    BosId,
    /// The activation byte named no known activation (0 = GELU, 1 = swish).
    Activation,
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EngineError::HeaderTooShort => {
                f.write_str("checkpoint is smaller than its config header")
            }
            EngineError::UnsupportedVersion { version } => write!(
                f,
                "unsupported checkpoint format version {version} (supported: 1, 2)"
            ),
            EngineError::InvalidConfig(field) => {
                write!(f, "invalid model config: bad {field:?} field")
            }
            EngineError::SizeMismatch { expected, actual } => write!(
                f,
                "checkpoint size mismatch: config implies {expected} bytes, file has {actual} bytes"
            ),
            EngineError::ArenaOverflow {
                requested,
                available,
            } => write!(
                f,
                "arena overflow: requested {requested} bytes, only {available} available"
            ),
            #[cfg(feature = "seq2seq")]
            EngineError::NotSeq2Seq => {
                f.write_str("checkpoint does not start with the seq2seq magic \"tis2\"")
            }
            #[cfg(feature = "seq2seq")]
            EngineError::InvalidSeq2SeqConfig(field) => {
                write!(f, "invalid seq2seq model config: bad {field:?} field")
            }
        }
    }
}

// `core::error::Error` is stable as of Rust 1.81; the engine stays no_std while
// still slotting into the host's `std::error::Error` chains via `?`.
impl core::error::Error for EngineError {}
