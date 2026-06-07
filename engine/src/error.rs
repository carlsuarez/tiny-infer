//! Engine error type.
//!
//! Small, `Copy`, `no_std` enum. The host maps these into richer, user-facing
//! diagnostics; the engine itself never formats strings or allocates.

use core::fmt;

/// Errors produced by the engine core.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum EngineError {
    /// The checkpoint header was shorter than the 28-byte config block.
    HeaderTooShort,
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
    /// `vocab_size` was zero.
    VocabSize,
    /// `seq_len` was zero.
    SeqLen,
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EngineError::HeaderTooShort => {
                f.write_str("checkpoint is smaller than the 28-byte config header")
            }
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
        }
    }
}

// `core::error::Error` is stable as of Rust 1.81; the engine stays no_std while
// still slotting into the host's `std::error::Error` chains via `?`.
impl core::error::Error for EngineError {}
