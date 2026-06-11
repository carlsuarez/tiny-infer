//! Host-side error type, wrapping engine errors and IO with user-facing messages.

use std::fmt;
use std::path::PathBuf;

use engine::EngineError;

/// Anything that can go wrong while loading files or parsing arguments.
#[derive(Debug)]
pub enum HostError {
    /// A command-line usage problem (bad flag, missing value, …).
    Usage(String),
    /// An IO failure, annotated with the path it happened on.
    Io { path: PathBuf, source: std::io::Error },
    /// A checkpoint whose byte length is not a whole number of `f32`.
    NotF32Aligned { path: PathBuf, len: u64 },
    /// An error bubbled up from the engine core (bad header, size mismatch, …).
    Engine { path: PathBuf, source: EngineError },
    /// A checkpoint whose (valid) format does not fit the requested operation,
    /// e.g. asking for an fp32 view of an int8 (v2) file.
    Format { path: PathBuf, msg: String },
    /// A malformed tokenizer file.
    Tokenizer(String),
}

impl HostError {
    /// Build an [`HostError::Io`] tagged with the offending path.
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        HostError::Io {
            path: path.into(),
            source,
        }
    }

    /// Build an [`HostError::Engine`] tagged with the offending path.
    pub fn engine(path: impl Into<PathBuf>, source: EngineError) -> Self {
        HostError::Engine {
            path: path.into(),
            source,
        }
    }
}

impl fmt::Display for HostError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HostError::Usage(msg) => write!(f, "{msg}"),
            HostError::Io { path, source } => {
                write!(f, "{}: {source}", path.display())
            }
            HostError::NotF32Aligned { path, len } => write!(
                f,
                "{}: file length {len} is not a multiple of 4 bytes, so it is not a valid fp32 checkpoint",
                path.display()
            ),
            HostError::Engine { path, source } => {
                write!(f, "{}: {source}", path.display())
            }
            HostError::Format { path, msg } => {
                write!(f, "{}: {msg}", path.display())
            }
            HostError::Tokenizer(msg) => write!(f, "tokenizer: {msg}"),
        }
    }
}

impl std::error::Error for HostError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            HostError::Io { source, .. } => Some(source),
            HostError::Engine { source, .. } => Some(source),
            _ => None,
        }
    }
}
