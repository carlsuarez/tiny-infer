//! Seq2seq (Marian / OPUS-MT) checkpoint loading — the sibling of [`crate::loader`]
//! for tiny-infer's own seq2seq format (magic `"tis2"`, written by
//! `scripts/export_marian.py`).
//!
//! The mechanics mirror [`Model`](crate::loader::Model): the file is read once
//! into a 4-aligned `Vec<f32>` (so the weight views borrow straight from it with
//! no copy), the header is parsed and validated by the engine, and the file
//! length is checked against what the config implies. The format is detected up
//! front by [`is_seq2seq_file`], which sniffs the four magic bytes so the CLI can
//! route a path to the right loader without reading the whole file twice.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use engine::seq2seq::config::{is_seq2seq, SEQ2SEQ_HEADER_BYTES};
use engine::seq2seq::{expected_seq2seq_file_bytes, Seq2SeqWeights};
use engine::Seq2SeqConfig;

use crate::error::HostError;
use crate::loader::{f32_as_bytes, f32_as_bytes_mut};

/// Whether the file at `path` opens with the seq2seq magic (`"tis2"`).
///
/// Reads only the first four bytes. A missing file is an [`HostError::Io`]
/// (the caller would fail to open it either way); a file shorter than the magic
/// is simply not a seq2seq checkpoint.
pub fn is_seq2seq_file(path: impl AsRef<Path>) -> Result<bool, HostError> {
    let path = path.as_ref();
    let mut file = File::open(path).map_err(|e| HostError::io(path, e))?;
    let mut magic = [0u8; 4];
    match file.read_exact(&mut magic) {
        Ok(()) => Ok(is_seq2seq(&magic)),
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Ok(false),
        Err(e) => Err(HostError::io(path, e)),
    }
}

/// A loaded seq2seq checkpoint: parsed config plus the whole file as `f32`.
pub struct Seq2SeqModel {
    /// The validated model configuration.
    pub config: Seq2SeqConfig,
    /// On-disk size in bytes (header + weights).
    pub file_bytes: usize,
    /// Entire file reinterpreted as `f32`: the header words followed by all weights.
    data: Vec<f32>,
    /// Path it was loaded from, for diagnostics.
    path: PathBuf,
}

impl Seq2SeqModel {
    /// Load and validate a seq2seq checkpoint from `path`.
    ///
    /// Validates that the file is `f32`-sized, parses the [`Seq2SeqConfig`]
    /// header, and confirms the file length equals what the config implies.
    pub fn load(path: impl AsRef<Path>) -> Result<Seq2SeqModel, HostError> {
        let path = path.as_ref();
        let mut file = File::open(path).map_err(|e| HostError::io(path, e))?;
        let len = file.metadata().map_err(|e| HostError::io(path, e))?.len();

        if !len.is_multiple_of(4) {
            return Err(HostError::NotF32Aligned {
                path: path.to_path_buf(),
                len,
            });
        }
        let n_floats = (len / 4) as usize;

        // Read the whole file into a 4-aligned f32 buffer in one pass.
        let mut data = vec![0.0f32; n_floats];
        file.read_exact(f32_as_bytes_mut(&mut data))
            .map_err(|e| HostError::io(path, e))?;

        let config =
            Seq2SeqConfig::parse(f32_as_bytes(&data)).map_err(|e| HostError::engine(path, e))?;

        let expected = expected_seq2seq_file_bytes(&config);
        let file_bytes = len as usize;
        if expected != file_bytes {
            return Err(HostError::engine(
                path,
                engine::EngineError::SizeMismatch {
                    expected,
                    actual: file_bytes,
                },
            ));
        }

        Ok(Seq2SeqModel {
            config,
            file_bytes,
            data,
            path: path.to_path_buf(),
        })
    }

    /// Zero-copy view of all weight tensors.
    pub fn weights(&self) -> Result<Seq2SeqWeights<'_>, HostError> {
        Seq2SeqWeights::new(&self.data[SEQ2SEQ_HEADER_BYTES / 4..], &self.config)
            .map_err(|e| HostError::engine(&self.path, e))
    }

    /// On-disk size of just the weights (excludes the header).
    pub fn weight_bytes(&self) -> usize {
        self.file_bytes - SEQ2SEQ_HEADER_BYTES
    }
}
