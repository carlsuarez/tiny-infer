//! Checkpoint loading: read a llama2.c `.bin` file and expose its config and a
//! zero-copy `f32` weight view.
//!
//! ## Alignment
//!
//! The engine's [`Weights`] view borrows `&[f32]` straight out of the file buffer,
//! so that buffer must be 4-byte aligned. A `Vec<u8>` only guarantees alignment 1,
//! so instead we read the file directly into a `Vec<f32>` (which is 4-aligned by
//! construction) via a byte view of its storage. That keeps the load to a single
//! read with no extra copy, and confines the only `unsafe` to one audited cast
//! (`f32` ↔ `u8`, which is always sound — `u8` has alignment 1).

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use engine::config::HEADER_BYTES;
use engine::weights::expected_file_bytes;
use engine::{Config, Weights};

use crate::error::HostError;

/// A loaded checkpoint: parsed config plus the whole file as `f32`.
pub struct Model {
    /// The validated model configuration.
    pub config: Config,
    /// On-disk size in bytes (header + weights).
    pub file_bytes: usize,
    /// Entire file reinterpreted as `f32`: 7 header words followed by all weights.
    data: Vec<f32>,
    /// Path it was loaded from, for diagnostics.
    path: PathBuf,
}

impl Model {
    /// Load and validate a checkpoint from `path`.
    ///
    /// Validates that the file is `f32`-sized, parses the [`Config`] header, and
    /// confirms the file length equals what the config implies.
    pub fn load(path: impl AsRef<Path>) -> Result<Model, HostError> {
        let path = path.as_ref();
        let mut file = File::open(path).map_err(|e| HostError::io(path, e))?;
        let len = file
            .metadata()
            .map_err(|e| HostError::io(path, e))?
            .len();

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

        // Parse and validate the config from the first 28 bytes.
        let header = &f32_as_bytes(&data)[..HEADER_BYTES.min(n_floats * 4)];
        let config = Config::parse(header).map_err(|e| HostError::engine(path, e))?;

        let expected = expected_file_bytes(&config);
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

        Ok(Model {
            config,
            file_bytes,
            data,
            path: path.to_path_buf(),
        })
    }

    /// The weight region (everything after the 7-word header) as `f32`.
    fn weight_floats(&self) -> &[f32] {
        &self.data[HEADER_BYTES / 4..]
    }

    /// Zero-copy view of all weight tensors.
    pub fn weights(&self) -> Result<Weights<'_>, HostError> {
        Weights::new(self.weight_floats(), &self.config)
            .map_err(|e| HostError::engine(&self.path, e))
    }

    /// On-disk size of just the weights (excludes the 28-byte header).
    pub fn weight_bytes(&self) -> usize {
        self.file_bytes - HEADER_BYTES
    }
}

/// Reinterpret an `&[f32]` as raw bytes. Sound: `u8` has alignment 1, and every
/// bit pattern of `f32` is a valid `u8`.
fn f32_as_bytes(floats: &[f32]) -> &[u8] {
    // SAFETY: `floats` is valid for `len*4` bytes; `u8` has no alignment or
    // validity requirements that `f32` storage does not already satisfy.
    unsafe { std::slice::from_raw_parts(floats.as_ptr() as *const u8, std::mem::size_of_val(floats)) }
}

/// Mutable counterpart of [`f32_as_bytes`], used to read the file in place.
fn f32_as_bytes_mut(floats: &mut [f32]) -> &mut [u8] {
    let len = std::mem::size_of_val(floats);
    // SAFETY: exclusive access to `floats`; same reasoning as `f32_as_bytes`.
    unsafe { std::slice::from_raw_parts_mut(floats.as_mut_ptr() as *mut u8, len) }
}
