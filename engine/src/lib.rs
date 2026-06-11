//! # tiny-infer engine
//!
//! `no_std`, allocation-free core of the tiny-infer transformer inference engine.
//!
//! This crate deliberately depends on nothing but `core`. It owns the parts of the
//! system that must run on an embedded/edge target: the model [`Config`], the
//! zero-copy [`Weights`] view over a checkpoint's `f32` region, the bump [`Arena`]
//! that backs every working buffer, and the [`memory`] budget math used to size that
//! arena ahead of time.
//!
//! The host crate (which *is* `std`) is responsible for file IO, the byte→`f32` cast,
//! the tokenizer, and the CLI. Nothing in here allocates or panics on the hot path:
//! fallible operations return [`EngineError`].
//!
//! Milestone 1 implements config/weight parsing, the arena, and the memory budget.
//! Milestone 2 adds the fp32 [`math`] kernels, the [`RunState`] working set, and the
//! [`forward`] pass that turns a token into vocabulary logits.

#![no_std]
// The SIMD matmul kernels in `math` use `core::simd`, which is still nightly-only;
// the workspace pins a nightly toolchain (see `rust-toolchain.toml`).
#![feature(portable_simd)]
#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod arena;
pub mod config;
pub mod error;
pub mod math;
pub mod memory;
pub mod model;
pub mod quantize;
pub mod state;
pub mod weights;

pub use arena::Arena;
pub use config::{parse_header, Config, ModelFormat};
pub use error::EngineError;
pub use model::{forward, Kernel, ModelWeights};
pub use quantize::{QuantizedTensor, QuantizedWeights};
pub use state::{QuantScratch, RunState};
pub use weights::Weights;
