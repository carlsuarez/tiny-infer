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
//! the tokenizer, and the CLI.
//!
//! Milestone 1 implements config/weight parsing, the arena, and the memory budget.
//! Milestone 2 adds the fp32 [`math`] kernels, the [`RunState`] working set, and the
//! [`forward`] pass that turns a token into vocabulary logits.
//!
//! Beyond the decoder-only Llama path, the [`seq2seq`] module (behind the on-by-default
//! `seq2seq` cargo feature) adds a second architecture: Marian / OPUS-MT encoder-decoder
//! translation models, with their own checkpoint format, weight views, and arena budget.
//! Disable the feature for a decoder-only build.
//!
//! # `no_std` and panic policy
//!
//! The crate compiles against nothing but `core` and [`libm`] (for the transcendental
//! functions `core` lacks). It is verified on a bare-metal target two ways: the library
//! itself (`cargo build -p engine --target thumbv7em-none-eabi`) and a freestanding
//! firmware binary that supplies its own `#[panic_handler]` and runs a full forward pass
//! out of stack buffers with no allocator (`examples/baremetal.rs`). Because every
//! transcendental goes through `libm` rather than `std`'s `f32` methods, that bare-metal
//! build doubles as the guard against a `std`-only float intrinsic ever sneaking in — it
//! simply would not compile.
//!
//! **Allocation.** Nothing here allocates. The working set is carved once from a
//! caller-provided [`Arena`] and reused in place every step; the budget is a `const fn`
//! of the [`Config`] ([`memory::arena_floats`]), so a host can size a `static` arena —
//! and `const`-assert it fits a fixed RAM budget — at compile time.
//!
//! **Panics.** Every operation driven by *external* input — parsing a checkpoint header,
//! carving the weight views, sizing the arena — is fallible and returns [`EngineError`]
//! instead of panicking, so malformed files can never crash a caller. The remaining
//! panic sites guard *internal* invariants (a programmer error, not input): the kernels'
//! `debug_assert!` length checks, which compile out of release builds, plus core's own
//! bounds/overflow checks on buffers whose sizes the validated [`Config`] already
//! guarantees. The engine is built to run under `panic = "abort"` (the release profile
//! sets it), needing no unwinder.

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
#[cfg(feature = "seq2seq")]
pub mod seq2seq;
pub mod state;
pub mod weights;

pub use arena::Arena;
pub use config::{parse_header, Config, ModelFormat};
pub use error::EngineError;
pub use model::{forward, Kernel, ModelWeights};
pub use quantize::{QuantizedTensor, QuantizedWeights};
#[cfg(feature = "seq2seq")]
pub use seq2seq::{Activation, Seq2SeqConfig, Seq2SeqWeights};
pub use state::{QuantScratch, RunState};
pub use weights::Weights;
