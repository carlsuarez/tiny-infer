//! # tiny-infer engine
//!
//! `no_std`, allocation-free core of the tiny-infer transformer inference engine.
//!
//! This crate deliberately keeps its dependencies tiny and `no_std`: [`libm`] for the
//! transcendental functions `core` lacks, [`rand`] (no default features) for the
//! sampler's seedable PRNG, and [`wide`] for portable SIMD vectors on stable Rust. The
//! host crate (which *is* `std`) owns file IO, the byte→`f32` cast, the tokenizer, and
//! the CLI.
//!
//! ## Structure: a shared core plus two architectures
//!
//! The crate is organized so the architecture split is visible in the module tree.
//! A small **shared core** holds the building blocks that are independent of any one
//! model shape:
//!
//! * [`arena`] — the bump [`Arena`] that backs every working buffer.
//! * [`error`] — the crate-wide [`EngineError`].
//! * [`math`] — the fp32 and int8 compute kernels (matmul, norms, attention, …).
//! * [`nn`] — the 1-D CNN kernels ([`conv1d`](nn::conv1d), [`relu`](nn::relu),
//!   [`global_avg_pool`](nn::global_avg_pool)) a feature-window classifier is built from.
//! * [`quant`] — the group-wise int8 quantization *primitives* ([`QuantizedTensor`]
//!   and the quantize/dequantize routines the kernels operate on).
//! * [`sample`] — the [`Sampler`] that turns a logits row into the next token id
//!   (greedy / temperature / top-p), shared by both decode loops.
//!
//! On top of that core sit two **parallel, self-contained architecture modules**,
//! each owning its own config, weight layout, arena budget, working state, and
//! forward pass:
//!
//! * [`llama`] — the decoder-only **Llama-2** path (llama2.c-compatible: RMSNorm,
//!   RoPE, SwiGLU, causal GQA). Always compiled in. Its public types
//!   ([`Config`](llama::Config), [`Weights`](llama::Weights),
//!   [`forward`](llama::forward), [`ModelWeights`](llama::ModelWeights), …) are
//!   reached through the module path, `engine::llama::*`.
//! * [`seq2seq`] — the encoder-decoder **Marian / OPUS-MT** path (LayerNorm with
//!   bias, sinusoidal positions, cross-attention), reached through
//!   `engine::seq2seq::*`.
//!
//! The two architecture modules never reference each other — they meet only at the
//! shared core — so the directory layout mirrors exactly how separate they are. Each
//! names its config `Config`, its weights `Weights`, and its working set `RunState`;
//! the module path keeps them distinct.
//!
//! # `no_std` and panic policy
//!
//! The crate compiles against `core` plus three `no_std` crates — [`libm`] (for the
//! transcendental functions `core` lacks), [`rand`] (the sampler PRNG, no default
//! features so it pulls in no `std`/OS entropy), and [`wide`] (portable SIMD vectors on
//! stable Rust, so the vectorized matmuls need no nightly `core::simd`). It is verified on a bare-metal target two ways: the library
//! itself (`cargo build -p engine --target thumbv7em-none-eabi`) and a freestanding
//! firmware binary that supplies its own `#[panic_handler]` and runs a full forward pass
//! out of stack buffers with no allocator (`examples/baremetal.rs`). Because every
//! transcendental goes through `libm` rather than `std`'s `f32` methods, that bare-metal
//! build doubles as the guard against a `std`-only float intrinsic ever sneaking in — it
//! simply would not compile.
//!
//! **Allocation.** Nothing here allocates. The working set is carved once from a
//! caller-provided [`Arena`] and reused in place every step; the budget is a `const fn`
//! of the [`Config`](llama::Config) ([`llama::memory::arena_floats`]), so a host can size a `static` arena —
//! and `const`-assert it fits a fixed RAM budget — at compile time.
//!
//! **Panics.** Every operation driven by *external* input — parsing a checkpoint header,
//! carving the weight views, sizing the arena — is fallible and returns [`EngineError`]
//! instead of panicking, so malformed files can never crash a caller. The remaining
//! panic sites guard *internal* invariants (a programmer error, not input): the kernels'
//! `debug_assert!` length checks, which compile out of release builds, plus core's own
//! bounds/overflow checks on buffers whose sizes the validated [`Config`](llama::Config) already
//! guarantees. The engine is built to run under `panic = "abort"` (the release profile
//! sets it), needing no unwinder.

#![no_std]
#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

// Shared core: building blocks independent of any one architecture.
pub mod arena;
pub mod error;
pub mod math;
pub mod nn;
pub mod quant;
pub mod sample;

// Architectures, each layered on the shared core. Their public types are reached
// through the module path — `engine::llama::Config`, `engine::seq2seq::Config`, … —
// so the two never collide and the directory layout *is* the API surface.
pub mod llama;
pub mod seq2seq;

// Shared-core re-exports (architecture-agnostic, so unqualified at the root).
pub use arena::Arena;
pub use error::EngineError;
pub use math::Kernel;
pub use quant::{QuantScratch, QuantizedTensor};
pub use sample::Sampler;
