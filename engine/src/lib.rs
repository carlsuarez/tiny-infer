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
//! ## The shared core
//!
//! This crate is purely the **architecture-independent core** — the building blocks
//! that don't depend on any one model shape:
//!
//! * [`arena`] — the bump [`Arena`] that backs every working buffer.
//! * [`error`] — the crate-wide [`EngineError`].
//! * [`math`] — the fp32 and int8 compute kernels (matmul, norms, attention, …).
//! * [`nn`] — the 1-D CNN kernels ([`conv1d`](nn::conv1d), [`relu`](nn::relu),
//!   [`global_avg_pool`](nn::global_avg_pool)) a feature-window classifier is built from.
//! * [`quant`] — the group-wise int8 quantization *primitives* ([`QuantizedTensor`]
//!   and the quantize/dequantize routines the kernels operate on).
//! * [`sample`] — the [`Sampler`] that turns a logits row into the next token id
//!   (greedy / temperature / top-p), shared by every decode loop.
//!
//! The model **architectures live in their own crates** that depend on this one — each
//! owning its config, weight layout, arena budget, working state, and forward pass:
//! `llama` (decoder-only Llama-2) and `seq2seq` (encoder-decoder Marian /
//! OPUS-MT). They build entirely on the core above and never reference each other, which
//! keeps the core small and genuinely reusable by any model crate.
//!
//! # `no_std` and panic policy
//!
//! The crate compiles against `core` plus three `no_std` crates — [`libm`] (for the
//! transcendental functions `core` lacks), [`rand`] (the sampler PRNG, no default
//! features so it pulls in no `std`/OS entropy), and [`wide`] (portable SIMD vectors on
//! stable Rust, so the vectorized matmuls need no nightly `core::simd`). It is verified on
//! a bare-metal target two ways: the library itself (`cargo build -p engine --target
//! thumbv7em-none-eabi`) and a freestanding firmware binary that supplies its own
//! `#[panic_handler]` and runs a full forward pass out of stack buffers with no allocator
//! (the `baremetal` example in the `llama` crate). Because every transcendental goes
//! through `libm` rather than `std`'s `f32` methods, that bare-metal build doubles as the
//! guard against a `std`-only float intrinsic ever sneaking in — it simply would not compile.
//!
//! **Allocation.** Nothing here allocates. A model crate carves its working set once from
//! a caller-provided [`Arena`] and reuses it in place every step; because each
//! architecture's arena budget is a `const fn` of its config, a host can size a `static`
//! arena — and `const`-assert it fits a fixed RAM budget — at compile time.
//!
//! **Panics.** Every operation driven by *external* input — parsing a checkpoint header,
//! carving the weight views, sizing the arena — is fallible and returns [`EngineError`]
//! instead of panicking, so malformed files can never crash a caller. The remaining
//! panic sites guard *internal* invariants (a programmer error, not input): the kernels'
//! `debug_assert!` length checks, which compile out of release builds, plus core's own
//! bounds/overflow checks on buffers whose sizes the validated config already
//! guarantees. The engine is built to run under `panic = "abort"` (the release profile
//! sets it), needing no unwinder.

#![no_std]
#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

// The shared core: building blocks independent of any one architecture. The model
// architectures live in their own crates (`llama`, `seq2seq`) that depend on this one.
pub mod arena;
pub mod error;
pub mod math;
pub mod nn;
pub mod quant;
pub mod sample;

// Shared-core re-exports (architecture-agnostic, so unqualified at the root).
pub use arena::Arena;
pub use error::EngineError;
pub use math::Kernel;
pub use quant::{QuantScratch, QuantizedTensor};
pub use sample::Sampler;
