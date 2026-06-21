//! Decoder-only **Llama-2** architecture.
//!
//! A standalone `no_std`, allocation-free crate that builds the decoder-only
//! transformer (TinyStories / Llama-2-style checkpoints) on top of the shared
//! [`engine`] core — its bump [`Arena`](engine::Arena), the [`math`](engine::math)
//! kernels, and the int8 [`quant`](engine::quant) primitives. Greedy output stays
//! bit-for-bit identical to the reference decoder. The public API — [`Config`],
//! [`Weights`], [`forward`], and friends — is re-exported here at the crate root.
//!
//! A Llama-2 layer is RMSNorm (gain only, no bias) → causal GQA self-attention
//! with RoPE position encoding → SwiGLU feed-forward, all pre-norm, with the token
//! embedding table tied to the output classifier. Checkpoints use the documented
//! binary formats (legacy, v1, and the quantized int8 v2), parsed by [`config`]
//! and [`weights`].
//!
//! Everything here builds on the shared core — the bump [`Arena`](engine::Arena),
//! the [`math`](engine::math) kernels, and the int8 [`quant`](engine::quant)
//! primitives — which it shares with the seq2seq path. What is specific to this
//! architecture lives here:
//!
//! * [`config`] — [`Config`], the 7-field header, format detection, validation.
//! * [`weights`] — the fp32 tensor layout and zero-copy [`Weights`] views.
//! * [`memory`] — `const fn` arena budgets (activations + KV cache) used to size a
//!   `static` arena ahead of time.
//! * [`state`] — [`RunState`], the working buffers carved once from the arena, plus
//!   the W8A8 [`QuantScratch`].
//! * [`model`] — [`forward`] (the per-token forward pass over fp32 *or* int8
//!   weights via [`ModelWeights`]) and the [`Kernel`] selector.
//! * [`quantize`] — int8 quantization of the Llama weight set ([`QuantizedWeights`],
//!   [`quantize_weights`]), layered on the shared [`quant`](engine::quant) primitives.

#![no_std]
#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod config;
pub mod memory;
pub mod model;
pub mod quantize;
pub mod state;
pub mod weights;

pub use config::{parse_header, Config, ModelFormat};
pub use engine::math::Kernel;
pub use memory::{arena_floats, MemoryBudget};
pub use model::{forward, ModelWeights};
pub use engine::quant::QuantScratch;
pub use quantize::{quantize_weights, QuantizedWeights};
pub use state::RunState;
pub use weights::Weights;
