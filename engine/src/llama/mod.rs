//! Decoder-only **Llama-2** architecture: the llama2.c-compatible path.
//!
//! This is the engine's first and reference architecture — a causal,
//! decoder-only transformer (TinyStories / Llama-2-style checkpoints) whose
//! greedy output must stay bit-for-bit identical to Karpathy's llama2.c. It is
//! always compiled in (unlike the optional [`seq2seq`](crate::seq2seq) path) and
//! defines the public crate-root API: [`Config`], [`Weights`], [`forward`], and
//! friends are all re-exported from here at the crate root.
//!
//! A Llama-2 layer is RMSNorm (gain only, no bias) → causal GQA self-attention
//! with RoPE position encoding → SwiGLU feed-forward, all pre-norm, with the token
//! embedding table tied to the output classifier. Checkpoints use the llama2.c
//! binary formats (legacy v1 and the quantized `runq.c` v2), parsed by [`config`]
//! and [`weights`].
//!
//! Everything here builds on the shared core — the bump [`Arena`](crate::Arena),
//! the [`math`](crate::math) kernels, and the int8 [`quant`](crate::quant)
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
//!   [`quantize_weights`]), layered on the shared [`quant`](crate::quant) primitives.

pub mod config;
pub mod memory;
pub mod model;
pub mod quantize;
pub mod state;
pub mod weights;

pub use config::{parse_header, Config, ModelFormat};
pub use memory::{arena_floats, MemoryBudget};
pub use model::{forward, Kernel, ModelWeights};
pub use quantize::{quantize_weights, QuantizedWeights};
pub use state::{QuantScratch, RunState};
pub use weights::Weights;
