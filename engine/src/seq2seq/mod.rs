//! Encoder-decoder (seq2seq) architecture support: Marian / OPUS-MT translation
//! models, alongside the decoder-only Llama path.
//!
//! This module is the second architecture the engine ships. It is deliberately
//! **parallel** to the decoder-only path — nothing here touches [`Config`],
//! [`forward`], or the RoPE/RMSNorm kernels, whose llama2.c parity gate must stay
//! bit-identical. The whole module sits behind the `seq2seq` cargo feature so a
//! decoder-only embedded build carries none of it.
//!
//! A Marian model differs from Llama in shape, not just weights: a bidirectional
//! encoder stack feeds a decoder stack through cross-attention, normalization is
//! LayerNorm (weight + bias) rather than RMSNorm, every linear has a bias,
//! positions are sinusoidal (computed, not stored) instead of RoPE, and the tied
//! lm_head adds a `final_logits_bias`. Checkpoints use their own on-disk format
//! (magic `"tis2"`), produced from a Hugging Face `MarianMTModel` by
//! `scripts/export_marian.py`.
//!
//! Current contents:
//! * [`config`] — [`Config`], header parsing and validation.
//! * [`weights`] — the fp32 tensor layout and zero-copy [`Weights`] views.
//! * [`memory`] — `const fn` arena budgets for the encoder buffers, the cross-
//!   and self-KV caches, and the per-step scratch.
//! * [`state`] — [`RunState`], the working buffers carved once from the arena.
//! * [`model`] — the forward passes: [`encode`] (the bidirectional encoder),
//!   [`decode_step`] (one autoregressive decoder step) with [`precompute_cross_kv`],
//!   and the [`greedy_decode`] convenience that drives them end to end — over fp32 *or*
//!   int8 weights via [`ModelWeights`].
//! * [`quantize`] — int8 quantization of the Marian weight set ([`QuantizedWeights`] for
//!   the 17 matmul matrices, [`KeptWeights`] for the fp32 biases / LayerNorms), layered
//!   on the shared [`quant`](crate::quant) primitives.
//!
//! Like the Llama path, the forward passes are allocation-free and carve their
//! working set once from the caller's [`Arena`](crate::Arena).
//!
//! [`Config`]: crate::llama::Config
//! [`forward`]: crate::llama::forward

pub mod config;
pub mod memory;
pub mod model;
pub mod quantize;
pub mod state;
pub mod weights;

pub use config::{Activation, Config};
pub use memory::{seq2seq_arena_floats, MemoryBudget};
pub use model::{decode_step, encode, greedy_decode, precompute_cross_kv, ModelWeights};
pub use quantize::{
    kept_floats, pack_kept, quantize_weights, quantized_scale_count, quantized_weight_count,
    KeptWeights, QuantizedWeights,
};
pub use state::RunState;
pub use weights::{expected_file_bytes, weight_floats, Weights};
