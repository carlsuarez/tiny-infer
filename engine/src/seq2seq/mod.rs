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
//! * [`config`] — [`Seq2SeqConfig`], header parsing and validation.
//! * [`weights`] — the fp32 tensor layout and zero-copy [`Seq2SeqWeights`] views.
//! * [`memory`] — `const fn` arena budgets for the encoder buffers, the cross-
//!   and self-KV caches, and the per-step scratch.
//!
//! The forward passes (encoder, then decoder + cross-attention) build on these in
//! later milestones; like the Llama path they will be allocation-free and carve
//! their working set once from the caller's [`Arena`](crate::Arena).
//!
//! [`Config`]: crate::Config
//! [`forward`]: crate::forward

pub mod config;
pub mod memory;
pub mod weights;

pub use config::{Activation, Seq2SeqConfig};
pub use memory::{seq2seq_arena_floats, Seq2SeqMemoryBudget};
pub use weights::{expected_seq2seq_file_bytes, seq2seq_weight_floats, Seq2SeqWeights};
