//! Host support for the decoder-only **Llama-2** path: checkpoint loading, the
//! SentencePiece-BPE tokenizer, format conversion, text generation, and reporting.
//!
//! The sibling of [`crate::seq2seq`]. [`run`] is the entry point `main` dispatches a
//! llama2.c checkpoint to (report, generate, or convert); the submodules hold the
//! pieces it drives.

pub mod convert;
pub mod generate;
pub mod loader;
pub mod report;
pub mod run;
pub mod tokenizer;

pub use run::run;
