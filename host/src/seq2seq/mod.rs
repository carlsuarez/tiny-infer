//! Host support for the encoder-decoder **Marian / OPUS-MT** path: checkpoint
//! loading, the SentencePiece Unigram tokenizer, greedy translation, and reporting.
//!
//! The sibling of [`crate::llama`]. [`run`] is the entry point `main` dispatches a
//! seq2seq checkpoint to (translate or report); [`is_seq2seq_file`] is how `main`
//! recognizes one in the first place.

pub mod loader;
pub mod report;
pub mod run;
pub mod tokenizer;
pub mod translate;

pub use loader::is_seq2seq_file;
pub use run::run;
