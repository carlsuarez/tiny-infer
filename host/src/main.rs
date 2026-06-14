//! `tiny-infer` command-line host.
//!
//! A thin front end over two architecture modules that mirror the engine's split:
//! * [`llama`] — decoder-only llama2.c checkpoints: report, generate (`--prompt`),
//!   or convert between formats.
//! * [`seq2seq`] — Marian / OPUS-MT encoder-decoder checkpoints: report, or generate an
//!   output sequence from the prompt (`--prompt`).
//!
//! [`main`] parses the arguments and [`run`] dispatches to the right module by
//! sniffing the checkpoint's magic; everything else lives in the modules.

mod args;
mod error;
mod llama;
mod seq2seq;

use std::process::ExitCode;

use crate::args::Args;
use crate::error::HostError;

const USAGE: &str = "\
tiny-infer — embedded-style transformer inference engine

The architecture is detected from the checkpoint's magic — no flag selects it:
  * Llama   — decoder-only llama2.c checkpoints (legacy / v1 / v2).
  * Seq2seq — Marian / OPUS-MT encoder-decoder models (scripts/export_marian.py).
With --prompt + a tokenizer, Llama continues the text and seq2seq runs the
encoder-decoder to generate an output sequence for the input (translation,
summarization, … depending on the model). With no --prompt, either reports the
model's config and memory budget instead. The seq2seq tokenizer defaults to
tokenizer.bin next to the model.

USAGE:
    tiny-infer [OPTIONS] <MODEL> [TOKENIZER]

ARGS:
    <MODEL>        Checkpoint (.bin): a llama2.c model or a tiny-infer seq2seq model
    [TOKENIZER]    Tokenizer (Llama: required to generate; seq2seq: optional,
                   defaults to tokenizer.bin next to the model)

OPTIONS:
    -m, --model <PATH>        Model path (alternative to the positional <MODEL>)
    -t, --tokenizer <PATH>    Tokenizer path (alternative to the positional)
    -p, --prompt <TEXT>       Llama: text to continue · seq2seq: input sequence to transform
        --scalar              Use the scalar matmul kernels instead of the default
                              core::simd ones (the readable reference path)
    -h, --help                Print this help and exit

  sampling (picks each generated token, Llama and seq2seq alike):
        --temperature <F>     Sampling temperature (default 0 = greedy/deterministic;
                              higher = more random). For seq2seq, greedy (temp 0) —
                              or beam search — is the quality path; temperature/top-p add
                              diversity, not accuracy.
        --topp <F>            Nucleus (top-p) threshold in (0,1): sample only from the
                              most-probable tokens summing to F
        --seed <N>            RNG seed for reproducible sampling (default: random)

  int8 quantization (Llama and seq2seq alike):
    -q, --quantize            Quantize matmul weights to int8 (group-wise) before
                              running — smaller weight footprint, slightly lossy
        --group-size <N>      Int8 group size; must divide the model's matmul dims
                              (Llama: dim/hidden_dim/kv_dim · seq2seq: d_model/enc_ffn/
                              dec_ffn) (default: 32)
        --dotprod             Use the hardware int8 dot-product kernel (x86 AVX-512 VNNI
                              or ARM NEON sdot, falling back to SIMD if the CPU has
                              neither) for the int8 matmuls produced by --quantize; has
                              no effect without --quantize (the fp32 path runs SIMD)

  Llama-only:
    -n, --steps <N>           Max tokens to generate (default: model seq_len); ignored
                              for seq2seq, which always stops at eos
        --convert <PATH>      Convert the checkpoint to PATH and exit (no generation;
                              seq2seq has no on-disk int8 format yet, so this is an
                              error for seq2seq checkpoints)
        --to <v1|v2>          Conversion target (default v2 = int8 Q8_0 / runq.c's
                              format, uses --group-size; v1 = fp32, versioned header)\
";

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(HostError::Usage(msg)) => {
            eprintln!("error: {msg}\n\n{USAGE}");
            ExitCode::from(2)
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), HostError> {
    let args = Args::parse(std::env::args().skip(1))?;
    if args.help {
        println!("{USAGE}");
        return Ok(());
    }
    let model_path = args
        .model
        .clone()
        .ok_or_else(|| HostError::Usage("no model file given".into()))?;
    if args.to.is_some() && args.convert.is_none() {
        return Err(HostError::Usage(
            "`--to` requires `--convert <PATH>`".into(),
        ));
    }
    if args.convert.is_some() && args.prompt.is_some() {
        return Err(HostError::Usage(
            "`--convert` writes a checkpoint and exits; it cannot be combined with --prompt".into(),
        ));
    }

    // Route by architecture, sniffing each format from the checkpoint's leading bytes.
    // Order matters: the magic-bearing formats are tested first, because the llama
    // *legacy* format has no magic and `is_llama_file` is a structural fallback that a
    // foreign header could otherwise match (see `engine::llama::config::is_llama`). To
    // add a new architecture, give it its own magic and slot its check above llama's.
    if seq2seq::is_seq2seq_file(&model_path)? {
        seq2seq::run(&model_path, &args)
    } else if llama::is_llama_file(&model_path)? {
        llama::run(&model_path, &args)
    } else {
        Err(HostError::Format {
            path: model_path.into(),
            msg: "unrecognized checkpoint format (expected a llama2.c .bin or a \
                  tiny-infer seq2seq \"tis2\" checkpoint)"
                .into(),
        })
    }
}
