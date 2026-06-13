//! `tiny-infer` command-line host.
//!
//! A thin front end over two architecture modules that mirror the engine's split:
//! * [`llama`] — decoder-only llama2.c checkpoints: report, generate (`--prompt`),
//!   or convert between formats.
//! * [`seq2seq`] — Marian / OPUS-MT translation checkpoints: report, or translate
//!   (`--prompt`).
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

The architecture is detected from the checkpoint's magic, and the flags below split
accordingly:
  * Llama   — decoder-only llama2.c checkpoints (legacy / v1 / v2). With --prompt +
              a tokenizer, generates text; supports int8 quantization and conversion.
  * Seq2seq — Marian / OPUS-MT translation models (scripts/export_marian.py). With
              --prompt, translates the text to the model's target language; supports
              int8 quantization too (only conversion is Llama-only).
With no --prompt, either path reports the model's config and memory budget instead.

USAGE:
    tiny-infer [OPTIONS] <MODEL> [TOKENIZER]

ARGS:
    <MODEL>        Checkpoint (.bin): a llama2.c model or a tiny-infer seq2seq model
    [TOKENIZER]    Tokenizer (Llama: required to generate; seq2seq: optional,
                   defaults to tokenizer.bin next to the model)

BOTH ARCHITECTURES:
    -m, --model <PATH>        Model path (alternative to the positional <MODEL>)
    -t, --tokenizer <PATH>    Tokenizer path (alternative to the positional)
    -p, --prompt <TEXT>       Llama: text to continue · seq2seq: text to translate
        --scalar              Use the scalar matmul kernels instead of the default
                              core::simd ones (the readable reference path)
    -h, --help                Print this help and exit

  int8 quantization (Llama generation · seq2seq translation):
    -q, --quantize            Quantize matmul weights to int8 (group-wise) before
                              running — smaller weight footprint, slightly lossy
        --group-size <N>      Int8 group size; must divide the model's matmul dims
                              (Llama: dim/hidden_dim/kv_dim · seq2seq: d_model/enc_ffn/
                              dec_ffn) (default: 32)
        --dotprod             Use the hardware int8 dot-product kernel for the
                              quantized path (x86 AVX-512 VNNI or ARM NEON sdot; falls
                              back to SIMD if the CPU has neither)

LLAMA ONLY:
  generation (with --prompt):
    -n, --steps <N>           Max tokens to generate (default: model seq_len)
        --temperature <F>     Sampling temperature (default 0 = greedy/deterministic;
                              higher = more random)
        --topp <F>            Nucleus (top-p) threshold in (0,1): sample only from the
                              most-probable tokens summing to F
        --seed <N>            RNG seed for reproducible sampling (default: random)
  format conversion:
        --convert <PATH>      Convert the checkpoint to PATH and exit (no generation)
        --to <v1|v2>          Conversion target (default v2 = int8 Q8_0 / runq.c's
                              format, uses --group-size; v1 = fp32, versioned header)

SEQ2SEQ (Marian / OPUS-MT) ONLY:
    Translation is driven by --prompt (add --scalar / -q / --dotprod as above); the
    tokenizer defaults to tokenizer.bin next to the model. Format conversion (--convert)
    is Llama-only — there is no on-disk int8 seq2seq format yet.\
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

    // Route by architecture: the seq2seq checkpoints carry their own magic, so a
    // four-byte sniff sends each path to its own loader and driver.
    if seq2seq::is_seq2seq_file(&model_path)? {
        seq2seq::run(&model_path, &args)
    } else {
        llama::run(&model_path, &args)
    }
}
