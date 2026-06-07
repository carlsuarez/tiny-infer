//! `tiny-infer` command-line host.
//!
//! Two modes:
//! * **Report** (default): load a model checkpoint and optional tokenizer, then
//!   print the parsed config, file-validation result, tokenizer metadata, and the
//!   pre-computed memory budget.
//! * **Generate** (`--prompt`): tokenize the prompt, run the forward pass per
//!   position, greedily pick the next token, and stream the decoded text. Requires
//!   a tokenizer. Temperature-0 (greedy) only in this milestone.

mod error;
mod loader;
mod tokenizer;

use std::io::{self, Write};
use std::process::ExitCode;
use std::time::Instant;

use engine::math::argmax;
use engine::memory::{arena_floats, MemoryBudget};
use engine::{forward, Arena, Config, RunState, Weights};

use crate::error::HostError;
use crate::loader::Model;
use crate::tokenizer::{Vocab, BOS_ID};

const USAGE: &str = "\
tiny-infer — embedded-style Llama-2 inference engine

USAGE:
    tiny-infer [OPTIONS] <MODEL> [TOKENIZER]

ARGS:
    <MODEL>        Path to a llama2.c checkpoint (.bin)
    [TOKENIZER]    Path to tokenizer.bin (optional)

OPTIONS:
    -m, --model <PATH>        Model checkpoint path (alternative to positional)
    -t, --tokenizer <PATH>    Tokenizer path (alternative to positional)
    -p, --prompt <TEXT>       Prompt to continue (enables text generation)
    -n, --steps <N>           Max tokens to generate (default: model seq_len)
        --temperature <F>     Sampling temperature; only 0 (greedy) is supported
    -h, --help                Print this help and exit

With no --prompt, tiny-infer prints the model config, validates the files, and
reports the memory budget. With --prompt (and a tokenizer) it generates text.";

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
        .ok_or_else(|| HostError::Usage("no model file given".into()))?;

    let model = Model::load(&model_path)?;
    // Validate the weight layout (proves every tensor view fits the file).
    let weights = model.weights()?;

    // Generation mode: requires a prompt and a tokenizer.
    if let Some(prompt) = args.prompt.as_deref() {
        let tok_path = args.tokenizer.ok_or_else(|| {
            HostError::Usage("generation (--prompt) requires a tokenizer (--tokenizer)".into())
        })?;
        let vocab = Vocab::load(&tok_path, model.config.vocab_size)?;
        return generate(
            &model.config,
            &weights,
            &vocab,
            prompt,
            args.steps,
            args.temperature,
        );
    }

    // Report mode.
    report_model(&model_path, &model);
    if let Some(tok_path) = args.tokenizer {
        let vocab = Vocab::load(&tok_path, model.config.vocab_size)?;
        report_tokenizer(&tok_path, &vocab);
    } else {
        println!("\nTokenizer: (none provided — pass a path to inspect tokenizer.bin)");
    }

    report_memory(&model);
    Ok(())
}

/// Greedily generate text from `prompt` and stream it to stdout.
///
/// Tokenizes the prompt (with BOS), then for each position runs [`forward`] and
/// takes the next token: still inside the prompt it replays the prompt tokens;
/// afterwards it picks `argmax(logits)`. Stops at `steps` or when the model emits
/// BOS (the sequence delimiter). A tokens/sec line is written to stderr.
fn generate(
    config: &Config,
    weights: &Weights,
    vocab: &Vocab,
    prompt: &str,
    steps: Option<usize>,
    temperature: f32,
) -> Result<(), HostError> {
    if temperature != 0.0 {
        return Err(HostError::Usage(
            "only --temperature 0 (greedy) is supported in this milestone; \
             sampling arrives in a later milestone"
                .into(),
        ));
    }

    // `encode` always yields at least the BOS token, so `prompt_tokens[0]` is safe.
    let prompt_tokens = vocab.encode(prompt, true);
    let steps = steps.unwrap_or(config.seq_len).min(config.seq_len);

    // One arena, sized exactly to the budget; the run state is carved once.
    let mut arena_buf = vec![0.0f32; arena_floats(config)];
    let mut arena = Arena::new(&mut arena_buf);
    let mut state =
        RunState::new(&mut arena, config).map_err(|e| HostError::engine("<arena>", e))?;

    let stdout = io::stdout();
    let mut out = stdout.lock();

    let start = Instant::now();
    let mut token = prompt_tokens[0];
    let mut pos = 0usize;
    while pos < steps {
        let logits = forward(config, weights, &mut state, token, pos);
        let next = if pos + 1 < prompt_tokens.len() {
            prompt_tokens[pos + 1]
        } else {
            argmax(logits)
        };
        pos += 1;
        // BOS delimits sequences in llama2.c — stop if the model emits it.
        if next == BOS_ID {
            break;
        }
        let piece = vocab.decode(token, next);
        out.write_all(&piece)
            .map_err(|e| HostError::io("<stdout>", e))?;
        token = next;
    }
    let _ = out.write_all(b"\n");
    let _ = out.flush();

    // Report throughput (skip trivially short runs).
    if pos > 1 {
        let secs = start.elapsed().as_secs_f64();
        let tps = pos as f64 / secs;
        eprintln!("[{pos} tokens, {secs:.3}s, {tps:.1} tok/s]");
    }
    Ok(())
}

fn report_model(path: &str, model: &Model) {
    let c = &model.config;
    println!("Model: {path}");
    println!("  status         OK (file size matches config)");
    println!("  dim            {}", c.dim);
    println!("  hidden_dim     {}", c.hidden_dim);
    println!("  n_layers       {}", c.n_layers);
    println!("  n_heads        {}", c.n_heads);
    println!("  n_kv_heads     {}", c.n_kv_heads);
    println!("  vocab_size     {}", c.vocab_size);
    println!("  seq_len        {}", c.seq_len);
    println!("  head_size      {}", c.head_size());
    println!("  kv_dim         {}", c.kv_dim());
    println!("  kv_mul         {}", c.kv_mul());
    println!(
        "  shared_weights {} (output projection {})",
        c.shared_weights,
        if c.shared_weights {
            "reuses token embedding"
        } else {
            "is a separate matrix"
        }
    );
    println!(
        "  weights        {} ({})",
        human_bytes(model.weight_bytes()),
        commas(model.weight_bytes())
    );
}

fn report_tokenizer(path: &str, vocab: &Vocab) {
    println!("\nTokenizer: {path}");
    println!("  tokens             {}", vocab.len());
    println!("  max_token_length   {}", vocab.max_token_length);
    println!("  sample tokens:");
    for id in 0..4.min(vocab.len()) {
        print_sample(id, vocab);
    }
    if let Some(id) = vocab.tokens.iter().position(|t| t.is_byte_fallback()) {
        println!("  first byte-fallback token:");
        print_sample(id, vocab);
    }
}

fn print_sample(id: usize, vocab: &Vocab) {
    let t = &vocab.tokens[id];
    println!("    [{id:>5}] score {:>9.4}  {:?}", t.score, t.display());
}

fn report_memory(model: &Model) {
    let budget = MemoryBudget::for_config(&model.config);
    println!("\nMemory budget (working arena, pre-allocated once):");
    println!(
        "  activations    {:>10}  ({} f32)",
        human_bytes(budget.activation_bytes()),
        commas(budget.activation_floats)
    );
    println!(
        "  KV cache       {:>10}  ({} f32)",
        human_bytes(budget.kv_cache_bytes()),
        commas(budget.kv_cache_floats)
    );
    println!(
        "  arena total    {:>10}  ({} f32)",
        human_bytes(budget.total_bytes()),
        commas(budget.total_floats())
    );
    println!(
        "  weights (disk) {:>10}",
        human_bytes(model.weight_bytes())
    );
    println!(
        "  peak RAM       {:>10}  (weights + arena)",
        human_bytes(model.weight_bytes() + budget.total_bytes())
    );
}

/// Render a byte count as a human-friendly KiB/MiB/GiB string.
fn human_bytes(n: usize) -> String {
    const UNITS: [&str; 4] = ["B", "KiB", "MiB", "GiB"];
    let mut v = n as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{n} B")
    } else {
        format!("{v:.2} {}", UNITS[u])
    }
}

/// Group an integer with thousands separators (e.g. `884736` → `884,736`).
fn commas(n: usize) -> String {
    let s = n.to_string();
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, &b) in bytes.iter().enumerate() {
        if i > 0 && (bytes.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(b as char);
    }
    out
}

/// Parsed command-line arguments.
struct Args {
    model: Option<String>,
    tokenizer: Option<String>,
    prompt: Option<String>,
    steps: Option<usize>,
    temperature: f32,
    help: bool,
}

impl Args {
    fn parse(args: impl Iterator<Item = String>) -> Result<Args, HostError> {
        let mut model = None;
        let mut tokenizer = None;
        let mut prompt = None;
        let mut steps = None;
        let mut temperature = 0.0;
        let mut help = false;
        let mut positionals = Vec::new();

        let mut it = args.peekable();
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "-h" | "--help" => help = true,
                "-m" | "--model" => model = Some(expect_value(&mut it, &arg)?),
                "-t" | "--tokenizer" => tokenizer = Some(expect_value(&mut it, &arg)?),
                "-p" | "--prompt" => prompt = Some(expect_value(&mut it, &arg)?),
                "-n" | "--steps" => steps = Some(parse_usize(&expect_value(&mut it, &arg)?, &arg)?),
                "--temperature" => {
                    temperature = parse_f32(&expect_value(&mut it, &arg)?, &arg)?;
                }
                s if s.starts_with("--model=") => model = Some(after_eq(s)),
                s if s.starts_with("--tokenizer=") => tokenizer = Some(after_eq(s)),
                s if s.starts_with("--prompt=") => prompt = Some(after_eq(s)),
                s if s.starts_with("--steps=") => steps = Some(parse_usize(&after_eq(s), "--steps")?),
                s if s.starts_with("--temperature=") => {
                    temperature = parse_f32(&after_eq(s), "--temperature")?;
                }
                s if s.starts_with('-') && s != "-" => {
                    return Err(HostError::Usage(format!("unknown option `{s}`")));
                }
                _ => positionals.push(arg),
            }
        }

        // Positionals fill model then tokenizer, but never override explicit flags.
        let mut pos = positionals.into_iter();
        if model.is_none() {
            model = pos.next();
        }
        if tokenizer.is_none() {
            tokenizer = pos.next();
        }
        if let Some(extra) = pos.next() {
            return Err(HostError::Usage(format!("unexpected argument `{extra}`")));
        }

        Ok(Args {
            model,
            tokenizer,
            prompt,
            steps,
            temperature,
            help,
        })
    }
}

fn expect_value(
    it: &mut std::iter::Peekable<impl Iterator<Item = String>>,
    flag: &str,
) -> Result<String, HostError> {
    it.next()
        .ok_or_else(|| HostError::Usage(format!("`{flag}` expects a value")))
}

fn after_eq(s: &str) -> String {
    s.split_once('=').map(|x| x.1).unwrap_or("").to_string()
}

fn parse_usize(s: &str, flag: &str) -> Result<usize, HostError> {
    s.parse::<usize>()
        .map_err(|_| HostError::Usage(format!("`{flag}` expects a non-negative integer, got `{s}`")))
}

fn parse_f32(s: &str, flag: &str) -> Result<f32, HostError> {
    s.parse::<f32>()
        .map_err(|_| HostError::Usage(format!("`{flag}` expects a number, got `{s}`")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Args, HostError> {
        Args::parse(args.iter().map(|s| s.to_string()))
    }

    #[test]
    fn positional_model_and_tokenizer() {
        let a = parse(&["model.bin", "tok.bin"]).unwrap();
        assert_eq!(a.model.as_deref(), Some("model.bin"));
        assert_eq!(a.tokenizer.as_deref(), Some("tok.bin"));
    }

    #[test]
    fn flags_with_values() {
        let a = parse(&["-m", "model.bin", "--tokenizer=tok.bin"]).unwrap();
        assert_eq!(a.model.as_deref(), Some("model.bin"));
        assert_eq!(a.tokenizer.as_deref(), Some("tok.bin"));
    }

    #[test]
    fn help_flag() {
        assert!(parse(&["--help"]).unwrap().help);
    }

    #[test]
    fn missing_flag_value_is_usage_error() {
        assert!(matches!(parse(&["-m"]), Err(HostError::Usage(_))));
    }

    #[test]
    fn unknown_flag_is_usage_error() {
        assert!(matches!(parse(&["--frobnicate"]), Err(HostError::Usage(_))));
    }

    #[test]
    fn too_many_positionals_is_error() {
        assert!(matches!(
            parse(&["a.bin", "b.bin", "c.bin"]),
            Err(HostError::Usage(_))
        ));
    }

    #[test]
    fn generation_flags_parse() {
        let a = parse(&["-m", "m.bin", "-p", "Once upon a time", "-n", "64"]).unwrap();
        assert_eq!(a.prompt.as_deref(), Some("Once upon a time"));
        assert_eq!(a.steps, Some(64));
        assert_eq!(a.temperature, 0.0);

        let b = parse(&["m.bin", "--prompt=hi", "--temperature=0"]).unwrap();
        assert_eq!(b.prompt.as_deref(), Some("hi"));
        assert_eq!(b.temperature, 0.0);
    }

    #[test]
    fn bad_steps_value_is_usage_error() {
        assert!(matches!(parse(&["m.bin", "-n", "lots"]), Err(HostError::Usage(_))));
        assert!(matches!(
            parse(&["m.bin", "--temperature=warm"]),
            Err(HostError::Usage(_))
        ));
    }

    #[test]
    fn human_bytes_scales() {
        assert_eq!(human_bytes(512), "512 B");
        assert_eq!(human_bytes(2048), "2.00 KiB");
        assert_eq!(human_bytes(3 * 1024 * 1024), "3.00 MiB");
    }

    #[test]
    fn commas_groups_thousands() {
        assert_eq!(commas(884_736), "884,736");
        assert_eq!(commas(100), "100");
        assert_eq!(commas(1_000), "1,000");
    }
}
