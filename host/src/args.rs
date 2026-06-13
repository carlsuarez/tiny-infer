//! Command-line argument parsing — a small hand-rolled parser (a handful of flags,
//! no `clap`), shared by both architecture dispatchers.

use crate::error::HostError;
use crate::llama::convert::Target;

/// Parsed command-line arguments.
pub(crate) struct Args {
    pub(crate) model: Option<String>,
    pub(crate) tokenizer: Option<String>,
    pub(crate) prompt: Option<String>,
    pub(crate) steps: Option<usize>,
    pub(crate) temperature: f32,
    pub(crate) topp: f32,
    pub(crate) seed: Option<u64>,
    pub(crate) quantize: bool,
    pub(crate) group_size: usize,
    pub(crate) scalar: bool,
    pub(crate) dotprod: bool,
    pub(crate) convert: Option<String>,
    pub(crate) to: Option<Target>,
    pub(crate) help: bool,
}

impl Args {
    pub(crate) fn parse(args: impl Iterator<Item = String>) -> Result<Args, HostError> {
        let mut model = None;
        let mut tokenizer = None;
        let mut prompt = None;
        let mut steps = None;
        let mut temperature = 0.0;
        let mut topp = 0.0;
        let mut seed = None;
        let mut quantize = false;
        let mut group_size = 32usize;
        let mut scalar = false;
        let mut dotprod = false;
        let mut convert = None;
        let mut to = None;
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
                "--topp" => topp = parse_f32(&expect_value(&mut it, &arg)?, &arg)?,
                "--seed" => seed = Some(parse_u64(&expect_value(&mut it, &arg)?, &arg)?),
                "-q" | "--quantize" => quantize = true,
                "--scalar" => scalar = true,
                "--dotprod" => dotprod = true,
                "--group-size" => group_size = parse_usize(&expect_value(&mut it, &arg)?, &arg)?,
                "--convert" => convert = Some(expect_value(&mut it, &arg)?),
                "--to" => to = Some(parse_target(&expect_value(&mut it, &arg)?)?),
                s if s.starts_with("--model=") => model = Some(after_eq(s)),
                s if s.starts_with("--tokenizer=") => tokenizer = Some(after_eq(s)),
                s if s.starts_with("--prompt=") => prompt = Some(after_eq(s)),
                s if s.starts_with("--steps=") => {
                    steps = Some(parse_usize(&after_eq(s), "--steps")?)
                }
                s if s.starts_with("--temperature=") => {
                    temperature = parse_f32(&after_eq(s), "--temperature")?;
                }
                s if s.starts_with("--topp=") => topp = parse_f32(&after_eq(s), "--topp")?,
                s if s.starts_with("--seed=") => seed = Some(parse_u64(&after_eq(s), "--seed")?),
                s if s.starts_with("--group-size=") => {
                    group_size = parse_usize(&after_eq(s), "--group-size")?
                }
                s if s.starts_with("--convert=") => convert = Some(after_eq(s)),
                s if s.starts_with("--to=") => to = Some(parse_target(&after_eq(s))?),
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
            topp,
            seed,
            quantize,
            group_size,
            scalar,
            dotprod,
            convert,
            to,
            help,
        })
    }
}

/// Parse a `--to` conversion target.
fn parse_target(s: &str) -> Result<Target, HostError> {
    match s {
        "v1" => Ok(Target::V1),
        "v2" => Ok(Target::V2),
        _ => Err(HostError::Usage(format!(
            "`--to` expects `v1` or `v2`, got `{s}`"
        ))),
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
    s.parse::<usize>().map_err(|_| {
        HostError::Usage(format!(
            "`{flag}` expects a non-negative integer, got `{s}`"
        ))
    })
}

fn parse_u64(s: &str, flag: &str) -> Result<u64, HostError> {
    s.parse::<u64>().map_err(|_| {
        HostError::Usage(format!(
            "`{flag}` expects a non-negative integer, got `{s}`"
        ))
    })
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
        assert_eq!(a.seed, None);

        let b = parse(&[
            "m.bin",
            "--prompt=hi",
            "--temperature=0.9",
            "--topp=0.8",
            "--seed=42",
        ])
        .unwrap();
        assert_eq!(b.prompt.as_deref(), Some("hi"));
        assert_eq!(b.temperature, 0.9);
        assert_eq!(b.topp, 0.8);
        assert_eq!(b.seed, Some(42));
    }

    #[test]
    fn quantization_flags_parse() {
        // Off by default, with a group size that divides the stories15M dims.
        let def = parse(&["m.bin"]).unwrap();
        assert!(!def.quantize);
        assert_eq!(def.group_size, 32);
        assert!(!def.scalar); // SIMD kernels by default
        assert!(!def.dotprod);

        let a = parse(&["m.bin", "-q", "--group-size", "64", "--scalar"]).unwrap();
        assert!(a.quantize);
        assert_eq!(a.group_size, 64);
        assert!(a.scalar);

        let d = parse(&["m.bin", "-q", "--dotprod"]).unwrap();
        assert!(d.dotprod);
        assert!(!d.scalar);

        let b = parse(&["m.bin", "--quantize", "--group-size=96"]).unwrap();
        assert!(b.quantize);
        assert_eq!(b.group_size, 96);
    }

    #[test]
    fn convert_flags_parse() {
        let def = parse(&["m.bin"]).unwrap();
        assert!(def.convert.is_none());
        assert!(def.to.is_none());

        let a = parse(&["m.bin", "--convert", "out.bin", "--to", "v1"]).unwrap();
        assert_eq!(a.convert.as_deref(), Some("out.bin"));
        assert_eq!(a.to, Some(Target::V1));

        let b = parse(&["m.bin", "--convert=out.bin", "--to=v2"]).unwrap();
        assert_eq!(b.convert.as_deref(), Some("out.bin"));
        assert_eq!(b.to, Some(Target::V2));

        // Anything but v1/v2 is a usage error.
        assert!(matches!(
            parse(&["m.bin", "--convert", "o.bin", "--to", "v3"]),
            Err(HostError::Usage(_))
        ));
    }

    #[test]
    fn bad_steps_value_is_usage_error() {
        assert!(matches!(
            parse(&["m.bin", "-n", "lots"]),
            Err(HostError::Usage(_))
        ));
        assert!(matches!(
            parse(&["m.bin", "--temperature=warm"]),
            Err(HostError::Usage(_))
        ));
        assert!(matches!(
            parse(&["m.bin", "--seed=soon"]),
            Err(HostError::Usage(_))
        ));
        assert!(matches!(
            parse(&["m.bin", "--topp=most"]),
            Err(HostError::Usage(_))
        ));
    }
}
