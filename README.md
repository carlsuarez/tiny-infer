# tiny-infer

A small, from-scratch inference **engine** in Rust — `no_std`, allocation-free, no ML
frameworks, no GPU, no training. The product is the `engine` crate: a library of compute
kernels (fp32 **and** int8 matmul / attention / norms, 1-D CNN ops, an optional real-FFT)
plus the machinery they run on — the bump [`Arena`](#the-engine--the-shared-core) every
working buffer is carved from, the int8 quantization primitives, and a token sampler. The
constraint that defines the project: all work happens out of a single pre-allocated memory
arena, so after init the forward pass allocates nothing and grows nothing.

The engine knows **no model shapes**. Each architecture lives in its own crate built on top
of the engine, and the repo ships two worked examples — both driven by the `tiny-infer`
CLI — that show how to wire a real model onto the core:

* **`llama`** — decoder-only **Llama-2** / TinyStories (RMSNorm, RoPE, grouped-query
  attention, SwiGLU). Greedy output is byte-for-byte identical to a reference C
  implementation at temperature 0.
* **`seq2seq`** — **Marian / OPUS-MT** encoder-decoder (bidirectional encoder,
  cross-attention, LayerNorm, sinusoidal positions). Greedy decode matches Hugging Face
  token-for-token; the OPUS-MT fixtures translate, so `"Hello, world!"` → `Bonjour, le monde !`.

The kernels are reused well beyond transformers: the `edge-pm` project's `pmcore` builds a
1-D CNN vibration classifier directly on the engine's `nn` and `dsp` modules — same arena,
same int8 path — on a Cortex-M microcontroller. That is the point of the split: the engine is
the reusable core, and Llama / seq2seq / edge-pm are just consumers of it.

Binary checkpoint formats follow the widely-used Llama-2 `.bin` layout, so exported
checkpoints load unchanged. The correctness goal, across every example, is token-for-token
parity with a reference implementation at temperature 0.

## Commands at a glance

`tiny-infer …` below is shorthand for `cargo run --release -p host -- …` (or the built
binary `target/release/tiny-infer`). Full details in [Commands](#commands).

| Task | Command |
|------|---------|
| Build everything (release) | `cargo build --release` |
| Generate text (Llama) | `tiny-infer models/stories15M.bin models/tokenizer.bin -p "Once upon a time"` |
| Sample (temperature + top-p) | `tiny-infer … -p "…" --temperature 1.0 --topp 0.9 --seed 42` |
| Run weights as int8 | `tiny-infer … --quantize --group-size 32` (add `--dotprod` for the hardware int8 kernel) |
| Convert a checkpoint | `tiny-infer models/stories15M.bin --convert out.v2.bin --to v2` |
| Inspect a checkpoint (report) | `tiny-infer models/stories15M.bin models/tokenizer.bin` (no `--prompt`) |
| Translate (seq2seq / Marian) | `tiny-infer models/opus-mt-en-fr/model.bin -p "Hello, world!"` |
| Export a seq2seq model | `python scripts/export_marian.py Helsinki-NLP/opus-mt-en-fr -o models/opus-mt-en-fr` |
| Run tests + lints | `cargo test` · `cargo clippy --all-targets` |
| Prove the engine is `no_std` | `cargo build -p engine --target thumbv7em-none-eabi` |

## Status

The engine grew up alongside its first example (the `llama` path) — Milestones 1–3 below
track that bring-up — and the later sections are cross-cutting engine capabilities (int8
quantization, the matmul kernels, the checkpoint formats) shared by every consumer. The
`seq2seq` example was then added as a second, independent architecture on the same core.

**Milestone 1 — parse & report (done).** Loads a checkpoint and tokenizer,
validates them, and prints the model config, file-size validation, tokenizer
metadata, and the pre-computed memory budget for the working arena.

**Milestone 2 — fp32 forward pass + greedy decode (done).** Implements the scalar
math kernels, the `RunState` working set carved from the arena, the full forward
pass (RMSNorm, RoPE, causal grouped-query attention, SwiGLU), and BPE
encode/decode. The CLI now generates text. Greedy (temperature-0) output is
**byte-for-byte identical to a reference C implementation** on stories15M — the
correctness gate (see [Parity](#parity-with-the-reference)).

**Milestone 3 — sampling (done).** Generation supports temperature and
top-p (nucleus) sampling in addition to greedy decode: `--temperature 0` (the
default) stays deterministic/greedy, a higher temperature draws each token from the
softmax distribution, and `--topp <F>` restricts that draw to the most-probable
tokens whose probabilities sum to `F`. `--seed` makes a sampled run reproducible.

**Seq2seq generation (Marian / OPUS-MT) — end to end (done).** A second
architecture, added *alongside* the decoder-only path without touching it (the
Llama parity gate stays bit-identical): bidirectional encoder, cross-attention,
LayerNorm with bias, sinusoidal positions, a tied lm_head with `final_logits_bias`.
The encoder-decoder shape handles any sequence-to-sequence task — translation,
summarization, paraphrasing — depending on the model; the OPUS-MT fixtures used here
happen to translate, so `tiny-infer models/opus-mt-en-fr/model.bin --prompt "Hello,
world!"` prints `Bonjour, le monde !`. The pieces:

* **Format + export** — a versioned header with its own `"tis2"` magic, zero-copy
  weight views, and a `const fn` arena budget (encoder buffers + cross-attention KV
  cache + decoder self-KV cache + per-step scratch). `scripts/export_marian.py` dumps
  a Hugging Face `MarianMTModel` to that format and emits the tokenizer artifact.
* **Encoder + decoder** — allocation-free, arena-backed forward passes built from the
  same `no_std` kernels, running on the scalar, `wide` SIMD, or hardware int8
  dot-product kernels (`--scalar` selects the reference path; SIMD is the default, ~6×
  faster and bit-identical in its greedy output). Two correctness gates against Hugging
  Face transformers: the encoder's `last_hidden_state` matches to ~2e-6, and greedy
  decode matches `generate(num_beams=1, do_sample=False)` **token-for-token**.
* **Tokenizer** — a from-scratch SentencePiece **Unigram** tokenizer (Viterbi over the
  piece-score table) that reproduces `MarianTokenizer` on ordinary text.
* **Int8 quantization** — `--quantize` / `--group-size` / `--dotprod` work for
  seq2seq generation exactly as they do for Llama generation (below): the 17 matmul matrices
  (the tied embedding included) quantize to int8 while biases and LayerNorms stay fp32,
  the fp32 checkpoint is freed at load (~285 → ~76 MiB), and the hardware dot-product
  kernel makes int8 a speed win too (~2.3× over the fp32 SIMD path on the OPUS model).
* **Sampling** — token selection is the shared `engine::Sampler` (greedy / temperature /
  top-p, a `no_std` PRNG and caller-provided nucleus scratch), so `--temperature` /
  `--topp` / `--seed` apply here too. For seq2seq, greedy (the default) — or beam
  search — is the quality path; sampling adds diversity, not accuracy.

The whole architecture is its own crate (`seq2seq`) — it never touches the Llama path, so
it can't disturb the Llama parity gate.

**Int8 quantization (`--quantize`).** All the matmul weights — the seven layer
projections and the token-embedding/classifier table — can be quantized to group-wise
symmetric int8; only the (tiny) RMSNorm gains stay fp32. A single `forward` runs over
either representation via a `llama::ModelWeights` enum. The embedding table is stored
**once** as int8 and the lookup dequantizes just the one row it needs, so it doubles as
the `wcls` classifier under weight tying. On stories15M the int8 greedy output is
identical to fp32.

The host quantizes the fp32 checkpoint in memory and then **frees it**, so
steady-state weight memory on stories15M drops from ~58 MiB to ~17 MiB (generation RSS
~66 → ~21 MiB). Peak RSS is briefly *higher* during quantizing (the fp32 file and int8
buffers are both resident); converting the checkpoint to the pre-quantized **v2**
format once (`--convert`, below) avoids that entirely — no fp32 weights ever load.

The int8 matmul is **W8A8** (full int8): each matmul quantizes its activation to int8
on the fly ([`quantize_activation`]), accumulates the dot product in `i32`, then applies
both scales — the standard group-wise int8 scheme. On stories15M the greedy output still
keeps the golden story opening.

**Matmul kernels — scalar, SIMD, and hardware dot-product.** The matrix–vector products
(the bulk of the work) have three implementations, selected per run via `engine::Kernel`:

- **scalar** — the readable reference (`--scalar`).
- **SIMD** — portable 8-wide lanes via the [`wide`](https://crates.io/crates/wide) crate
  (the default). `wide` is SIMD on **stable** Rust, so the workspace needs no nightly
  toolchain or `core::simd`.
- **dot-product** — for the int8 path, a hardware int8 dot-product kernel via `core::arch`
  intrinsics (`--dotprod`), one per architecture:
  - **x86 AVX-512 VNNI** (`vpdpbusd`): 32 int8 multiply-accumulates into `i32` per
    instruction. It multiplies *unsigned × signed*, so the kernel offsets each weight by
    `+128` in-register and subtracts a per-group `128·Σ(activations)` correction.
  - **ARM NEON `sdot`** (`vdotq_s32`): 16 int8 multiply-accumulates into `i32` per
    instruction. `sdot` multiplies *signed × signed* directly, so there is **no** offset
    correction (and no group-size restriction — a scalar tail covers any remainder).

  Either way the integer dot product comes out **bit-identical** to the scalar kernel. The
  std host runtime-detects the feature and falls back to SIMD when it (or, on x86, a
  non-multiple-of-32 group size) is unavailable. These two kernels are the engine's only
  `unsafe`, each cfg-gated to its target arch.

Measured on stories15M (release, 256 tokens, x86-64 with AVX-512 VNNI):

| kernel       | scalar     | SIMD       | dot-product (`--dotprod`) |
| ---          | ---        | ---        | ---                       |
| fp32         | ~124 tok/s | ~520 tok/s | —                         |
| int8 (W8A8)  | ~231 tok/s | ~556 tok/s | **~835 tok/s** (VNNI)     |

So int8 is both a **memory** win (~3× smaller weights) *and*, with a hardware int8 dot, a
**speed** win: ~835 tok/s is ~1.6× the fp32 SIMD path and ~3.6× the scalar int8 baseline.
(Without a hardware int8 dot — i.e. portable `i32x8` `pmulld` — int8 only matches fp32; the
dedicated `vpdpbusd`/`sdot` instruction is what makes it pull ahead. On a CPU with neither,
`--dotprod` transparently falls back to SIMD.) Greedy output stays coherent and identical
across all kernels. (The ARM `sdot` kernel mirrors the VNNI path and is bit-exact by
construction; throughput is unbenchmarked here, as the dev machine is x86-64 only.)

[`quantize_activation`]: engine/src/quant.rs

**Checkpoint formats — legacy, v1, and v2 (`--convert`).** All three checkpoint
formats load transparently (the header is auto-detected):

| format | header | weights | notes |
| ---    | ---    | ---     | ---   |
| legacy (v0) | 28 B, 7 × `i32` | fp32 | what the stock TinyStories checkpoints use; sign of `vocab_size` encodes classifier sharing |
| v1 | 256 B, magic `ak42` | fp32 | same tensors, norms-first order, no legacy `freq_cis` padding tables |
| v2 | 256 B, magic `ak42` | **int8** (Q8_0) | group-wise int8 with fp32 scales, RMSNorm gains fp32 |

A v2 checkpoint always runs on the int8 path (`--quantize` is implied; the flag is
ignored with a note). The loader de-interleaves the file's per-tensor data/scales
into the engine's flat layout in one linear pass — no fp32 weights ever
materialize, so both load time and peak memory beat quantize-at-load. `--convert
<out> --to <v1|v2>` writes any fp32 checkpoint back out in either format; the v2
writer uses the engine's own quantizer, so a converted file reproduces
`--quantize` **bit-for-bit** (pinned in `host/tests/formats.rs`, alongside the
byte-identical v1 gate). The tokenizer side is equally format-agnostic: any
exported SentencePiece vocabulary works, including ones trained without
byte-fallback tokens (unknown codepoints then encode as `<unk>` instead of
indexing out of bounds).

**Streaming output.** Generation streams to stdout token by token — each decoded
piece is flushed the moment it is produced, so text appears live instead of in
bursts at line boundaries (the stdout lock is a `LineWriter`) or all at once when
piped. A closed reader (e.g. `tiny-infer … | head`) ends the run quietly rather than
erroring. The trailing status line reports prompt (prefill) and generation (decode)
throughput separately, since those two phases scale differently.

**Bare-metal `no_std`.** The engine core compiles against nothing but `core` and
`libm`, allocates nothing after init, and is built to run under `panic = "abort"`.
Operations driven by file input (header parsing, weight carving, arena sizing) are
fallible and return an `EngineError` rather than panicking, so a malformed checkpoint
can never crash a caller; the memory budget is a `const fn` of the config, so a host can
size a `static` arena at compile time. Beyond the build-only library check, a freestanding
firmware example (`llama/examples/baremetal.rs`) supplies its own `#[panic_handler]` and
runs a full forward pass out of stack buffers with no allocator — see
[Bare-metal / `no_std` builds](#bare-metal--no_std-builds-the-embedded-target).

## Repository map

Four crates in a Cargo workspace, plus Python helpers and git-ignored model fixtures.
**`engine` is the reusable core; each model architecture ships as its own example crate
built on it** — the same way an external project (edge-pm's `pmcore`) consumes the engine
from outside this workspace. The two architecture crates never reference each other.

| crate / dir | role |
|-------------|------|
| `engine/`   | `#![no_std]`, allocation-free **shared core** — arena, error, math/quant kernels, the 1-D CNN ops, the optional FFT (`fft` feature), and the sampler. No model shapes. Depends only on `core` + `libm`, `rand`, `wide` (all `no_std`), plus `microfft` when `fft` is enabled. |
| `llama/`    | `#![no_std]` **Llama-2** example model crate (decoder-only). Depends on `engine`. Holds the baremetal example. |
| `seq2seq/`  | `#![no_std]` **Marian / OPUS-MT** example model crate (encoder-decoder). Depends on `engine`. Holds the encoder/decoder parity gates. |
| `host/`     | `std` CLI binary `tiny-infer`: file loading, tokenizers, generation, reporting. Depends on `engine` + `llama` + `seq2seq`; `llama/` and `seq2seq/` driver modules mirror the model crates. |
| `scripts/`  | Python: export a Hugging Face seq2seq model + build its tokenizer, and capture parity references (see [Python scripts](#python-scripts)). |
| `models/`   | downloaded / exported checkpoints + parity fixtures (**git-ignored**; recreate via curl + the scripts). |

### The engine — the shared core

The building blocks that don't depend on any one model shape. Nothing here knows about
Llama or Marian; the model crates build entirely on top of it.

```
engine/src/
  lib.rs           # crate root; re-exports Arena, EngineError, QuantizedTensor, QuantScratch, Kernel, Sampler
  arena.rs         # the bump Arena every working buffer is carved from
  error.rs         # the crate-wide EngineError
  math.rs          # fp32 + int8 compute kernels (matmul, W8A8, norms, attention, RoPE, …)
  nn.rs            # 1-D CNN kernels (conv1d, relu, global_avg_pool) — fp32 + int8 — for feature-window classifiers
  dsp.rs           # real-FFT magnitude spectrum (512-pt, behind the `fft` feature) for spectral feature extraction
  quant.rs         # group-wise int8 quantization primitives (QuantizedTensor, QuantScratch, quantize/dequantize)
  sample.rs        # the Sampler (greedy / temperature / top-p) over rand's no_std Xoshiro128++ PRNG
```

The `nn` and `dsp` modules don't serve the transformer path at all — they exist so a
downstream classifier (edge-pm's `pmcore`) can build a 1-D CNN over windowed sensor
features on the same arena + kernels. `dsp` (real FFT via [`microfft`]) sits behind the
**off-by-default `fft` feature**, so the `llama` / `seq2seq` crates pull in no FFT
dependency; only a crate that asks for it (`edge-pm`) does.

[`microfft`]: https://crates.io/crates/microfft

### The model crates — `llama` and `seq2seq`

Two parallel, self-contained `no_std` crates, each consuming `engine`. Their files play
the same roles, so once you know one you know the other:

```
llama/src/         seq2seq/src/      # role of each file
  config.rs          config.rs       #   header parse, format detection, derived dims
  weights.rs         weights.rs      #   zero-copy fp32 tensor views
  memory.rs          memory.rs       #   const fn arena budget (activations + KV cache)
  state.rs           state.rs        #   RunState — the working set carved once from the arena
  model.rs           model.rs        #   the allocation-free forward / encode / greedy_decode
  quantize.rs        quantize.rs     #   int8 weight layout (on engine::quant primitives)
  lib.rs             lib.rs          #   crate root; re-exports the public types
llama/examples/baremetal.rs          # freestanding #![no_std]/#![no_main] firmware proof
seq2seq/tests/{encode,decode}_parity.rs   # gates vs Hugging Face (fixtures from scripts/dump_*_ref.py)
```

Each crate names its config `Config`, its weights `Weights`, and its working set
`RunState`; the crate path keeps them distinct — `llama::{Config, Weights, forward, …}`
and `seq2seq::{Config, Weights, encode, greedy_decode, …}`.

### The host — the same split behind the CLI

```
host/src/
  main.rs          # thin dispatcher: routes a checkpoint to llama/ or seq2seq/ by its magic
  args.rs          # the hand-rolled flag parser (every --flag the CLI accepts)
  error.rs         # host error type wrapping engine + io errors into clean messages
  llama/           # loader.rs (legacy/v1/v2 + tokenizer.bin), tokenizer.rs (BPE), generate.rs (decode loop),
                   #   run.rs (orchestration), report.rs (inspect mode), convert.rs (--convert v1/v2)
  seq2seq/         # loader.rs, tokenizer.rs (SentencePiece Unigram/Viterbi), generate.rs, run.rs, report.rs
```

### Tests

| file | gate |
|------|------|
| `host/tests/cli.rs` | CLI arg handling (always) + stories15M metadata (when fixtures present) |
| `host/tests/generate.rs` | greedy output pinned to a golden string; sampling reproducible under `--seed` |
| `host/tests/formats.rs` | converted v1/v2 checkpoints generate exactly what the legacy file does |
| `host/tests/seq2seq.rs` | seq2seq loader/report on a synthetic checkpoint + real OPUS-MT when exported |
| `seq2seq/tests/encode_parity.rs` | `seq2seq::encode` == HF `last_hidden_state` (fixture from `dump_encoder_ref.py`) |
| `seq2seq/tests/decode_parity.rs` | `seq2seq::greedy_decode` == HF greedy stream (fixture from `dump_decode_ref.py`) |

Every fixture-dependent test **skips itself on a clean checkout** and runs the real
comparison once the model/fixture is present, so `cargo test` is green either way.

The engine works in units of `f32` — the element type of every activation and
KV-cache buffer — so the arena hands out disjoint `&mut [f32]` slices with no
`unsafe` and no aliasing. The host reinterprets a checkpoint's `f32` storage as bytes
(to read the file in one pass) with a *safe* [`bytemuck`](https://docs.rs/bytemuck)
cast, so neither crate carries any `unsafe` of its own — except the engine's one
cfg-gated x86 block for the AVX-512 VNNI int8 kernel.

## Python scripts

The Rust engine and CLI need **no Python at runtime**. The scripts in `scripts/` are
host-side, offline helpers used only to (a) export a Hugging Face seq2seq model into
tiny-infer's format and (b) capture Hugging Face reference outputs for the engine's
seq2seq parity gates. The Llama path needs none of them — its checkpoint and tokenizer
are downloaded directly (see [Getting the models](#getting-the-models)).

They run in a venv:

```sh
python -m venv venv && source venv/bin/activate
pip install -r scripts/requirements.txt      # torch, transformers, sentencepiece, numpy
```

| script | what it does | writes (under `models/<dir>/`) | needs | used by |
|--------|--------------|-------------------------------|-------|---------|
| **export_marian.py** | Converts a Hugging Face `MarianMTModel` (OPUS-MT) to tiny-infer's `tis2` seq2seq format, and saves the SentencePiece artifacts; calls `dump_tokenizer.py` for you. | `model.bin`, `source.spm`/`target.spm`/`vocab.json`, `tokenizer.bin` | torch, transformers, sentencepiece | the seq2seq CLI; `tests/seq2seq.rs` |
| **dump_tokenizer.py** | Builds the host's `tokenizer.bin` (Unigram piece/score table + id→piece map) from `source.spm` + `vocab.json`. Usually run automatically by `export_marian.py`; run it standalone to rebuild just the tokenizer. | `tokenizer.bin` | sentencepiece (no torch) | the seq2seq tokenizer |
| **dump_encoder_ref.py** | Runs HF's Marian **encoder** on a fixed input and dumps its `last_hidden_state`. | `encoder_ref.bin` | torch, transformers | `seq2seq/tests/encode_parity.rs` |
| **dump_decode_ref.py** | Runs HF **greedy generation** (`num_beams=1, do_sample=False`) on a fixed sentence and dumps the source + generated token ids. | `decode_ref.bin` | torch, transformers | `seq2seq/tests/decode_parity.rs` |

Typical flow — export a translation model, then (optionally) generate the parity
fixtures so the engine's seq2seq gates run instead of skipping:

```sh
python scripts/export_marian.py Helsinki-NLP/opus-mt-en-fr -o models/opus-mt-en-fr
HF_HUB_OFFLINE=1 python scripts/dump_encoder_ref.py    # writes encoder_ref.bin  (after the hub cache is warm)
HF_HUB_OFFLINE=1 python scripts/dump_decode_ref.py     # writes decode_ref.bin
```

The on-disk binary layouts are documented in each script's header and kept in lockstep
with the matching Rust module (`seq2seq/src/` and `host/src/seq2seq/tokenizer.rs`).

## Getting the models

The **Llama** path downloads its checkpoint and tokenizer directly (no scripts):

```sh
mkdir -p models
curl -L -o models/stories15M.bin \
  https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
curl -L -o models/tokenizer.bin \
  https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.bin
```

The **seq2seq** path is produced by `scripts/export_marian.py` (see
[Python scripts](#python-scripts)).

## Commands

These run the `host` CLI — tiny-infer's "simulator": it loads a checkpoint and runs the
engine on the CPU. `tiny-infer …` is shorthand for `cargo run --release -p host -- …`.

```sh
cargo build --release      # build the engine + host CLI (release; needed for real throughput)
```

**Generate text:**

```sh
cargo run --release -p host -- \
  models/stories15M.bin models/tokenizer.bin \
  --prompt "Once upon a time" --steps 80
```

```
Once upon a time, there was a little girl named Lily. She loved to play outside
in the sunshine. ...
[prompt 5 tok in 0.009s (539.4 tok/s), generated 75 tok in 0.655s (114.5 tok/s)]
```

`--steps` defaults to the model's `seq_len`; generation stops early at the BOS
delimiter. `--temperature 0` (the default) is greedy and deterministic; a higher
temperature samples from the softmax distribution for more varied output, and
`--topp` adds nucleus sampling on top (sampling only from the most-probable tokens
whose probabilities sum to the given threshold, which trims the unreliable tail):

```sh
cargo run --release -p host -- \
  models/stories15M.bin models/tokenizer.bin \
  --prompt "Once upon a time" --temperature 1.0 --topp 0.9 --seed 42
```

Pass `--seed <N>` to make a sampled run reproducible (omit it for fresh
randomness each run).

Add `--quantize` to run the matmul weights as int8 (see Status). The optional
`--group-size <N>` must divide the model's `dim`, `hidden_dim`, and `kv_dim`
(default 32; for stories15M, 64 is rejected because 288 is not divisible by 64):

```sh
cargo run --release -p host -- \
  models/stories15M.bin models/tokenizer.bin \
  --prompt "Once upon a time" --quantize --group-size 32
```

The matmul kernels are SIMD (`wide`) by default; add `--scalar` for the readable
reference kernels, or `--dotprod` to run int8 through the hardware dot-product kernel
(x86 AVX-512 VNNI or ARM NEON `sdot`) — the fastest path on a CPU that has one (same
output; see the table above).

**Convert a checkpoint** (`--convert <out>`, target `--to v1` or `--to v2`,
default v2). Converting to v2 quantizes once, on disk — afterwards the file loads
straight into the int8 path with no fp32 step and ~3.6× less disk and memory:

```sh
cargo run --release -p host -- models/stories15M.bin \
  --convert models/stories15M.v2.bin --to v2 --group-size 32
cargo run --release -p host -- \
  models/stories15M.v2.bin models/tokenizer.bin --prompt "Once upon a time"
```

**Encoder-decoder models (Marian / OPUS-MT).** Export a Hugging Face encoder-decoder
(seq2seq) model to tiny-infer's seq2seq format, then run it with `--prompt` (the
tokenizer is found next to the model automatically). The example below uses an OPUS-MT
en→fr translation model, but the same path serves any seq2seq task the model was
trained for:

```sh
pip install -r scripts/requirements.txt      # see Python scripts
python scripts/export_marian.py Helsinki-NLP/opus-mt-en-fr -o models/opus-mt-en-fr
cargo run --release -p host -- models/opus-mt-en-fr/model.bin --prompt "Hello, world!"
# -> Bonjour, le monde !
```

Without `--prompt` it reports the model instead:

```sh
cargo run --release -p host -- models/opus-mt-en-fr/model.bin
```

```
Model: models/opus-mt-en-fr/model.bin
  format          tiny-infer seq2seq v1 (Marian encoder-decoder), fp32 weights
  d_model         512
  encoder         6 layers, 8 heads (head_dim 64), ffn 2048
  decoder         6 layers, 8 heads (head_dim 64), ffn 2048
  ...
Memory budget (working arena at max_src=512 / max_tgt=512):
  cross-KV cache   12.00 MiB  (3,145,728 f32)
  self-KV cache    12.00 MiB  (3,145,728 f32)
  arena total      28.26 MiB  (7,408,250 f32)
```

**Inspect a checkpoint** (no `--prompt` → report mode; the `format` line shows
what was detected — `legacy (v0)`, `v1`, or `v2` with its group size):

```sh
cargo run --release -p host -- models/stories15M.bin models/tokenizer.bin
```

```
Model: models/stories15M.bin
  status         OK (file size matches config)
  dim            288
  ...
Memory budget (working arena, pre-allocated once):
  activations      141.50 KiB  (36,224 f32)
  KV cache           3.38 MiB  (884,736 f32)
  arena total        3.51 MiB  (920,960 f32)
  weights (disk)    58.00 MiB

Weights + peak RAM (fp32 vs int8, group_size 32):
                      fp32         int8
  weights          57.95 MiB    16.31 MiB   (3.6× smaller)
  peak RAM         61.46 MiB    19.82 MiB   (weights + arena)
```

## Parity with the reference

At temperature 0 the token stream must match a reference C implementation of the
same model exactly. To reproduce, run the reference binary (built as `run`) and this
engine on the same checkpoint and prompt:

```sh
# reference C implementation, compiled to `run`
./run stories15M.bin -z tokenizer.bin -t 0 -n 40 -i "Tom went to the park"

# this engine
tiny-infer stories15M.bin tokenizer.bin -p "Tom went to the park" -n 40
```

Both print identical text. The golden output is also pinned in
`host/tests/generate.rs`, so the parity gate runs with no external reference present.

## Tests & checks

```sh
cargo test            # unit tests (engine + host) and CLI/generation integration tests
cargo clippy --all-targets
```

`cargo test` is green on a clean checkout: the fixture-dependent gates skip themselves
until the models/fixtures exist (see [Tests](#tests) for the full matrix and which
script produces each fixture).

### Bare-metal / `no_std` builds (the embedded target)

The `engine` core and both model crates (`llama`, `seq2seq`) are genuinely `no_std`;
verify by building them for a bare-metal target with no `std` available:

```sh
cargo build -p engine -p llama -p seq2seq --target thumbv7em-none-eabi
```

That builds the libraries, which borrow the host's panic handler and allocator. To
prove they stand on their own, the `baremetal` example (in the `llama` crate) is a
freestanding `#![no_std]` / `#![no_main]` firmware binary that supplies its own
`#[panic_handler]` and runs a full forward pass entirely out of stack buffers — no
heap, no allocator, no `std`:

```sh
cargo build -p llama --example baremetal --target thumbv7em-none-eabi
```

(The same file builds as an ordinary example on a hosted target, where its `main` just
points you at the command above. Because every transcendental routes through `libm`
rather than `std`'s `f32` methods, the bare-metal build also fails fast if a `std`-only
float intrinsic ever slips in.)
