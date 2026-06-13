# tiny-infer

An embedded-style transformer inference engine in Rust. It runs small
Llama-2-style (TinyStories) language models on the CPU from scratch — no ML
frameworks, no GPU, no training. The engine core is `no_std` and does all of its
work out of a single pre-allocated memory arena, the constraint that defines the
project: after init, the forward pass allocates nothing and grows nothing.

Binary file formats mirror [Andrej Karpathy's llama2.c](https://github.com/karpathy/llama2.c),
so exported checkpoints load unchanged. The correctness goal is token-for-token
parity with llama2.c at temperature 0.

## Status

**Milestone 1 — parse & report (done).** Loads a checkpoint and tokenizer,
validates them, and prints the model config, file-size validation, tokenizer
metadata, and the pre-computed memory budget for the working arena.

**Milestone 2 — fp32 forward pass + greedy decode (done).** Implements the scalar
math kernels, the `RunState` working set carved from the arena, the full forward
pass (RMSNorm, RoPE, causal grouped-query attention, SwiGLU), and BPE
encode/decode. The CLI now generates text. Greedy (temperature-0) output is
**byte-for-byte identical to llama2.c `run.c`** on stories15M — the correctness
gate (see [Parity](#parity-with-llama2c)).

**Milestone 3 — sampling (in progress).** Generation now supports temperature and
top-p (nucleus) sampling in addition to greedy decode: `--temperature 0` (the
default) stays deterministic/greedy, a higher temperature draws each token from the
softmax distribution, and `--topp <F>` restricts that draw to the most-probable
tokens whose probabilities sum to `F`. `--seed` makes a sampled run reproducible.

**Seq2seq translation (Marian / OPUS-MT) — end to end (done).** A second
architecture, added *alongside* the decoder-only path without touching it (the
llama2.c parity gate stays bit-identical): bidirectional encoder, cross-attention,
LayerNorm with bias, sinusoidal positions, a tied lm_head with `final_logits_bias`.
`tiny-infer models/opus-mt-en-fr/model.bin --prompt "Hello, world!"` prints
`Bonjour, le monde !`. The pieces:

* **Format + export** — a versioned header with its own `"tis2"` magic, zero-copy
  weight views, and a `const fn` arena budget (encoder buffers + cross-attention KV
  cache + decoder self-KV cache + per-step scratch). `scripts/export_marian.py` dumps
  a Hugging Face `MarianMTModel` to that format and emits the tokenizer artifact.
* **Encoder + decoder** — allocation-free, arena-backed forward passes built from the
  same `no_std` kernels, running on the scalar, `core::simd`, or hardware int8
  dot-product kernels (`--scalar` selects the reference path; SIMD is the default, ~6×
  faster and bit-identical in its greedy output). Two correctness gates against Hugging
  Face transformers: the encoder's `last_hidden_state` matches to ~2e-6, and greedy
  decode matches `generate(num_beams=1, do_sample=False)` **token-for-token**.
* **Tokenizer** — a from-scratch SentencePiece **Unigram** tokenizer (Viterbi over the
  piece-score table) that reproduces `MarianTokenizer` on ordinary text.
* **Int8 quantization** — `--quantize` / `--group-size` / `--dotprod` work for
  translation exactly as they do for Llama generation (below): the 17 matmul matrices
  (the tied embedding included) quantize to int8 while biases and LayerNorms stay fp32,
  the fp32 checkpoint is freed at load (~285 → ~76 MiB), and the hardware dot-product
  kernel makes int8 a speed win too (~2.3× over the fp32 SIMD path on the OPUS model).

The whole architecture sits behind an on-by-default `seq2seq` cargo feature, so a
decoder-only embedded build (`--no-default-features`) carries none of it.

**Int8 quantization (`--quantize`).** All the matmul weights — the seven layer
projections and the token-embedding/classifier table — can be quantized to group-wise
symmetric int8; only the (tiny) RMSNorm gains stay fp32. A single `forward` runs over
either representation via an `engine::ModelWeights` enum. The embedding table is stored
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
both scales — the scheme in llama2.c `runq.c`. On stories15M the greedy output still
keeps the golden story opening.

**Matmul kernels — scalar, SIMD, and hardware dot-product.** The matrix–vector products
(the bulk of the work) have three implementations, selected per run via `engine::Kernel`:

- **scalar** — the readable reference (`--scalar`).
- **SIMD** — `core::simd` / `portable_simd`, 8-wide lanes (the default). Because
  `core::simd` is nightly-only, the workspace pins a nightly toolchain (`rust-toolchain.toml`).
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

**Checkpoint formats — legacy, v1, and v2 (`--convert`).** All three llama2.c
export formats load transparently (the header is auto-detected):

| format | header | weights | notes |
| ---    | ---    | ---     | ---   |
| legacy (v0) | 28 B, 7 × `i32` | fp32 | what `export.py --version 0` and the tinyllamas checkpoints use; sign of `vocab_size` encodes classifier sharing |
| v1 | 256 B, magic `ak42` | fp32 | same tensors, norms-first order, no legacy `freq_cis` padding tables |
| v2 | 256 B, magic `ak42` | **int8** (Q8_0) | `runq.c`'s format: group-wise int8 with fp32 scales, RMSNorm gains fp32 |

A v2 checkpoint always runs on the int8 path (`--quantize` is implied; the flag is
ignored with a note). The loader de-interleaves the file's per-tensor data/scales
into the engine's flat layout in one linear pass — no fp32 weights ever
materialize, so both load time and peak memory beat quantize-at-load. `--convert
<out> --to <v1|v2>` writes any fp32 checkpoint back out in either format; the v2
writer uses the engine's own quantizer, so a converted file reproduces
`--quantize` **bit-for-bit** (pinned in `host/tests/formats.rs`, alongside the
byte-identical v1 gate). The tokenizer side is equally format-agnostic: any
llama2.c-exported SentencePiece vocabulary works, including ones trained without
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
firmware example (`engine/examples/baremetal.rs`) supplies its own `#[panic_handler]` and
runs a full forward pass out of stack buffers with no allocator — see [Test](#test).

## Workspace layout

| crate / dir | role |
|-------------|------|
| `engine/`   | `#![no_std]`, allocation-free inference core. Depends only on `core` (+ `libm`). |
| `host/`     | `std` CLI binary `tiny-infer`: file loading, tokenizers, generation/translation, argument handling, reporting. Split into `llama/` and `seq2seq/` modules mirroring the engine. |
| `host/tests/` | end-to-end CLI tests (metadata + parity assertions run against the real fixtures when present). |
| `scripts/`  | `export_marian.py` converts a Hugging Face `MarianMTModel` (OPUS-MT) to tiny-infer's seq2seq format; `dump_tokenizer.py` builds the tokenizer artifact; `dump_*_ref.py` capture Hugging Face parity references. |
| `models/`   | downloaded checkpoints (git-ignored). |

### Inside the engine: a shared core plus two architectures

The engine's module tree makes the architecture split explicit. A small **shared
core** holds the building blocks that don't depend on any one model shape, and two
**parallel, self-contained architecture modules** sit on top of it — they never
reference each other, meeting only at the shared core.

```
engine/src/
  lib.rs           # crate root; re-exports the shared core (Arena, EngineError, QuantizedTensor, QuantScratch, Kernel)
  arena.rs         # shared: the bump Arena every working buffer is carved from
  error.rs         # shared: the crate-wide EngineError
  math.rs          # shared: fp32 + int8 compute kernels (matmul, W8A8, norms, attention, RoPE, …)
  quant.rs         # shared: group-wise int8 quantization primitives (QuantizedTensor, QuantScratch, quantize/dequantize)
  llama/           # decoder-only Llama-2 (llama2.c-compatible): RMSNorm, RoPE, SwiGLU, causal GQA
    config.rs weights.rs memory.rs state.rs model.rs quantize.rs
  seq2seq/         # encoder-decoder Marian / OPUS-MT: LayerNorm+bias, sinusoidal positions, cross-attention
    config.rs weights.rs memory.rs state.rs model.rs quantize.rs
```

Each architecture's public types are reached through its module path —
`engine::llama::{Config, Weights, forward, …}` and `engine::seq2seq::{Config, Weights,
encode, greedy_decode, …}` — so the two never collide and the directory layout *is*
the API surface. The `llama/` path is always compiled in; the `seq2seq/` path sits
behind the on-by-default `seq2seq` feature, and `cargo build -p engine
--no-default-features` drops it for a leaner decoder-only embedded build. Both are
`#![no_std]`, allocation-free, and arena-backed.

The host mirrors that split: `host/src/{llama,seq2seq}/` each own their loader,
tokenizer, driver (generate / translate), and report code, over a small shared base
(`error`, `args`), with `main.rs` a thin dispatcher that routes a checkpoint to
the right module by its magic.

The engine works in units of `f32` — the element type of every activation and
KV-cache buffer — so the arena hands out disjoint `&mut [f32]` slices with no
`unsafe` and no aliasing. The host reinterprets a checkpoint's `f32` storage as bytes
(to read the file in one pass) with a *safe* [`bytemuck`](https://docs.rs/bytemuck)
cast, so neither crate carries any `unsafe` of its own — except the engine's one
cfg-gated x86 block for the AVX-512 VNNI int8 kernel.

## Getting the model

```sh
mkdir -p models
curl -L -o models/stories15M.bin \
  https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
curl -L -o models/tokenizer.bin \
  https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.bin
```

## Build & run

```sh
cargo build --release
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

The matmul kernels are SIMD (`core::simd`) by default; add `--scalar` for the readable
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

**Translation models (Marian / OPUS-MT).** Export a Hugging Face encoder-decoder
translation model to tiny-infer's seq2seq format, then translate with `--prompt`
(the tokenizer is found next to the model automatically):

```sh
pip install torch transformers sentencepiece
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

## Parity with llama2.c

At temperature 0 the token stream must match the reference exactly. To reproduce:

```sh
# reference (in a checkout of github.com/karpathy/llama2.c)
gcc -O3 -o run run.c -lm
./run stories15M.bin -z tokenizer.bin -t 0 -n 40 -i "Tom went to the park"

# this engine
tiny-infer stories15M.bin tokenizer.bin -p "Tom went to the park" -n 40
```

Both print identical text. The golden output is also pinned in
`host/tests/generate.rs`.

## Test

```sh
cargo test            # unit tests (engine + host) and CLI/generation integration tests
cargo clippy --all-targets
```

The `engine` crate is genuinely `no_std`; verify by building it for a bare-metal
target with no `std` available:

```sh
cargo build -p engine --target thumbv7em-none-eabi
```

That builds the library, which borrows the host's panic handler and allocator. To
prove the engine stands on its own, the `baremetal` example is a freestanding
`#![no_std]` / `#![no_main]` firmware binary that supplies its own `#[panic_handler]`
and runs a full forward pass entirely out of stack buffers — no heap, no allocator, no
`std`:

```sh
cargo build -p engine --example baremetal --target thumbv7em-none-eabi
```

(The same file builds as an ordinary example on a hosted target, where its `main` just
points you at the command above. Because every transcendental routes through `libm`
rather than `std`'s `f32` methods, the bare-metal build also fails fast if a `std`-only
float intrinsic ever slips into the engine.)
