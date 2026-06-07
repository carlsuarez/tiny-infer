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

Upcoming: temperature / top-p sampling + a streaming CLI, then `no_std` hardening
polish, then optional int8 / SIMD.

## Workspace layout

| crate / dir | role |
|-------------|------|
| `engine/`   | `#![no_std]` core: `Config` parsing, zero-copy `Weights` views, the bump `Arena`, and `memory` budget math. Depends only on `core`. |
| `host/`     | `std` CLI binary `tiny-infer`: file loading, tokenizer parsing, argument handling, and the reporting output. |
| `host/tests/` | end-to-end CLI tests (metadata assertions run against the real fixtures when present). |
| `models/`   | downloaded checkpoints (git-ignored). |

The engine works in units of `f32` — the element type of every activation and
KV-cache buffer — so the arena hands out disjoint `&mut [f32]` slices with no
`unsafe` and no aliasing. The single audited `unsafe` cast lives in
`host/src/loader.rs`, reinterpreting the loaded file as `f32`.

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

**Generate text** (greedy / temperature 0):

```sh
cargo run --release -p host -- \
  models/stories15M.bin models/tokenizer.bin \
  --prompt "Once upon a time" --steps 80
```

```
Once upon a time, there was a little girl named Lily. She loved to play outside
in the sunshine. ...
[80 tokens, 0.699s, 114.5 tok/s]
```

`--steps` defaults to the model's `seq_len`; generation stops early at the BOS
delimiter. Only `--temperature 0` (greedy) is supported so far; sampling arrives
in the next milestone.

**Inspect a checkpoint** (no `--prompt` → report mode):

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
  arena total        3.52 MiB  (920,960 f32)
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
