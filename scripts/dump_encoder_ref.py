#!/usr/bin/env python3
"""Dump a Hugging Face Marian encoder's `last_hidden_state` for a fixed input.

Produces the fixture that `engine/tests/encode_parity.rs` checks the engine's
`encode` against (the M11 correctness gate). The engine is fed the *same* token
ids HF used, so this isolates the encoder math from any tokenizer differences.

Usage:
    venv/bin/python scripts/dump_encoder_ref.py [out_dir] [text]

Binary layout of `<out_dir>/encoder_ref.bin` (all little-endian):
    u32 src_len | u32 d_model | src_len * u32 token_ids | src_len*d_model * f32
"""

import struct
import sys

import torch
from transformers import AutoTokenizer, MarianMTModel

MODEL_ID = "Helsinki-NLP/opus-mt-en-fr"


def main() -> None:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "models/opus-mt-en-fr"
    text = sys.argv[2] if len(sys.argv) > 2 else "Hello, world! This is a small test."

    model = MarianMTModel.from_pretrained(MODEL_ID).eval()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    ids = tok(text, return_tensors="pt").input_ids

    with torch.no_grad():
        hidden = model.get_encoder()(input_ids=ids).last_hidden_state[0]  # [src_len, d]

    id_list = ids[0].tolist()
    src_len, d_model = hidden.shape
    path = f"{out_dir}/encoder_ref.bin"
    with open(path, "wb") as f:
        f.write(struct.pack("<II", src_len, d_model))
        f.write(struct.pack(f"<{src_len}I", *id_list))
        f.write(hidden.numpy().astype("<f4").tobytes())

    print(f"wrote {path}: src_len={src_len}, d_model={d_model}")
    print(f"  token ids: {id_list}")
    print(f"  hidden[0,:5] = {[round(x, 5) for x in hidden[0, :5].tolist()]}")


if __name__ == "__main__":
    main()
