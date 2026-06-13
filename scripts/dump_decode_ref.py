#!/usr/bin/env python3
"""Dump a greedy-decode reference for the engine's seq2seq decoder parity gate.

Tokenizes a fixed English sentence with the OPUS-MT tokenizer, runs Hugging Face
greedy generation (``num_beams=1, do_sample=False``), and writes the source token
ids and the generated target ids to ``models/opus-mt-en-fr/decode_ref.bin``. The
engine test ``engine/tests/decode_parity.rs`` feeds the *same* source ids to its
own ``greedy_decode`` and asserts the generated id sequence matches — isolating the
decoder math from the tokenizer (a separate concern).

Binary layout (little-endian):
    u32 n_src | u32 n_gen | n_src × u32 src_ids | n_gen × u32 gen_ids
where ``gen_ids`` is HF's raw output: it begins with the decoder-start token
(``pad_id``) and ends with ``eos``.

Needs torch + transformers (run inside the project venv). The model weights load
from the hub id (the local export dir holds only our ``model.bin``); the tokenizer
loads from the local export. Run offline once the hub cache is warm:

    HF_HUB_OFFLINE=1 venv/bin/python scripts/dump_decode_ref.py
"""
import struct
import sys

import torch
from transformers import MarianMTModel, MarianTokenizer

MODEL_DIR = "models/opus-mt-en-fr"
HUB_ID = "Helsinki-NLP/opus-mt-en-fr"
SENTENCE = "This is a small test."


def main() -> int:
    tok = MarianTokenizer.from_pretrained(MODEL_DIR)
    model = MarianMTModel.from_pretrained(HUB_ID)
    model.eval()

    src_ids = tok(SENTENCE)["input_ids"]
    with torch.no_grad():
        out = model.generate(
            torch.tensor([src_ids]), num_beams=1, do_sample=False, max_length=64
        )
    gen_ids = out[0].tolist()

    path = f"{MODEL_DIR}/decode_ref.bin"
    with open(path, "wb") as f:
        f.write(struct.pack("<II", len(src_ids), len(gen_ids)))
        f.write(struct.pack(f"<{len(src_ids)}I", *src_ids))
        f.write(struct.pack(f"<{len(gen_ids)}I", *gen_ids))

    print(f"wrote {path}")
    print(f"  sentence : {SENTENCE!r}")
    print(f"  src_ids  : {src_ids}")
    print(f"  gen_ids  : {gen_ids}  (leading decoder_start={gen_ids[0]}, trailing eos)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
