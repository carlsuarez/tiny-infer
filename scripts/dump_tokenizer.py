#!/usr/bin/env python3
"""Build tiny-infer's seq2seq tokenizer artifact (`tokenizer.bin`) from a model dir.

Reads the SentencePiece *source* model (`source.spm`) and the shared vocabulary
(`vocab.json`) that `export_marian.py` saves, and writes `<dir>/tokenizer.bin` — the
file the host's Marian tokenizer loads. Encoding reproduces SentencePiece **Unigram**:
the host normalizes whitespace, runs Viterbi over the `(piece, score)` table, maps
the chosen pieces to ids, and appends eos. Decoding inverts the id→piece map and
joins, turning `▁` back into spaces.

Only the *source* model is needed: segmentation runs on the source (English) side,
and decoding the (French) target is a plain id→piece join (no re-segmentation). The
ids come from the shared `vocab.json`, the model's real vocabulary, not the spm's
internal ids.

Layout (little-endian) — kept in lockstep with `host/src/seq2seq/tokenizer.rs`:

    magic "tit1" | version i32 (=1) |
    i32 vocab_size | i32 n_src_pieces | i32 unk_id | i32 eos_id | i32 pad_id
    n_src_pieces × { i32 model_id | f32 score | u32 len | len utf8 bytes }   # encode
    vocab_size  × { u32 len | len utf8 bytes }   # decode: piece for id 0,1,…

Needs sentencepiece (no torch). Run standalone, or it is called automatically by
export_marian.py:

    venv/bin/python scripts/dump_tokenizer.py models/opus-mt-en-fr
"""
import argparse
import json
import pathlib
import struct
import sys

import sentencepiece as spm

MAGIC = b"tit1"
VERSION = 1


def build(model_dir) -> pathlib.Path:
    d = pathlib.Path(model_dir)
    vocab = json.loads((d / "vocab.json").read_text())
    inv = {i: s for s, i in vocab.items()}
    vocab_size = max(vocab.values()) + 1
    unk_id, eos_id, pad_id = vocab["<unk>"], vocab["</s>"], vocab["<pad>"]

    sp = spm.SentencePieceProcessor()
    sp.Load(str(d / "source.spm"))
    # Real, segmentable pieces only: drop control/unknown/unused symbols so a literal
    # "</s>" in the input can never be matched as a piece.
    pieces = []
    for i in range(sp.GetPieceSize()):
        if sp.IsControl(i) or sp.IsUnknown(i) or sp.IsUnused(i):
            continue
        piece = sp.IdToPiece(i)
        pieces.append((piece, sp.GetScore(i), vocab.get(piece, unk_id)))

    out = bytearray(MAGIC + struct.pack("<i", VERSION))
    out += struct.pack("<iiiii", vocab_size, len(pieces), unk_id, eos_id, pad_id)
    for piece, score, model_id in pieces:
        b = piece.encode("utf-8")
        out += struct.pack("<ifI", model_id, score, len(b)) + b
    for i in range(vocab_size):
        b = inv.get(i, "").encode("utf-8")
        out += struct.pack("<I", len(b)) + b

    path = d / "tokenizer.bin"
    path.write_bytes(out)
    print(
        f"wrote {path} ({len(out) / 2**20:.2f} MiB): "
        f"{len(pieces):,} source pieces, vocab {vocab_size:,}, "
        f"unk/eos/pad {unk_id}/{eos_id}/{pad_id}"
    )
    return path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("dir", help="model dir holding source.spm + vocab.json")
    build(p.parse_args().dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
