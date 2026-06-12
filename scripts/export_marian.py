#!/usr/bin/env python3
"""Export a Hugging Face MarianMTModel (OPUS-MT) to tiny-infer's seq2seq format.

Writes `<out>/model.bin` — the binary the Rust engine loads — plus the model's
SentencePiece tokenizer artifacts (source.spm / target.spm / vocab.json) into the
same directory for the upcoming tokenizer milestone.

Usage:
    python scripts/export_marian.py Helsinki-NLP/opus-mt-en-fr -o models/opus-mt-en-fr

Requires: torch, transformers (Marian tensor names are stable across recent
versions; developed against transformers 4.x), sentencepiece (for the tokenizer
files).

On-disk format (must stay in lockstep with `engine/src/seq2seq/`):

  header — 256 bytes:
    magic "tis2" | version i32 (=1) |
    13 × i32: d_model, enc_layers, dec_layers, enc_heads, dec_heads,
              enc_ffn, dec_ffn, vocab_size, max_src, max_tgt,
              pad_id, eos_id, bos_id |
    3 × u8: norm_before, activation (0=gelu, 1=swish), scale_embedding |
    zero padding to 256

  payload — raw little-endian fp32, each named tensor flat across layers
  (PyTorch nn.Linear storage is already the engine's row-major [d_out, d_in]):
    token_embedding [vocab, d]
    encoder: wq*, bq*, wk*, bk*, wv*, bv*, wo*, bo*,
             ln_att_w*, ln_att_b*, fc1_w*, fc1_b*, fc2_w*, fc2_b*,
             ln_ffn_w*, ln_ffn_b*          (* = all layers contiguously)
    decoder: self {wq..bo}, ln_self_w/b, cross {wq..bo}, ln_cross_w/b,
             fc1_w/b, fc2_w/b, ln_ffn_w/b
    final_logits_bias [vocab]

  Sinusoidal position embeddings are NOT stored (the engine computes them),
  and the lm_head is NOT stored (it is tied to token_embedding; asserted here).
"""

import argparse
import pathlib
import struct
import sys

import torch
from transformers import AutoTokenizer, MarianMTModel

MAGIC = b"tis2"
VERSION = 1
HEADER_BYTES = 256
ACTIVATION_CODES = {"gelu": 0, "swish": 1, "silu": 1}


def fail(msg: str) -> None:
    sys.exit(f"error: {msg}")


def check_exportable(model, cfg) -> None:
    """Assert every structural property the engine's seq2seq path relies on."""
    if not getattr(cfg, "static_position_embeddings", True):
        fail(
            "model uses learned position embeddings; the engine computes sinusoidal ones"
        )
    if cfg.activation_function not in ACTIVATION_CODES:
        fail(
            f"unsupported activation_function {cfg.activation_function!r} "
            f"(supported: {sorted(ACTIVATION_CODES)})"
        )
    if not getattr(cfg, "share_encoder_decoder_embeddings", True):
        fail(
            "model has separate source/target embeddings; the format stores one shared table"
        )

    m = model.model
    shared = m.shared.weight
    if not torch.equal(m.encoder.embed_tokens.weight, shared) or not torch.equal(
        m.decoder.embed_tokens.weight, shared
    ):
        fail("encoder/decoder embeddings are not tied to the shared table")
    if not torch.equal(model.lm_head.weight, shared):
        fail("lm_head is not tied to the shared embedding table")
    if shared.shape[0] != cfg.vocab_size:
        fail(
            f"embedding rows ({shared.shape[0]}) != config vocab_size ({cfg.vocab_size})"
        )
    if cfg.decoder_start_token_id != cfg.pad_token_id:
        fail(
            "decoder_start_token_id != pad_token_id; the engine assumes Marian's "
            "pad-seeded decoding"
        )
    for ids in (cfg.pad_token_id, cfg.eos_token_id, cfg.bos_token_id or 0):
        if not 0 <= ids < cfg.vocab_size:
            fail(f"special token id {ids} outside vocabulary (size {cfg.vocab_size})")


def header(cfg) -> bytes:
    fields = [
        cfg.d_model,
        cfg.encoder_layers,
        cfg.decoder_layers,
        cfg.encoder_attention_heads,
        cfg.decoder_attention_heads,
        cfg.encoder_ffn_dim,
        cfg.decoder_ffn_dim,
        cfg.vocab_size,
        cfg.max_position_embeddings,  # max_src
        cfg.max_position_embeddings,  # max_tgt
        cfg.pad_token_id,
        cfg.eos_token_id,
        cfg.bos_token_id or 0,
    ]
    out = MAGIC + struct.pack("<i", VERSION) + struct.pack("<13i", *fields)
    # HF Marian models are post-norm and MarianConfig has no normalize_before
    # field; read it defensively so a pre-norm export is recorded if one appears.
    out += bytes(
        [
            int(getattr(cfg, "normalize_before", False)),
            ACTIVATION_CODES[cfg.activation_function],
            int(cfg.scale_embedding),
        ]
    )
    out += b"\0" * (HEADER_BYTES - len(out))
    assert len(out) == HEADER_BYTES
    return out


def attention_tensors(layers, attn: str):
    """One attention block's tensors in disk order: wq*, bq*, wk*, bk*, ... bo*."""
    for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
        for kind in ("weight", "bias"):
            for layer in layers:
                yield getattr(getattr(getattr(layer, attn), proj), kind)


def layernorm_tensors(layers, norm: str):
    for kind in ("weight", "bias"):
        for layer in layers:
            yield getattr(getattr(layer, norm), kind)


def ffn_tensors(layers):
    for fc in ("fc1", "fc2"):
        for kind in ("weight", "bias"):
            for layer in layers:
                yield getattr(getattr(layer, fc), kind)


def tensors_in_disk_order(model):
    m = model.model
    yield m.shared.weight

    enc = m.encoder.layers
    yield from attention_tensors(enc, "self_attn")
    yield from layernorm_tensors(enc, "self_attn_layer_norm")
    yield from ffn_tensors(enc)
    yield from layernorm_tensors(enc, "final_layer_norm")

    dec = m.decoder.layers
    yield from attention_tensors(dec, "self_attn")
    yield from layernorm_tensors(dec, "self_attn_layer_norm")
    yield from attention_tensors(dec, "encoder_attn")
    yield from layernorm_tensors(dec, "encoder_attn_layer_norm")
    yield from ffn_tensors(dec)
    yield from layernorm_tensors(dec, "final_layer_norm")

    yield model.final_logits_bias  # buffer, shape [1, vocab]


def expected_floats(cfg) -> int:
    """The engine's `seq2seq_weight_floats`, kept in lockstep for the size check."""
    d = cfg.d_model
    att = 4 * d * d + 4 * d + 2 * d
    enc_ffn = (
        cfg.encoder_ffn_dim * d
        + cfg.encoder_ffn_dim
        + d * cfg.encoder_ffn_dim
        + d
        + 2 * d
    )
    dec_ffn = (
        cfg.decoder_ffn_dim * d
        + cfg.decoder_ffn_dim
        + d * cfg.decoder_ffn_dim
        + d
        + 2 * d
    )
    return (
        cfg.vocab_size * d
        + cfg.encoder_layers * (att + enc_ffn)
        + cfg.decoder_layers * (2 * att + dec_ffn)
        + cfg.vocab_size
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "model",
        help="Hugging Face model id or local path, e.g. Helsinki-NLP/opus-mt-en-fr",
    )
    p.add_argument(
        "-o", "--out", help="output directory (default: models/<model name>)"
    )
    args = p.parse_args()

    outdir = pathlib.Path(
        args.out or pathlib.Path("models") / args.model.split("/")[-1]
    )
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.model} ...")
    model = MarianMTModel.from_pretrained(args.model)
    model.eval()
    cfg = model.config
    check_exportable(model, cfg)

    bin_path = outdir / "model.bin"
    n_floats = 0
    with open(bin_path, "wb") as f:
        f.write(header(cfg))
        for t in tensors_in_disk_order(model):
            data = t.detach().cpu().contiguous().to(torch.float32).numpy().astype("<f4")
            n_floats += data.size
            f.write(data.tobytes())

    want = expected_floats(cfg)
    if n_floats != want:
        fail(
            f"wrote {n_floats} floats but the format implies {want}; "
            "layout drifted from engine/src/seq2seq/weights.rs"
        )
    size = bin_path.stat().st_size
    assert size == HEADER_BYTES + 4 * n_floats

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.save_pretrained(outdir)

    print(f"wrote {bin_path} ({size / 2**20:.1f} MiB, {n_floats:,} fp32)")
    print(
        f"  d_model {cfg.d_model}, encoder {cfg.encoder_layers}x{cfg.encoder_attention_heads}h "
        f"ffn {cfg.encoder_ffn_dim}, decoder {cfg.decoder_layers}x{cfg.decoder_attention_heads}h "
        f"ffn {cfg.decoder_ffn_dim}, vocab {cfg.vocab_size}"
    )
    print(
        f"  pad/eos/bos {cfg.pad_token_id}/{cfg.eos_token_id}/{cfg.bos_token_id or 0}, "
        f"activation {cfg.activation_function}, scale_embedding {cfg.scale_embedding}"
    )
    print(f"tokenizer artifacts saved to {outdir}/")
    print(f"verify with: cargo run --release -p host -- {bin_path}")


if __name__ == "__main__":
    main()
