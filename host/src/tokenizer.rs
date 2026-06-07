//! Tokenizer vocabulary loading and BPE encode/decode (llama2.c `tokenizer.bin`).
//!
//! Layout: a leading `i32 max_token_length`, then `vocab_size` entries, each
//! `f32 score`, `i32 len`, `len` raw bytes. `vocab_size` is *not* stored in the
//! file — it comes from the model config — so the loader is told how many entries
//! to expect and verifies the file ends exactly there.
//!
//! [`Vocab::encode`] and [`Vocab::decode`] implement the same SentencePiece-style
//! BPE as run.c, so prompts tokenize identically and generation can be checked for
//! parity. In this checkpoint's vocabulary the SentencePiece space marker has been
//! exported as an ASCII space (`0x20`), ids `3..=258` are the byte-fallback tokens
//! `<0x00>`..`<0xFF>`, and id `1` (`<s>`) is BOS.

use std::collections::HashMap;
use std::path::Path;

use crate::error::HostError;

/// BOS token id (`<s>`). Also the sequence delimiter that decoding/generation stop on.
pub const BOS_ID: usize = 1;

/// Byte-fallback tokens occupy ids `3..=258`, so raw byte `b` maps to id `b + 3`.
const BYTE_TOKEN_OFFSET: usize = 3;

/// One vocabulary entry: its merge score and raw byte string.
#[derive(Debug, Clone)]
pub struct TokenEntry {
    /// Merge priority used by the BPE encoder (later milestone).
    pub score: f32,
    /// Raw bytes of the token piece (not necessarily valid UTF-8).
    pub bytes: Vec<u8>,
}

impl TokenEntry {
    /// Render the piece for display: the SentencePiece space marker `▁`
    /// (`U+2581`) becomes a visible `_`, and invalid UTF-8 is shown lossily.
    pub fn display(&self) -> String {
        String::from_utf8_lossy(&self.bytes).replace('\u{2581}', "_")
    }

    /// Whether this is a byte-fallback token of the form `<0xNN>`.
    pub fn is_byte_fallback(&self) -> bool {
        self.bytes.len() == 6
            && self.bytes.starts_with(b"<0x")
            && self.bytes.ends_with(b">")
    }

    /// If this is a `<0xNN>` byte-fallback token, the raw byte value it stands for.
    ///
    /// These tokens designate a literal byte (e.g. `<0x0A>` is a newline); decoding
    /// must emit that byte rather than the six literal characters of the name.
    pub fn byte_fallback_value(&self) -> Option<u8> {
        if !self.is_byte_fallback() {
            return None;
        }
        // bytes are `<0xNN>`: parse the two hex digits at positions 3 and 4.
        let hi = (self.bytes[3] as char).to_digit(16)?;
        let lo = (self.bytes[4] as char).to_digit(16)?;
        Some((hi * 16 + lo) as u8)
    }
}

/// A parsed tokenizer vocabulary.
#[derive(Debug)]
pub struct Vocab {
    /// Longest token piece in bytes (from the file header).
    pub max_token_length: usize,
    /// All `vocab_size` entries, indexed by token id.
    pub tokens: Vec<TokenEntry>,
}

impl Vocab {
    /// Number of tokens in the vocabulary.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Whether the vocabulary is empty. (Paired with [`len`](Vocab::len).)
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Load `vocab_size` entries from `path`.
    pub fn load(path: impl AsRef<Path>, vocab_size: usize) -> Result<Vocab, HostError> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|e| HostError::io(path, e))?;
        Self::parse(&bytes, vocab_size)
    }

    /// Parse a tokenizer blob holding exactly `vocab_size` entries.
    pub fn parse(bytes: &[u8], vocab_size: usize) -> Result<Vocab, HostError> {
        let mut r = Reader::new(bytes);
        let max_token_length = r.read_i32("max_token_length")?;
        if max_token_length < 0 {
            return Err(HostError::Tokenizer(format!(
                "negative max_token_length ({max_token_length})"
            )));
        }

        let mut tokens = Vec::with_capacity(vocab_size);
        for i in 0..vocab_size {
            let score = r.read_f32(i)?;
            let len = r.read_i32_len(i)?;
            let piece = r.read_bytes(len, i)?;
            tokens.push(TokenEntry {
                score,
                bytes: piece.to_vec(),
            });
        }

        if !r.at_end() {
            return Err(HostError::Tokenizer(format!(
                "trailing data: {} bytes remain after {vocab_size} tokens",
                r.remaining()
            )));
        }

        Ok(Vocab {
            max_token_length: max_token_length as usize,
            tokens,
        })
    }

    /// Encode `text` into token ids using SentencePiece-style greedy BPE.
    ///
    /// Reproduces run.c's `encode()`:
    /// 1. optionally emit BOS (`<s>`);
    /// 2. emit a "dummy prefix" space token before non-empty text;
    /// 3. map each UTF-8 codepoint to its token, or fall back to raw byte tokens
    ///    (`byte + 3`) when the codepoint isn't a vocabulary entry;
    /// 4. repeatedly merge the adjacent pair with the highest merge score until no
    ///    adjacent pair forms a known token.
    pub fn encode(&self, text: &str, bos: bool) -> Vec<usize> {
        // Exact piece-bytes -> id map. On duplicate pieces the lowest id wins,
        // which is deterministic (the merge scores are what actually drive output).
        let mut lookup: HashMap<&[u8], usize> = HashMap::with_capacity(self.tokens.len());
        for (id, tok) in self.tokens.iter().enumerate() {
            lookup.entry(tok.bytes.as_slice()).or_insert(id);
        }

        let mut tokens: Vec<usize> = Vec::new();
        if bos {
            tokens.push(BOS_ID);
        }
        // Dummy prefix: a leading space token before the real text.
        if !text.is_empty() {
            if let Some(&space) = lookup.get(b" ".as_slice()) {
                tokens.push(space);
            }
        }

        // First pass: one token per codepoint, with raw-byte fallback.
        let mut buf = [0u8; 4];
        for ch in text.chars() {
            let piece = ch.encode_utf8(&mut buf).as_bytes();
            if let Some(&id) = lookup.get(piece) {
                tokens.push(id);
            } else {
                for &b in piece {
                    tokens.push(b as usize + BYTE_TOKEN_OFFSET);
                }
            }
        }

        // Greedy merges: fuse the best-scoring adjacent pair until none remain.
        let mut pair = Vec::new();
        loop {
            let mut best: Option<(f32, usize, usize)> = None; // (score, merged_id, index)
            for i in 0..tokens.len().saturating_sub(1) {
                pair.clear();
                pair.extend_from_slice(&self.tokens[tokens[i]].bytes);
                pair.extend_from_slice(&self.tokens[tokens[i + 1]].bytes);
                if let Some(&id) = lookup.get(pair.as_slice()) {
                    let score = self.tokens[id].score;
                    if best.is_none_or(|(b, _, _)| score > b) {
                        best = Some((score, id, i));
                    }
                }
            }
            match best {
                Some((_, id, idx)) => {
                    tokens[idx] = id;
                    tokens.remove(idx + 1);
                }
                None => break,
            }
        }
        tokens
    }

    /// Decode `token` (whose predecessor was `prev_token`) to its raw output bytes.
    ///
    /// Mirrors run.c's `decode()`: drop a single leading space immediately after
    /// BOS, and expand `<0xNN>` byte-fallback tokens to the raw byte they denote.
    pub fn decode(&self, prev_token: usize, token: usize) -> Vec<u8> {
        let entry = &self.tokens[token];
        // A byte-fallback token is a single literal byte regardless of context.
        if let Some(b) = entry.byte_fallback_value() {
            return vec![b];
        }
        let piece = entry.bytes.as_slice();
        // SentencePiece strips one leading space right after BOS.
        if prev_token == BOS_ID && piece.first() == Some(&b' ') {
            piece[1..].to_vec()
        } else {
            piece.to_vec()
        }
    }
}

/// Minimal little-endian cursor over the tokenizer blob.
struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Reader { bytes, pos: 0 }
    }

    fn at_end(&self) -> bool {
        self.pos == self.bytes.len()
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.pos
    }

    fn take(&mut self, n: usize, what: impl std::fmt::Display) -> Result<&'a [u8], HostError> {
        let end = self.pos.checked_add(n).filter(|&e| e <= self.bytes.len());
        match end {
            Some(end) => {
                let out = &self.bytes[self.pos..end];
                self.pos = end;
                Ok(out)
            }
            None => Err(HostError::Tokenizer(format!(
                "unexpected end of file reading {what} (need {n} more bytes, have {})",
                self.remaining()
            ))),
        }
    }

    fn read_i32(&mut self, what: impl std::fmt::Display) -> Result<i32, HostError> {
        let b = self.take(4, what)?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_f32(&mut self, token_index: usize) -> Result<f32, HostError> {
        let b = self.take(4, format_args!("score for token {token_index}"))?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// Read a token's byte length and validate it is non-negative.
    fn read_i32_len(&mut self, token_index: usize) -> Result<usize, HostError> {
        let len = self.read_i32(format_args!("length of token {token_index}"))?;
        if len < 0 {
            return Err(HostError::Tokenizer(format!(
                "token {token_index} has negative length {len}"
            )));
        }
        Ok(len as usize)
    }

    fn read_bytes(&mut self, n: usize, token_index: usize) -> Result<&'a [u8], HostError> {
        self.take(n, format_args!("bytes of token {token_index}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic tokenizer blob from `(score, piece)` pairs.
    fn blob(max_len: i32, entries: &[(f32, &[u8])]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&max_len.to_le_bytes());
        for (score, piece) in entries {
            out.extend_from_slice(&score.to_le_bytes());
            out.extend_from_slice(&(piece.len() as i32).to_le_bytes());
            out.extend_from_slice(piece);
        }
        out
    }

    #[test]
    fn parses_synthetic_vocab() {
        let data = blob(5, &[(0.0, b"<unk>"), (-1.0, b"\xe2\x96\x81the"), (-2.0, b"<0x0A>")]);
        let v = Vocab::parse(&data, 3).unwrap();
        assert_eq!(v.max_token_length, 5);
        assert_eq!(v.len(), 3);
        assert_eq!(v.tokens[0].display(), "<unk>");
        assert_eq!(v.tokens[1].display(), "_the"); // space marker rendered as _
        assert!(v.tokens[2].is_byte_fallback());
        assert!(!v.tokens[0].is_byte_fallback());
    }

    #[test]
    fn rejects_truncated_blob() {
        let mut data = blob(5, &[(0.0, b"hi")]);
        data.pop(); // chop a byte off the last piece
        let err = Vocab::parse(&data, 1).unwrap_err();
        assert!(matches!(err, HostError::Tokenizer(_)));
    }

    #[test]
    fn rejects_trailing_data() {
        let mut data = blob(5, &[(0.0, b"hi")]);
        data.push(0xFF); // one extra byte
        assert!(matches!(
            Vocab::parse(&data, 1).unwrap_err(),
            HostError::Tokenizer(_)
        ));
    }

    #[test]
    fn rejects_too_few_tokens() {
        let data = blob(5, &[(0.0, b"hi")]);
        // Asking for 2 tokens when only 1 is present hits EOF.
        assert!(matches!(
            Vocab::parse(&data, 2).unwrap_err(),
            HostError::Tokenizer(_)
        ));
    }

    fn entry(score: f32, bytes: &[u8]) -> TokenEntry {
        TokenEntry {
            score,
            bytes: bytes.to_vec(),
        }
    }

    /// Small vocab with a real merge: `a` + `b` -> `ab` (high score) and a leading
    /// space token, plus `" a"` to check that the higher-scoring merge wins.
    fn ab_vocab() -> Vocab {
        Vocab {
            max_token_length: 5,
            tokens: vec![
                entry(0.0, b"<unk>"),    // 0
                entry(0.0, b"\n<s>\n"),  // 1 BOS
                entry(0.0, b"\n</s>\n"), // 2 EOS
                entry(-1.0, b"a"),       // 3
                entry(-2.0, b"b"),       // 4
                entry(5.0, b"ab"),       // 5  best merge
                entry(-3.0, b" "),       // 6  space
                entry(1.0, b" a"),       // 7  lower-scoring merge
            ],
        }
    }

    /// Vocab with the three specials followed by all 256 byte-fallback tokens, so
    /// `byte + 3` indexing is exercised end to end.
    fn byte_vocab() -> Vocab {
        let mut tokens = vec![
            entry(0.0, b"<unk>"),
            entry(0.0, b"\n<s>\n"),
            entry(0.0, b"\n</s>\n"),
        ];
        for n in 0u32..256 {
            tokens.push(entry(0.0, format!("<0x{n:02X}>").as_bytes()));
        }
        Vocab {
            max_token_length: 6,
            tokens,
        }
    }

    #[test]
    fn byte_fallback_value_parses_hex() {
        assert_eq!(entry(0.0, b"<0x0A>").byte_fallback_value(), Some(0x0A));
        assert_eq!(entry(0.0, b"<0xFF>").byte_fallback_value(), Some(0xFF));
        assert_eq!(entry(0.0, b"hello!").byte_fallback_value(), None);
        assert_eq!(entry(0.0, b" the").byte_fallback_value(), None);
    }

    #[test]
    fn encode_emits_bos_dummy_prefix_and_best_merge() {
        let v = ab_vocab();
        // BOS(1), space(6), then "a"+"b" merges to "ab"(5) — beating " "+"a"=" a"(7).
        assert_eq!(v.encode("ab", true), vec![1, 6, 5]);
        // Without BOS, no leading 1.
        assert_eq!(v.encode("ab", false), vec![6, 5]);
        // Empty text gets no dummy prefix, only BOS.
        assert_eq!(v.encode("", true), vec![1]);
    }

    #[test]
    fn encode_falls_back_to_byte_tokens() {
        let v = byte_vocab();
        // '\n' (0x0A) isn't a codepoint token here, so it falls back to id 0x0A+3.
        assert_eq!(v.encode("\n", false), vec![0x0A + 3]);
        // Two bytes, no possible merge in this vocab.
        assert_eq!(v.encode("\n\n", false), vec![0x0A + 3, 0x0A + 3]);
    }

    #[test]
    fn decode_strips_leading_space_after_bos() {
        let v = ab_vocab();
        // After BOS, the space token decodes to nothing.
        assert_eq!(v.decode(BOS_ID, 6), b"");
        // After BOS, " a" loses its leading space.
        assert_eq!(v.decode(BOS_ID, 7), b"a");
        // Not after BOS, the space is preserved.
        assert_eq!(v.decode(7, 7), b" a");
    }

    #[test]
    fn decode_expands_byte_fallback_tokens() {
        let v = byte_vocab();
        // id 0x0A+3 is the `<0x0A>` token -> a literal newline byte.
        assert_eq!(v.decode(0, 0x0A + 3), vec![b'\n']);
    }

    #[test]
    fn encode_decode_round_trips() {
        let v = ab_vocab();
        let toks = v.encode("ab", true);
        let mut out = Vec::new();
        for w in toks.windows(2) {
            out.extend(v.decode(w[0], w[1]));
        }
        assert_eq!(out, b"ab");
    }
}
