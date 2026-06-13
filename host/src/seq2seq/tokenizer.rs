//! Marian / OPUS-MT **SentencePiece Unigram** tokenizer.
//!
//! Loads the `tokenizer.bin` artifact built by `scripts/dump_tokenizer.py` and
//! reproduces SentencePiece encoding so the engine sees the same token ids Hugging
//! Face's `MarianTokenizer` would produce:
//!
//! 1. **Normalize** — collapse every run of whitespace to one space, strip the ends,
//!    and prefix each word with the `▁` whitespace marker (SentencePiece's
//!    `add_dummy_prefix` + space escaping).
//! 2. **Segment** — Viterbi over the `(piece, score)` table: the segmentation with the
//!    greatest total Unigram score wins (this is *not* longest-match). Characters no
//!    piece covers fall back to the unknown token.
//! 3. **Map + terminate** — look each chosen piece up in the shared vocabulary and
//!    append `</s>` (eos). Marian prepends no BOS on the source side.
//!
//! Decoding inverts the id→piece map, concatenates, turns `▁` back into spaces, and
//! trims — exactly SentencePiece's detokenization for this model.
//!
//! **Normalization caveat.** SentencePiece's full normalizer also applies NFKC
//! compatibility folding (e.g. `½`→`1⁄2`, the `ﬁ` ligature→`fi`). This implementation
//! does whitespace handling only, so it matches Hugging Face on ASCII and NFC text —
//! all ordinary English input, including accented words like `café` — and differs only
//! on rare compatibility characters. (A full NFKC pass is a future refinement.)

use std::collections::HashMap;
use std::path::Path;

use crate::error::HostError;

/// SentencePiece's whitespace marker (`U+2581`, "lower one eighth block").
const WHITESPACE_MARKER: char = '▁';

const MAGIC: &[u8; 4] = b"tit1";
const VERSION: i32 = 1;

/// A Marian source-side tokenizer: encode English text → ids, decode ids → text.
pub struct Tokenizer {
    /// Segmentable pieces: string → (shared-vocab id, Unigram score).
    pieces: HashMap<String, (u32, f32)>,
    /// Longest piece in `char`s — the Viterbi inner-loop bound.
    max_piece_chars: usize,
    /// Score charged for a single-character unknown edge (more negative than any real
    /// piece, so Viterbi only takes it when nothing else covers the character).
    unk_score: f32,
    /// id → piece string, for decoding (the full shared vocabulary).
    id_to_piece: Vec<String>,
    unk_id: usize,
    eos_id: usize,
    pad_id: usize,
}

/// A little-endian cursor over the artifact bytes.
struct Reader<'a> {
    bytes: &'a [u8],
    off: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Reader<'a> {
        Reader { bytes, off: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], ()> {
        let end = self.off.checked_add(n).ok_or(())?;
        let slice = self.bytes.get(self.off..end).ok_or(())?;
        self.off = end;
        Ok(slice)
    }

    fn u32(&mut self) -> Result<u32, ()> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn i32(&mut self) -> Result<i32, ()> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn f32(&mut self) -> Result<f32, ()> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    /// A `u32` length prefix followed by that many UTF-8 bytes.
    fn string(&mut self) -> Result<String, ()> {
        let len = self.u32()? as usize;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| ())
    }
}

impl Tokenizer {
    /// Load a `tokenizer.bin` artifact (see `scripts/dump_tokenizer.py`).
    pub fn load(path: impl AsRef<Path>) -> Result<Tokenizer, HostError> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|e| HostError::io(path, e))?;
        Self::parse(&bytes).ok_or_else(|| HostError::Format {
            path: path.to_path_buf(),
            msg: "malformed seq2seq tokenizer artifact (run scripts/dump_tokenizer.py)".into(),
        })
    }

    /// Parse the binary layout; `None` on any structural problem.
    fn parse(bytes: &[u8]) -> Option<Tokenizer> {
        let mut r = Reader::new(bytes);
        if r.take(4).ok()? != MAGIC || r.i32().ok()? != VERSION {
            return None;
        }
        let _vocab_size = r.i32().ok()? as usize;
        let n_src = r.i32().ok()? as usize;
        let unk_id = r.i32().ok()? as usize;
        let eos_id = r.i32().ok()? as usize;
        let pad_id = r.i32().ok()? as usize;

        let mut pieces = HashMap::with_capacity(n_src);
        let mut max_piece_chars = 1;
        let mut min_score = 0.0f32;
        for _ in 0..n_src {
            let model_id = r.u32().ok()?;
            let score = r.f32().ok()?;
            let piece = r.string().ok()?;
            max_piece_chars = max_piece_chars.max(piece.chars().count());
            min_score = min_score.min(score);
            pieces.insert(piece, (model_id, score));
        }

        let id_to_piece_len = _vocab_size;
        let mut id_to_piece = Vec::with_capacity(id_to_piece_len);
        for _ in 0..id_to_piece_len {
            id_to_piece.push(r.string().ok()?);
        }

        Some(Tokenizer {
            pieces,
            max_piece_chars,
            // SentencePiece-style: unknown costs a bit more than the worst real piece.
            unk_score: min_score - 10.0,
            id_to_piece,
            unk_id,
            eos_id,
            pad_id,
        })
    }

    /// Encode source text into shared-vocab token ids, terminated with `</s>`.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let norm = self.normalize(text);
        let chars: Vec<char> = norm.chars().collect();
        let mut ids = self.viterbi(&chars);
        ids.push(self.eos_id);
        ids
    }

    /// Whitespace normalization: collapse runs, strip ends, and prefix each word with
    /// the `▁` marker (the input side of SentencePiece's `escape_whitespaces`).
    fn normalize(&self, text: &str) -> String {
        let mut out = String::with_capacity(text.len() + 8);
        for word in text.split_whitespace() {
            out.push(WHITESPACE_MARKER);
            out.push_str(word);
        }
        out
    }

    /// Best-scoring Unigram segmentation of `chars` into piece ids (no eos).
    fn viterbi(&self, chars: &[char]) -> Vec<usize> {
        let n = chars.len();
        if n == 0 {
            return Vec::new();
        }
        // best[i] = greatest total score of any segmentation of chars[..i].
        let mut best = vec![f32::NEG_INFINITY; n + 1];
        // back[i] = (start index, model id) of the piece ending at i on the best path.
        let mut back = vec![(0usize, self.unk_id as u32); n + 1];
        best[0] = 0.0;

        let mut buf = String::new();
        for i in 0..n {
            if best[i] == f32::NEG_INFINITY {
                continue; // unreachable (unk edges keep every position reachable)
            }
            // Try every known piece starting at i, growing the candidate char by char.
            buf.clear();
            let upper = (i + self.max_piece_chars).min(n);
            for (j, &ch) in chars.iter().enumerate().take(upper).skip(i) {
                buf.push(ch);
                if let Some(&(id, score)) = self.pieces.get(&buf) {
                    let cand = best[i] + score;
                    if cand > best[j + 1] {
                        best[j + 1] = cand;
                        back[j + 1] = (i, id);
                    }
                }
            }
            // Single-character unknown fallback edge i → i+1.
            let cand = best[i] + self.unk_score;
            if cand > best[i + 1] {
                best[i + 1] = cand;
                back[i + 1] = (i, self.unk_id as u32);
            }
        }

        // Backtrack from the end, collecting piece ids, then reverse.
        let mut ids = Vec::new();
        let mut pos = n;
        while pos > 0 {
            let (start, id) = back[pos];
            ids.push(id as usize);
            pos = start;
        }
        ids.reverse();
        ids
    }

    /// Decode target token ids into text: id→piece, join, `▁`→space, trim.
    pub fn decode(&self, ids: &[usize]) -> String {
        let mut s = String::new();
        for &id in ids {
            if id == self.eos_id || id == self.pad_id {
                continue;
            }
            if let Some(piece) = self.id_to_piece.get(id) {
                s.push_str(piece);
            }
        }
        s.replace(WHITESPACE_MARKER, " ").trim().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A hand-built tokenizer over a few pieces, to exercise Viterbi/normalize/decode
    /// without the real artifact. Scores favor longer pieces where they exist.
    fn toy() -> Tokenizer {
        let entries = [
            ("▁he", 10u32, -1.0f32),
            ("▁hello", 11, -1.5),
            ("ll", 12, -3.0),
            ("o", 13, -3.0),
            ("e", 14, -3.0),
            ("h", 15, -4.0),
            ("▁", 16, -2.0),
            ("!", 17, -1.0),
        ];
        let mut pieces = HashMap::new();
        let mut max_piece_chars = 1;
        let mut min_score = 0.0f32;
        let mut id_to_piece = vec![String::new(); 32];
        for (p, id, sc) in entries {
            max_piece_chars = max_piece_chars.max(p.chars().count());
            min_score = min_score.min(sc);
            pieces.insert(p.to_string(), (id, sc));
            id_to_piece[id as usize] = p.to_string();
        }
        Tokenizer {
            pieces,
            max_piece_chars,
            unk_score: min_score - 10.0,
            id_to_piece,
            unk_id: 0,
            eos_id: 1,
            pad_id: 2,
        }
    }

    #[test]
    fn normalize_collapses_and_marks_whitespace() {
        let t = toy();
        assert_eq!(t.normalize("  hello   world  "), "▁hello▁world");
        assert_eq!(t.normalize("hi"), "▁hi");
        assert_eq!(t.normalize(""), "");
    }

    #[test]
    fn viterbi_prefers_the_higher_scoring_segmentation() {
        let t = toy();
        // "▁hello!" → the single piece "▁hello" (−1.5) beats "▁he"+"ll"+"o" (−7),
        // then "!" and the eos.
        let ids = t.encode("hello!");
        assert_eq!(ids, vec![11, 17, t.eos_id]);
    }

    #[test]
    fn unknown_characters_fall_back_to_unk() {
        let t = toy();
        // 'z' is covered by no piece, so it takes the unk edge; the leading ▁ marker
        // is its own piece, giving [▁, unk, eos].
        let ids = t.encode("z");
        assert_eq!(*ids.last().unwrap(), t.eos_id);
        assert!(ids.contains(&t.unk_id), "expected an unk token in {ids:?}");
    }

    #[test]
    fn decode_joins_pieces_and_restores_spaces() {
        let t = toy();
        // ▁hello + ! → " hello!" → trimmed "hello!"; eos/pad are skipped.
        assert_eq!(t.decode(&[11, 17, t.eos_id]), "hello!");
        assert_eq!(t.decode(&[t.pad_id, 11, t.pad_id]), "hello");
    }
}
