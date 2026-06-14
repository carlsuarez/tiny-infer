//! Token sampling: turn a logits row into the next token id.
//!
//! The decode step is the same on every architecture — a forward pass yields a `vocab`-long
//! logits row, and *something* must pick the next token from it. That choice is
//! architecture-agnostic (it touches only the logits), so the [`Sampler`] lives in the
//! shared core and serves both the Llama generation loop and the seq2seq decode loop.
//!
//! Three modes, decided by `temperature` and `topp` (matching llama2.c's `Sampler`):
//! * `temperature == 0` → deterministic **greedy** ([`argmax`]); the path the llama2.c
//!   parity gate and the Marian greedy-decode gate exercise.
//! * `temperature > 0`, `topp` disabled (`<= 0` or `>= 1`) → draw from the full
//!   `softmax(logits / temperature)` distribution.
//! * `temperature > 0`, `0 < topp < 1` → **nucleus** (top-p) sampling: draw only from the
//!   smallest set of most-probable tokens whose cumulative probability reaches `topp`.
//!
//! For **seq2seq** tasks (translation, summarization, …), greedy (or beam search, a separate
//! *search* not a sampler) is the quality path; temperature/top-p trade accuracy for
//! diversity. They are offered for both architectures because the math is identical, not
//! because sampling improves a faithful sequence-to-sequence output.
//!
//! Like the rest of the engine the sampler is `no_std` and allocation-free: it carries a
//! seedable PRNG from the `no_std` [`rand`] crate ([`Xoshiro128PlusPlus`] — a fixed,
//! cross-platform algorithm with native 32-bit ops, so a given `--seed` reproduces the same
//! run on a 64-bit host and a 32-bit MCU alike) seeded by the host rather than OS entropy,
//! and the top-p nucleus is sorted in a **caller-provided** [`ProbIndex`] scratch (`vocab`
//! long) rather than a `Vec`.

use rand::rngs::Xoshiro128PlusPlus;
use rand::{RngExt, SeedableRng};

use crate::math::{argmax, maxf};

/// A `(probability, token)` pair — one element of the top-p nucleus scratch.
///
/// [`Sampler::sample`] fills and sorts a prefix of a caller-provided `&mut [ProbIndex]`
/// (at least `vocab` long) for nucleus sampling; greedy and full sampling ignore it. The
/// host owns the buffer (`vec![ProbIndex::default(); vocab]`); a bare-metal caller hands in
/// a static array.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ProbIndex {
    /// Normalized probability of this token.
    pub prob: f32,
    /// The token id.
    pub token: usize,
}

/// Turns a logits row into the next token id — greedy, full, or nucleus (top-p) sampling.
///
/// Built once per run with [`Sampler::new`] and called once per decode step. It owns its
/// PRNG (seeded by the caller) so a given seed reproduces a run exactly; the only external
/// state is the [`ProbIndex`] scratch passed to [`sample`](Sampler::sample).
pub struct Sampler {
    temperature: f32,
    topp: f32,
    rng: Xoshiro128PlusPlus,
}

impl Sampler {
    /// Build a sampler.
    ///
    /// `temperature == 0` is greedy (the RNG and `topp` are then unused). `0 < topp < 1`
    /// enables nucleus sampling; any other `topp` draws from the full distribution. `seed`
    /// makes a `temperature > 0` run reproducible — the host passes an explicit `--seed`
    /// or one drawn from OS entropy.
    pub fn new(temperature: f32, topp: f32, seed: u64) -> Sampler {
        Sampler {
            temperature,
            topp,
            rng: Xoshiro128PlusPlus::seed_from_u64(seed),
        }
    }

    /// Pick the next token id from `logits`.
    ///
    /// `scratch` is the nucleus buffer; it must be at least `logits.len()` long for top-p
    /// sampling and is ignored by the greedy and full-distribution paths (pass an empty
    /// slice if you only ever sample greedily).
    ///
    /// # Panics
    /// In debug builds, if top-p sampling is active and `scratch` is shorter than `logits`.
    pub fn sample(&mut self, logits: &[f32], scratch: &mut [ProbIndex]) -> usize {
        if self.temperature == 0.0 {
            return argmax(logits);
        }

        // Unnormalized softmax weight of one logit (max-shifted for stability). Captured by
        // value so the closure doesn't borrow `self` (the helpers below need `&mut self`).
        let (max, temperature) = (maxf(logits), self.temperature);
        let weight = move |v: f32| libm::expf((v - max) / temperature);
        let sum: f32 = logits.iter().copied().map(weight).sum();

        if self.topp <= 0.0 || self.topp >= 1.0 {
            return self.sample_full(logits, &weight, sum);
        }
        self.sample_topp(logits, &weight, sum, scratch)
    }

    /// Draw from the full distribution via inverse-transform sampling: compare a scaled
    /// uniform draw against the running unnormalized cumulative weight.
    fn sample_full(&mut self, logits: &[f32], weight: &impl Fn(f32) -> f32, sum: f32) -> usize {
        let target = self.rng.random::<f32>() * sum; // uniform in [0, sum)
        let mut cdf = 0.0f32;
        for (i, &v) in logits.iter().enumerate() {
            cdf += weight(v);
            if cdf > target {
                return i;
            }
        }
        logits.len() - 1 // only reached if f32 rounding leaves cdf just under target
    }

    /// Top-p (nucleus) sampling: keep the smallest set of highest-probability tokens whose
    /// cumulative probability reaches `topp`, then sample within it (using `scratch`).
    fn sample_topp(
        &mut self,
        logits: &[f32],
        weight: &impl Fn(f32) -> f32,
        sum: f32,
        scratch: &mut [ProbIndex],
    ) -> usize {
        debug_assert!(
            scratch.len() >= logits.len(),
            "top-p scratch must be at least `vocab` long"
        );
        // Tokens below this probability cannot belong to the nucleus, so crop them before
        // sorting. Holds whenever `topp >= 1/n`.
        let cutoff = (1.0 - self.topp) / (logits.len() - 1) as f32;
        let mut n = 0;
        for (i, &v) in logits.iter().enumerate() {
            let p = weight(v) / sum;
            if p >= cutoff {
                scratch[n] = ProbIndex { prob: p, token: i };
                n += 1;
            }
        }
        if n == 0 {
            // `topp` smaller than 1/vocab cropped everything; fall back to greedy.
            return argmax(logits);
        }
        let nucleus = &mut scratch[..n];
        // Highest probability first. `total_cmp` gives a total order (no NaN unwrap).
        nucleus.sort_unstable_by(|a, b| b.prob.total_cmp(&a.prob));

        // Truncate where the cumulative probability first exceeds `topp`.
        let mut cumulative = 0.0f32;
        let mut last = nucleus.len() - 1;
        for (i, pi) in nucleus.iter().enumerate() {
            cumulative += pi.prob;
            if cumulative > self.topp {
                last = i;
                break;
            }
        }

        // Sample within the nucleus, renormalized by its cumulative mass.
        let target = self.rng.random::<f32>() * cumulative;
        let mut cdf = 0.0f32;
        for pi in &nucleus[..=last] {
            cdf += pi.prob;
            if cdf > target {
                return pi.token;
            }
        }
        nucleus[last].token // f32 rounding fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    extern crate std;
    use std::vec;
    use std::vec::Vec;

    #[test]
    fn temperature_zero_is_greedy() {
        // Greedy ignores the RNG and the scratch; it is exactly `argmax`.
        let mut s = Sampler::new(0.0, 0.0, 1);
        assert_eq!(s.sample(&[0.1, 9.0, 0.2, 0.3], &mut []), 1);
        // Ties resolve to the first index (matching `argmax`).
        assert_eq!(s.sample(&[2.0, 2.0, 1.0], &mut []), 0);
    }

    #[test]
    fn tiny_topp_collapses_to_top_token() {
        // A near-zero topp shrinks the nucleus to the single most-probable token, so the
        // draw is deterministic regardless of temperature or seed.
        let logits = [1.0f32, 5.0, 2.0, 0.5];
        let mut scratch = vec![ProbIndex::default(); logits.len()];
        let mut s = Sampler::new(1.0, 1e-6, 123);
        for _ in 0..8 {
            assert_eq!(s.sample(&logits, &mut scratch), 1);
        }
    }

    #[test]
    fn is_reproducible_with_a_seed() {
        let logits = [1.0f32, 2.0, 0.5, 1.5, 3.0];
        let draws = |seed| {
            let mut scratch = vec![ProbIndex::default(); logits.len()];
            let mut s = Sampler::new(1.0, 0.9, seed);
            (0..16)
                .map(|_| s.sample(&logits, &mut scratch))
                .collect::<Vec<_>>()
        };
        assert_eq!(draws(7), draws(7));
        // Every sampled id is a valid token index.
        assert!(draws(7).iter().all(|&t| t < logits.len()));
    }

    #[test]
    fn full_sampling_favors_the_peak() {
        // With one logit far above the rest, full-distribution sampling lands on it the
        // overwhelming majority of the time.
        let logits = [0.0f32, 0.0, 8.0, 0.0];
        let mut s = Sampler::new(1.0, 0.0, 42); // topp 0 ⇒ full distribution
        let hits = (0..200)
            .filter(|_| s.sample(&logits, &mut []) == 2)
            .count();
        assert!(hits > 180, "peak token won only {hits}/200 draws");
    }

    #[test]
    fn zero_seed_samples_validly() {
        // `seed_from_u64` expands the seed through SplitMix, so even a zero seed yields a
        // healthy generator state (no all-zero lockup) and every draw is a valid token.
        let logits = [1.0f32, 5.0, 2.0, 0.5];
        let mut scratch = vec![ProbIndex::default(); logits.len()];
        let mut s = Sampler::new(1.0, 0.9, 0);
        for _ in 0..1000 {
            assert!(s.sample(&logits, &mut scratch) < logits.len());
        }
    }
}
