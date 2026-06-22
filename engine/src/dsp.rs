//! Real-FFT magnitude spectrum — a small `no_std` DSP primitive.
//!
//! Gated behind the crate's `fft` feature so the decoder model crates (`llama`,
//! `seq2seq`) pull in no FFT dependency. `edge-pm`'s vibration feature extractor enables
//! it to turn an acquisition window into a magnitude spectrum, from which it sums
//! log-spaced band energies.
//!
//! The transform is fixed at [`FFT_LEN`] = 512 real samples (the project's window length)
//! and runs in place via [`microfft`], so it allocates nothing.

use microfft::real::rfft_512;

/// Real input samples the [`rfft512_mag`] transform consumes.
pub const FFT_LEN: usize = 512;

/// Magnitude bins [`rfft512_mag`] produces (`FFT_LEN / 2`): bin `k` is frequency
/// `k · fs / FFT_LEN`, so for `fs = 1600 Hz` the bins span DC to just under Nyquist at
/// `1600 / 512 = 3.125 Hz` spacing.
pub const FFT_BINS: usize = FFT_LEN / 2;

/// Magnitude spectrum of a 512-sample real signal: `out[k] = |X[k]|`.
///
/// `signal` is transformed **in place** (microfft reuses its storage), so its contents are
/// scratch afterwards — fill it fresh for each call. The caller owns both buffers; nothing
/// is allocated.
///
/// Per microfft's real-FFT packing, bin 0 carries the DC term in its real part and the
/// Nyquist term in its imaginary part, so `out[0]` is `√(DC² + Nyquist²)` rather than a
/// pure DC magnitude. Callers that use only bins `1..FFT_BINS` (as the bearing band
/// features do, skipping DC) are unaffected.
pub fn rfft512_mag(signal: &mut [f32; FFT_LEN], out: &mut [f32; FFT_BINS]) {
    let spectrum = rfft_512(signal);
    for (o, c) in out.iter_mut().zip(spectrum.iter()) {
        *o = libm::sqrtf(c.re * c.re + c.im * c.im);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::PI;

    /// A pure tone at bin `k` puts essentially all of its energy in that bin.
    #[test]
    fn tone_concentrates_in_its_bin() {
        let k = 8usize;
        let mut sig = [0.0f32; FFT_LEN];
        for (n, s) in sig.iter_mut().enumerate() {
            *s = libm::cosf(2.0 * PI * k as f32 * n as f32 / FFT_LEN as f32);
        }
        let mut mag = [0.0f32; FFT_BINS];
        rfft512_mag(&mut sig, &mut mag);

        // Bin k dominates; neighbours and a far-away bin are tiny by comparison.
        let peak = mag[k];
        assert!(peak > 200.0, "tone bin too small: {peak}");
        assert!(mag[k + 4] < peak * 0.05);
        assert!(mag[100] < peak * 0.05);
        // The peak bin is the argmax over 1..FFT_BINS.
        let argmax = (1..FFT_BINS).max_by(|&a, &b| mag[a].total_cmp(&mag[b])).unwrap();
        assert_eq!(argmax, k);
    }

    /// A constant (DC) signal has energy only in bin 0; all other bins are ~zero.
    #[test]
    fn dc_signal_has_no_ac_energy() {
        let mut sig = [3.0f32; FFT_LEN];
        let mut mag = [0.0f32; FFT_BINS];
        rfft512_mag(&mut sig, &mut mag);
        assert!(mag[0] > 1.0);
        for &m in &mag[1..] {
            assert!(m < 1e-2, "spurious AC energy: {m}");
        }
    }
}
