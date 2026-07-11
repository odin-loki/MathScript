#pragma once

#include <complex>
#include <vector>
#include "ms/core/matrix.hpp"
#include "ms/error/error_types.hpp"

namespace ms {

std::vector<double> butterworth(const std::vector<double>& x, double cutoff, double fs);
std::vector<double> lowpass(const std::vector<double>& x, double cutoff, double fs);
std::vector<double> highpass(const std::vector<double>& x, double cutoff, double fs);
std::vector<double> bandpass(const std::vector<double>& x, double low, double high, double fs);
std::vector<double> convolve(const std::vector<double>& a, const std::vector<double>& b);
std::vector<double> correlate(const std::vector<double>& a, const std::vector<double>& b);
std::vector<double> moving_average(const std::vector<double>& x, size_t window);
std::vector<double> hamming(size_t n);
std::vector<double> hanning(size_t n);
std::vector<double> blackman(size_t n);
std::vector<double> parzen(size_t n);
std::vector<double> triangular(size_t n);

// Resampling family: change the effective sample rate of a signal by inserting or dropping
// samples, optionally with anti-aliasing/anti-imaging low-pass filtering. `n`/`p`/`q` <= 0 or
// an empty input signal returns an empty vector (no exceptions), matching moving_average's
// defensive style.

// Zero-stuffing upsample: inserts n-1 zeros after each input sample. Output length is
// x.size()*n. E.g. upsample({1,2}, 3) == {1,0,0,2,0,0}.
std::vector<double> upsample(const std::vector<double>& x, int n);

// Naive decimation: keeps every n-th sample starting at index 0, with no anti-aliasing filter.
// E.g. downsample({1,2,3,4,5,6}, 2) == {1,3,5}.
std::vector<double> downsample(const std::vector<double>& x, int n);

// Anti-aliased downsampling by integer factor q: low-pass filters x (ms::lowpass, treating x as
// having unit sample rate fs = 1.0) with cutoff = 0.8 * new_nyquist, where new_nyquist =
// 1/(2*q) is the Nyquist frequency of the decimated signal and 0.8 is a safety margin below it,
// then keeps every q-th sample (ms::downsample). Pre-filtering attenuates frequency content
// above the new Nyquist rate so it cannot alias into the decimated signal.
std::vector<double> decimate(const std::vector<double>& x, int q);

// Band-limited interpolation by integer factor p: zero-stuffs x (ms::upsample) then low-pass
// filters (ms::lowpass, unit sample rate fs = 1.0) with cutoff = 0.8 * new_nyquist, where
// new_nyquist = 1/(2*p), smoothing the inserted zeros into interpolated values. The filtered
// result is scaled by p to undo the amplitude loss caused by zero-stuffing (averaging in zeros
// divides signal energy by p).
std::vector<double> interpolate(const std::vector<double>& x, int p);

// Rational resampling: changes the sample rate by a factor of p/q via
// ms::decimate(ms::interpolate(x, p), q) — interpolate (upsample by p) first so no information
// is lost to decimation, then decimate (downsample by q) to the target rate.
std::vector<double> resample(const std::vector<double>& x, int p, int q);

// Welch's method for power spectral density estimation: splits x into overlapping segments of
// length `segment_len`, applies a Hanning window (ms::hanning), computes the FFT magnitude-squared
// of each windowed segment, and averages across segments. Returns a one-sided PSD (non-negative
// frequencies only, matching ms::rfft output). Normalization: each segment contributes
// |X[k]|^2 / (fs * sum(window^2)); segment periodograms are averaged, then interior one-sided
// bins are doubled (factor 2 for k = 1 .. n/2-1 when n_fft is even, or k = 1 .. n/2 when odd)
// so that integrating the one-sided PSD recovers total signal power (standard scipy-style
// 'density' scaling). `overlap_frac` is the fractional overlap between consecutive segments
// (e.g. 0.5 = 50% overlap).
struct PSDResult {
    std::vector<double> frequencies; // Hz
    std::vector<double> power;       // PSD values, one per frequency bin
};
Result<PSDResult> welch_psd(const std::vector<double>& x, double fs,
                             size_t segment_len, double overlap_frac = 0.5);

// Short-time Fourier transform spectrogram: same windowing and overlap convention as welch_psd,
// but stores the FFT magnitude (not magnitude-squared) of each segment. `magnitude` is a
// times.size() x frequencies.size() matrix where row i is the frequency-magnitude spectrum of
// time segment i (column j is frequency bin j). Segment center times are in `times`.
struct SpectrogramResult {
    std::vector<double> times;       // segment center times, seconds
    std::vector<double> frequencies; // Hz
    Matrix<double> magnitude;        // rows = times, cols = frequencies
};
Result<SpectrogramResult> spectrogram(const std::vector<double>& x, double fs,
                                       size_t segment_len, double overlap_frac = 0.5);

// Discrete Hilbert transform / analytic signal via FFT: negative-frequency bins are zeroed,
// positive-frequency bins (except DC and Nyquist) are doubled, then inverse FFT. Returns the
// complex analytic signal z[n] = x[n] + j*H{x[n]} with the same length as x (FFT zero-padding
// used internally is trimmed). Empty input returns an empty vector.
std::vector<std::complex<double>> hilbert(const std::vector<double>& x);

// Amplitude envelope: |hilbert(x)| element-wise.
std::vector<double> envelope(const std::vector<double>& x);

// Instantaneous phase: arg(hilbert(x)) element-wise, in radians in (-pi, pi].
std::vector<double> instantaneous_phase(const std::vector<double>& x);

// Instantaneous frequency in Hz from the unwrapped phase derivative:
//   f[n] = (unwrap(arg(hilbert(x)))[n+1] - unwrap(...)[n]) * fs / (2*pi)
// for n = 0 .. N-2; f[N-1] repeats f[N-2] so the output length matches x (N samples). Edge
// samples near the ends inherit FFT-based Hilbert boundary effects; callers should exclude
// boundary regions for accuracy checks.
std::vector<double> instantaneous_freq(const std::vector<double>& x, double fs);

// Cross-correlation for lags -max_lag .. +max_lag (2*max_lag+1 values, index max_lag is lag 0).
// Positive lag means b is delayed relative to a (MATLAB/scipy-style). Unnormalized linear
// correlation implemented by extracting the corresponding slice from ms::correlate(a,b).
// Values outside the valid range of the full linear correlation are zero-padded.
std::vector<double> xcorr(const std::vector<double>& a, const std::vector<double>& b, int max_lag);

// Cross-covariance: xcorr applied to mean-removed a and b (same lag convention as xcorr).
std::vector<double> xcov(const std::vector<double>& a, const std::vector<double>& b, int max_lag);

// Autocorrelation convenience wrapper: xcorr(x, x, max_lag). Unlike ms::acf (stats), this
// returns symmetric lags -max_lag..+max_lag, is unnormalized (zero-lag value is sum(x[i]^2)),
// and does not subtract the mean. Use ms::acf for normalized, mean-centered, one-sided ACF
// suitable for time-series modeling; use autocorr for raw lag-domain signal matching.
std::vector<double> autocorr(const std::vector<double>& x, int max_lag);

// 2D discrete convolution of matrices (full output, size (A.rows()+B.rows()-1) x
// (A.cols()+B.cols()-1)). Direct nested-loop implementation; FFT-based 2D convolution could
// be added later for large inputs.
Matrix<double> conv2(const Matrix<double>& A, const Matrix<double>& B);

// Polynomial/signal deconvolution: if y = convolve(x, b), returns x via polynomial long
// division (ms::poly::poly_div_quot). Coefficient vectors use ascending-power convention.
std::vector<double> deconv(const std::vector<double>& y, const std::vector<double>& b);

} // namespace ms
