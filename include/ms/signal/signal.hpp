#pragma once

#include <array>
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

// Magnitude-squared coherence via Welch cross/auto spectral density estimation: same segmented,
// Hanning-windowed, overlapped FFT convention as welch_psd. For each frequency bin k,
// Cxy(k) = |Pxy(k)|^2 / (Pxx(k) * Pyy(k)) where Pxx/Pyy are Welch auto-PSDs of x/y and Pxy is
// the Welch cross-PSD (complex cross-periodogram averaged across segments). Values lie in [0, 1]:
// 1 means perfectly linearly related at that frequency, 0 means unrelated.
struct CoherenceResult {
    std::vector<double> frequencies; // Hz
    std::vector<double> coherence;   // magnitude-squared coherence, one per frequency bin
};
Result<CoherenceResult> coherence(const std::vector<double>& x, const std::vector<double>& y,
                                   double fs, size_t nperseg, double overlap_frac = 0.5);

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

// IIR filter design family: analog prototype + bilinear transform, returning direct-form
// (b, a) coefficients usable with ms::filter / ms::filtfilt / ms::sosfilt.
enum class FilterType { Lowpass, Highpass };

struct IirCoeffs {
    std::vector<double> b;
    std::vector<double> a;
};

// Chebyshev Type I IIR design: equiripple passband, monotonic stopband. `cutoff` is the
// -3 dB edge frequency in Hz; `fs` is the sample rate in Hz; `rp_db` is the passband ripple
// in dB (peak-to-minimum within the passband). Designs the analog prototype at normalized
// cutoff 1 rad/s, applies frequency prewarping and the bilinear transform, and returns
// ascending-power (b, a) with a[0] == 1. Defensive early returns (empty b/a) for order < 1,
// fs <= 0, rp_db < 0, or cutoff outside (0, fs/2) for lowpass/highpass.
IirCoeffs cheby1(int order, double rp_db, double cutoff, double fs,
                  FilterType type = FilterType::Lowpass);

// Generic digital filter application (direct-form difference equation): applies an IIR/FIR
// filter with numerator (feedforward) coefficients b and denominator (feedback) coefficients a
// to input x:
//   a[0]*y[n] = b[0]*x[n] + b[1]*x[n-1] + ... - a[1]*y[n-1] - a[2]*y[n-2] - ...
// If a[0] != 1, b and a are normalized internally by dividing through by a[0]. Zero initial
// conditions are assumed (x[n] = 0, y[n] = 0 for n < 0). If a.size() <= 1 (e.g. {1.0} or empty),
// this reduces to a pure FIR filter. Returns a vector the same length as x; empty x or empty b
// returns an empty vector.
std::vector<double> filter(const std::vector<double>& b, const std::vector<double>& a,
                            const std::vector<double>& x);

// Zero-phase filtering: applies ms::filter forward, reverses, applies ms::filter again, then
// reverses back, cancelling the phase delay a single causal pass would introduce (the standard
// filtfilt technique). To reduce edge transients, x is padded at both ends by reflecting
// ~3*max(a.size(), b.size()) samples before the double pass; padding is trimmed off afterward.
// Empty x or empty b returns an empty vector.
std::vector<double> filtfilt(const std::vector<double>& b, const std::vector<double>& a,
                              const std::vector<double>& x);

// Second-order-sections (SOS) IIR filtering: cascades ms::filter once per SOS row. Each row is
// [b0, b1, b2, a0, a1, a2]; a0 is normalized internally (same as filter). Empty sos returns x
// unchanged. Output length matches x (or empty when x is empty and sos is non-empty).
std::vector<double> sosfilt(const std::vector<std::array<double, 6>>& sos,
                             const std::vector<double>& x);

// Window functions usable with firwin/firwin_highpass.
enum class FirWindow { Rectangular, Hamming, Hann, Blackman };

// Designs a windowed-sinc FIR lowpass filter with n_taps coefficients and cutoff frequency
// `cutoff` normalized to Nyquist (cutoff in (0,1), where 1.0 == fs/2): the ideal (infinite)
// lowpass sinc impulse response is truncated/centered to n_taps samples, multiplied pointwise
// by the chosen window, then normalized so the DC gain (sum of taps) equals 1.0. n_taps should
// be odd for a symmetric/linear-phase filter centered at (n_taps-1)/2. n_taps <= 0 returns an
// empty vector.
std::vector<double> firwin(int n_taps, double cutoff, FirWindow window = FirWindow::Hamming);

// FIR highpass via spectral inversion of the equivalent firwin lowpass design: negates all taps
// and adds 1 to the center tap (highpass = delta - lowpass). Requires odd n_taps (so a unique
// center tap exists); even or non-positive n_taps returns an empty vector, matching this
// module's defensive early-return style for inputs that can't be validated any other way.
std::vector<double> firwin_highpass(int n_taps, double cutoff, FirWindow window = FirWindow::Hamming);

// Savitzky-Golay smoothing filter: fits a degree-`polyorder` polynomial by least squares to
// every sliding window of `window_length` consecutive samples, and takes the fitted
// polynomial's value at the window center as the smoothed output. Because the window is
// uniformly spaced, this reduces to a single fixed set of FIR coefficients (independent of
// position): letting A be the (window_length x (polyorder+1)) Vandermonde design matrix of
// integer offsets -(window_length-1)/2 .. +(window_length-1)/2 raised to powers 0..polyorder,
// the coefficients are the first row of the least-squares pseudoinverse (A^T A)^{-1} A^T,
// obtained here by solving the (polyorder+1) x (polyorder+1) normal-equations system
// (A^T A) v = e_0 (ms::solve) and forming h = A*v. These taps are then convolved directly with
// x over every fully-covered centered window.
//
// @param window_length Number of samples in the sliding window; must be odd and positive (so a
//                       unique integer center offset 0 exists) and no larger than x.size().
// @param polyorder      Degree of the local polynomial fit; must satisfy 0 <= polyorder <
//                       window_length (a polynomial of degree >= window_length - 1 would exactly
//                       interpolate every window with no smoothing effect at all, and one of
//                       degree >= window_length is not identifiable by least squares).
// @return Vector the same length as x with each interior point (indices with a full centered
//         window available) replaced by its local polynomial fit at the center; window_length
//         <= 0, even, polyorder out of [0, window_length), or x.size() < window_length returns
//         an empty vector, matching this module's defensive early-return style (e.g. firwin).
// @note Boundary handling: the first/last (window_length-1)/2 points, where a full centered
//       window does not fit, are simply copied unfiltered from the input rather than using a
//       reduced/asymmetric window — the simplest of the standard choices, at the cost of no
//       smoothing directly at the edges.
// @note Unlike a plain moving average (ms::moving_average), which is only exact for locally
//       constant/linear signals, Savitzky-Golay with polyorder >= 2 preserves polynomial trends
//       (e.g. quadratic/cubic peaks and slopes) up to that order essentially undistorted while
//       still averaging out higher-frequency noise, at the cost of slightly weaker noise
//       rejection than a same-length moving average for genuinely flat signals.
// @accuracy For a signal that is exactly a polynomial of degree <= polyorder plus i.i.d. noise,
//           the filtered output reproduces the polynomial trend to within the least-squares fit
//           residual; noise variance is reduced by a factor approximately equal to sum(h_i^2)
//           (< 1 for window_length > polyorder + 1) relative to the raw signal's noise variance.
std::vector<double> savgol(const std::vector<double>& x, int window_length, int polyorder);

// Median filter: replaces each sample with the median of a centered sliding window of the
// given (odd) length, a classic nonlinear denoising filter that — unlike moving_average or
// savgol — is highly robust to outliers/impulsive noise (a single extreme sample only shifts
// the local median by one rank position, rather than pulling a linear average arbitrarily far),
// at the cost of not preserving smooth polynomial trends the way savgol does.
// @param x input signal
// @param window_length window size; MUST be odd and >= 1 so a unique center sample exists.
//        Even or non-positive window_length, or x.size() < window_length, returns {},
//        matching this module's defensive early-return convention (e.g. savgol, firwin).
// @note Boundary handling: mirrors savgol's convention exactly — the first/last
//       (window_length-1)/2 points, where a full centered window does not fit, are copied
//       unfiltered from the input rather than using a shrinking/asymmetric window.
std::vector<double> median_filter(const std::vector<double>& x, int window_length);

// LMS (Least Mean Squares) adaptive filter: an online adaptive FIR filter that adjusts its
// filter_length taps to minimize the mean squared error between its output and a "desired"
// reference signal, sample by sample, via stochastic gradient descent on the instantaneous
// squared error. Classic applications: adaptive noise cancellation, system identification,
// channel equalization. At each time step n:
//   y[n] = sum_{k=0}^{filter_length-1} w[k] * x[n-k]     (filter output, using current weights)
//   e[n] = d[n] - y[n]                                    (error vs desired signal)
//   w[k] = w[k] + mu * e[n] * x[n-k]  for all k            (weight update, gradient step)
// where x is the input signal, d is the desired/reference signal (same length as x), mu is the
// step size (learning rate; must be small enough for stability -- typically mu <
// 2/(filter_length * signal_power), though you don't need to auto-tune this, just accept it as
// a caller-supplied parameter), and weights w start at all-zero.
struct LMSResult {
    std::vector<double> output;      // y[n] at each time step
    std::vector<double> error;       // e[n] at each time step
    std::vector<double> weights;     // final filter_length weights after processing all samples
};

// @param x input signal. @param d desired/reference signal, same length as x (mismatched
//        lengths, or filter_length<=0, or x/d shorter than filter_length, are defensive early-
//        return cases -- document and return a sane empty/trivial LMSResult).
// @param filter_length number of adaptive FIR taps.
// @param mu step size / learning rate (small positive value for stable convergence; this
//        function does not clamp or validate mu's stability bound itself -- that's the
//        caller's responsibility, matching how e.g. other numeric methods in this module with
//        tunable step sizes don't second-guess the caller's choice).
// @return per-sample output/error and final converged weights. x[n-k] for k > n (before the
//         start of the signal) is treated as 0 (zero-padding convention, matching how a causal
//         FIR filter naturally handles startup transients).
LMSResult lms_adaptive_filter(const std::vector<double>& x, const std::vector<double>& d,
                               int filter_length, double mu);

// Chirp Z-Transform (CZT): evaluates the Z-transform of a real signal x (length N) at M
// points z_k = a * w^(-k) for k = 0 .. m-1 along an arbitrary spiral/arc contour in the
// complex plane. Generalizes the DFT (which is recovered when m = N, a = 1, and
// w = exp(-2*pi*i/N)) and is most commonly used as a "zoom FFT" to obtain fine-grained
// frequency resolution over a narrow band without computing a full-length FFT.
// Implemented via Bluestein's algorithm (three zero-padded FFTs on a chirp-premultiplied
// signal convolved with a chirp kernel, followed by a chirp postmultiply).
//
// @param x Real input samples, length N.
// @param m Number of contour points to evaluate (output length).
// @param w Complex ratio between successive contour points (W in the standard notation).
// @param a Complex starting point on the contour (A in the standard notation).
// @return Complex CZT values X[k] = sum_n x[n] * z_k^(-n) for k = 0 .. m-1. Empty x, m <= 0,
//         or failed internal FFT steps return {} (no exceptions).
// @note When m = N, a = 1, and w = exp(-2*pi*i/N), the output matches ms::fft(x) bin-for-bin
//       (forward DFT convention used throughout ms::fft).
// @accuracy Matches a direct O(N*M) Z-transform evaluation to within ordinary double-precision
//           round-off; Bluestein sign conventions follow the scipy.signal.czt reference.
std::vector<std::complex<double>> czt(const std::vector<double>& x, int m,
                                       std::complex<double> w, std::complex<double> a);

// Zoom-FFT convenience wrapper around czt: samples m equally spaced frequency points from
// f_start to f_stop (Hz) at sample rate fs by setting
//   a = exp(2*pi*i*f_start/fs),  w = exp(-2*pi*i*delta_f/fs)
// where delta_f = (f_stop - f_start) / max(m - 1, 1). Output bin k corresponds to frequency
// f_start + k * delta_f. Useful for resolving closely spaced tones in a narrow band without
// zero-padding to a much longer FFT.
//
// @param x Real input signal.
// @param f_start Lower frequency bound (Hz), must lie in [0, fs/2).
// @param f_stop Upper frequency bound (Hz), must satisfy f_start < f_stop <= fs/2.
// @param m Number of frequency bins in the zoomed spectrum.
// @param fs Sample rate (Hz), must be positive.
// @return Complex zoom-FFT spectrum of length m, or {} for empty x, invalid m/fs/frequency
//         range, or other defensive early-return cases (no exceptions).
std::vector<std::complex<double>> czt_zoom_fft(const std::vector<double>& x, double f_start,
                                                double f_stop, int m, double fs);

} // namespace ms
