#pragma once

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

} // namespace ms
