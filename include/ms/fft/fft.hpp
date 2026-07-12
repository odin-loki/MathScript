#pragma once

#include <complex>
#include <span>
#include <vector>
#include "ms/error/error_types.hpp"

namespace ms {

Result<std::vector<std::complex<double>>> fft(const std::vector<double>& x);
Result<std::vector<double>> ifft(const std::vector<std::complex<double>>& x);
Result<std::vector<std::complex<double>>> fft2(const std::vector<std::complex<double>>& data);
Result<std::vector<std::complex<double>>> ifft2(const std::vector<std::complex<double>>& data);
Result<std::vector<std::complex<double>>> dft(std::span<const double> data);

Result<std::vector<std::complex<double>>> rfft(const std::vector<double>& x);
Result<std::vector<double>> irfft(const std::vector<std::complex<double>>& x, size_t n);

std::vector<std::complex<double>> fftshift(const std::vector<std::complex<double>>& x);
std::vector<std::complex<double>> ifftshift(const std::vector<std::complex<double>>& x);

/// Returns the DFT sample frequencies for a length-`n` signal sampled with
/// spacing `d`, matching NumPy's `numpy.fft.fftfreq` convention exactly. The
/// result is ordered to match the bin layout produced by `fft`/`dft` (i.e.
/// index 0 is the DC term, followed by increasing positive frequencies, then
/// the negative frequencies in increasing order, wrapping around):
///   - even `n`: [0, 1, ..., n/2-1, -n/2, -n/2+1, ..., -1] / (n*d)
///   - odd  `n`: [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (n*d)
/// @param n Number of samples/bins.
/// @param d Sample spacing (defaults to 1.0).
/// @return  Length-`n` vector of frequencies (cycles per unit of `d`).
///          Returns an empty vector for `n == 0`. If `d == 0`, every entry is
///          divided by zero and follows IEEE-754 semantics (i.e. the result
///          contains `+-inf`/`nan` rather than crashing).
std::vector<double> fftfreq(size_t n, double d = 1.0);

/// Returns the sample frequencies for the non-negative-frequency half of the
/// spectrum produced by `rfft`, matching NumPy's `numpy.fft.rfftfreq`
/// convention exactly. Output length is `n/2 + 1` (integer division), the
/// same length as `rfft`'s output:
///   [0, 1, ..., n/2] / (n*d)
/// @param n Number of samples in the original (real-valued) signal.
/// @param d Sample spacing (defaults to 1.0).
/// @return  Length-`n/2+1` vector of non-negative frequencies (cycles per
///          unit of `d`). Returns an empty vector for `n == 0`. If `d == 0`,
///          follows IEEE-754 division semantics (`+inf`/`nan`) rather than
///          crashing.
std::vector<double> rfftfreq(size_t n, double d = 1.0);

Result<std::vector<double>> dct2(const std::vector<double>& x);
Result<std::vector<double>> idct2(const std::vector<double>& x);
Result<std::vector<double>> dst2(const std::vector<double>& x);
Result<std::vector<double>> idst2(const std::vector<double>& x);

/// Goertzel algorithm: extracts a single DFT bin X(k) of a real signal at a
/// target frequency `f` (Hz), given sample rate `fs` (Hz), without computing
/// the full spectrum. Equivalent to `fft(x)[k]` for `k = round(f/fs*n)`
/// where `n = x.size()`, but runs in O(n) time and O(1) extra space via the
/// second-order IIR recurrence:
///   w = 2*pi*k/n, coeff = 2*cos(w)
///   s[i] = x[i] + coeff*s[i-1] - s[i-2]   (s[-1] = s[-2] = 0)
///   X(k) = (s[n-1]*cos(w) - s[n-2]) + i*(s[n-1]*sin(w))
///
/// @param x  Real input signal samples.
/// @param f  Target frequency in Hz to extract, must lie in [0, fs/2].
/// @param fs Sample rate in Hz, must be positive.
/// @return   The complex DFT bin value X(k). Returns {0,0} for defensive
///           cases: empty `x`, non-positive `fs`, or `f` outside the valid
///           Nyquist range [0, fs/2].
/// @note Use this instead of `fft`/`rfft` when only one (or a few) bins are
///       needed: O(n) per bin vs O(n log n) for a full FFT plus indexing.
///       For more than ~log2(n) bins, a full FFT is more efficient overall.
/// @accuracy Matches the corresponding full-FFT bin to within ordinary
///           floating-point round-off (double precision), since both
///           evaluate the same DFT sum via numerically equivalent recurrences.
std::complex<double> goertzel(std::span<const double> x, double f, double fs);

} // namespace ms
