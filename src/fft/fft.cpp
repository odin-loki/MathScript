#include "ms/fft/fft.hpp"
#include "ms/cuda/fft.hpp"
#include "ms/runtime/dispatch.hpp"
#include "ms/runtime/topology.hpp"
#include <cmath>
#include <mutex>
#include <unordered_map>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ms {

namespace {

bool is_power_of_two(size_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

size_t log2_power_of_two(size_t n) {
    size_t log_n = 0;
    while (n > 1) {
        n >>= 1;
        ++log_n;
    }
    return log_n;
}

void bit_reverse_permute(std::vector<std::complex<double>>& x) {
    const size_t n = x.size();
    size_t j = 0;
    for (size_t i = 1; i < n; ++i) {
        size_t bit = n >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) {
            std::swap(x[i], x[j]);
        }
    }
}

using StageTwiddles = std::vector<std::vector<std::complex<double>>>;

const StageTwiddles& twiddles_for_size(size_t n) {
    static std::mutex cache_mutex;
    static std::unordered_map<size_t, StageTwiddles> cache;

    std::lock_guard lock(cache_mutex);
    const auto found = cache.find(n);
    if (found != cache.end()) {
        return found->second;
    }

    const size_t log_n = log2_power_of_two(n);
    StageTwiddles stages(log_n);
    for (size_t stage = 1; stage <= log_n; ++stage) {
        const size_t m = size_t{1} << stage;
        const size_t half_m = m / 2;
        stages[stage - 1].resize(half_m);
        for (size_t j = 0; j < half_m; ++j) {
            const double angle = -2.0 * M_PI * static_cast<double>(j) / static_cast<double>(m);
            stages[stage - 1][j] = std::complex<double>(std::cos(angle), std::sin(angle));
        }
    }

    return cache.emplace(n, std::move(stages)).first->second;
}

void fft_inplace(std::vector<std::complex<double>>& x) {
    const size_t n = x.size();
    if (n <= 1) {
        return;
    }

    bit_reverse_permute(x);

    const auto& stages = twiddles_for_size(n);
    const size_t log_n = stages.size();
    for (size_t stage = 0; stage < log_n; ++stage) {
        const size_t m = size_t{1} << (stage + 1);
        const size_t half_m = m / 2;
        const auto& tw = stages[stage];
        for (size_t k = 0; k < n; k += m) {
            for (size_t j = 0; j < half_m; ++j) {
                const std::complex<double> t = tw[j] * x[k + j + half_m];
                const std::complex<double> u = x[k + j];
                x[k + j] = u + t;
                x[k + j + half_m] = u - t;
            }
        }
    }
}

std::vector<std::complex<double>> dft_direct(const std::vector<std::complex<double>>& x) {
    const size_t n = x.size();
    std::vector<std::complex<double>> out(n);
    for (size_t k = 0; k < n; ++k) {
        std::complex<double> sum(0.0, 0.0);
        for (size_t t = 0; t < n; ++t) {
            const double angle = -2.0 * M_PI * static_cast<double>(k * t) / static_cast<double>(n);
            sum += x[t] * std::complex<double>(std::cos(angle), std::sin(angle));
        }
        out[k] = sum;
    }
    return out;
}

std::vector<std::complex<double>> fft_transform(std::vector<std::complex<double>> x) {
    const size_t n = x.size();
    if (n <= 1) {
        return x;
    }
    if (!is_power_of_two(n)) {
        return dft_direct(x);
    }

    fft_inplace(x);
    return x;
}

size_t next_power_of_two(size_t n) {
    size_t p = 1;
    while (p < n) {
        p <<= 1;
    }
    return p;
}

std::vector<std::complex<double>> ifft_transform(std::vector<std::complex<double>> x) {
    const size_t n = x.size();
    for (size_t i = 0; i < n; ++i) {
        x[i] = std::conj(x[i]);
    }
    x = fft_transform(std::move(x));
    const double inv_n = 1.0 / static_cast<double>(n);
    for (size_t i = 0; i < n; ++i) {
        x[i] = std::conj(x[i]) * inv_n;
    }
    return x;
}

// Real-input FFT via N/2-point complex FFT (two-real-sequences packing).
// Post-processing follows the standard even/odd deinterleave + untangle step
// used by FFTW/phastft (twiddle W^k pre-multiplied by 0.5).
std::vector<std::complex<double>> rfft_half_transform(const std::vector<double>& x, size_t full_len) {
    const size_t half = full_len / 2;
    const size_t out_len = half + 1;

    if (full_len == 1) {
        return {{x.empty() ? 0.0 : x[0], 0.0}};
    }

    std::vector<std::complex<double>> packed(half);
    for (size_t j = 0; j < half; ++j) {
        const double re = (2 * j < x.size()) ? x[2 * j] : 0.0;
        const double im = (2 * j + 1 < x.size()) ? x[2 * j + 1] : 0.0;
        packed[j] = {re, im};
    }

    auto z = fft_transform(std::move(packed));

    std::vector<std::complex<double>> out(out_len);
    const double a0 = z[0].real();
    const double b0 = z[0].imag();
    out[0] = {a0 + b0, 0.0};
    out[half] = {a0 - b0, 0.0};

    const size_t q = half / 2;
    for (size_t k = 1; k < q; ++k) {
        const size_t mirror = half - k;
        const double a = z[k].real();
        const double b = z[k].imag();
        const double c = z[mirror].real();
        const double d = z[mirror].imag();
        const double s_re = 0.5 * (a + c);
        const double s_im = 0.5 * (b - d);
        const double t_re = b + d;
        const double t_im = c - a;
        const double angle = -2.0 * M_PI * static_cast<double>(k) / static_cast<double>(full_len);
        const double wkr = 0.5 * std::cos(angle);
        const double wki = 0.5 * std::sin(angle);
        const double wzr = wkr * t_re - wki * t_im;
        const double wzi = wkr * t_im + wki * t_re;
        out[k] = {s_re + wzr, s_im + wzi};
        out[mirror] = {s_re - wzr, wzi - s_im};
    }

    if (q >= 1 && half >= 2) {
        const double a = z[q].real();
        const double b = z[q].imag();
        const double angle = -2.0 * M_PI * static_cast<double>(q) / static_cast<double>(full_len);
        const double wkr = 0.5 * std::cos(angle);
        const double wki = 0.5 * std::sin(angle);
        out[q] = {a + 2.0 * wkr * b, 2.0 * wki * b};
    }

    return out;
}

std::vector<std::complex<double>> hermitian_extend_rfft_spectrum(
    const std::vector<std::complex<double>>& x, size_t full_len) {
    const size_t half = full_len / 2;
    std::vector<std::complex<double>> spectrum(half + 1);
    for (size_t i = 0; i < x.size() && i <= half; ++i) {
        spectrum[i] = x[i];
    }
    for (size_t i = x.size(); i <= half; ++i) {
        spectrum[i] = std::conj(spectrum[full_len - i]);
    }
    return spectrum;
}

std::vector<double> irfft_half_transform(
    const std::vector<std::complex<double>>& x, size_t full_len) {
    const size_t half = full_len / 2;

    if (full_len == 1) {
        return {x.empty() ? 0.0 : x[0].real()};
    }

    const auto spectrum = hermitian_extend_rfft_spectrum(x, full_len);

    std::vector<std::complex<double>> z(half);

    {
        const double re_first = spectrum[0].real();
        const double im_first = spectrum[0].imag();
        const double re_second = spectrum[half].real();
        const double im_second = -spectrum[half].imag();
        const double zx_re = 0.5 * (re_first + re_second);
        const double zx_im = 0.5 * (im_first + im_second);
        const double dr = re_first - re_second;
        const double di = im_first - im_second;
        const double c_h = 0.5;
        const double s_h = 0.0;
        const double zy_re = c_h * dr + s_h * di;
        const double zy_im = c_h * di - s_h * dr;
        z[0] = {zx_re - zy_im, zx_im + zy_re};
    }

    for (size_t k = 1; k < half; ++k) {
        const size_t mirror = half - k;
        const double re_first = spectrum[k].real();
        const double im_first = spectrum[k].imag();
        const double re_second = spectrum[mirror].real();
        const double im_second = -spectrum[mirror].imag();
        const double zx_re = 0.5 * (re_first + re_second);
        const double zx_im = 0.5 * (im_first + im_second);
        const double dr = re_first - re_second;
        const double di = im_first - im_second;
        const double angle = -2.0 * M_PI * static_cast<double>(k) / static_cast<double>(full_len);
        const double c_h = 0.5 * std::cos(angle);
        const double s_h = 0.5 * std::sin(angle);
        const double zy_re = c_h * dr + s_h * di;
        const double zy_im = c_h * di - s_h * dr;
        z[k] = {zx_re - zy_im, zx_im + zy_re};
    }

    auto packed = ifft_transform(std::move(z));

    std::vector<double> out(full_len);
    for (size_t j = 0; j < half; ++j) {
        out[2 * j] = packed[j].real();
        out[2 * j + 1] = packed[j].imag();
    }
    return out;
}

} // namespace

Result<std::vector<std::complex<double>>> fft(const std::vector<double>& x) {
    if (x.empty()) {
        return std::vector<std::complex<double>>{};
    }

    const size_t n = next_power_of_two(x.size());
    const auto decision = decide(n, ExecPolicy::AUTO);
    if (decision.backend == Backend::CUDA) {
        if (auto gpu = cuda::fft(x)) {
            return gpu;
        }
    }

    std::vector<std::complex<double>> input(n);
    for (size_t i = 0; i < x.size(); ++i) {
        input[i] = std::complex<double>(x[i], 0.0);
    }

    return fft_transform(std::move(input));
}

Result<std::vector<double>> ifft(const std::vector<std::complex<double>>& x) {
    if (x.empty()) {
        return std::vector<double>{};
    }

    std::vector<std::complex<double>> conj(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        conj[i] = std::conj(x[i]);
    }

    auto spectrum = fft_transform(std::move(conj));
    std::vector<double> out(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        out[i] = spectrum[i].real() / static_cast<double>(x.size());
    }
    return out;
}

Result<std::vector<std::complex<double>>> fft2(const std::vector<std::complex<double>>& data) {
    if (data.empty()) {
        return std::vector<std::complex<double>>{};
    }

    const size_t n = data.size();
    size_t cols = 1;
    const size_t root = static_cast<size_t>(std::sqrt(static_cast<double>(n)));
    for (size_t c = root; c >= 1; --c) {
        if (n % c == 0) {
            cols = c;
            break;
        }
    }
    const size_t rows = n / cols;

    std::vector<std::complex<double>> out = data;

    for (size_t r = 0; r < rows; ++r) {
        std::vector<std::complex<double>> row(cols);
        for (size_t c = 0; c < cols; ++c) {
            row[c] = out[r * cols + c];
        }
        row = fft_transform(std::move(row));
        for (size_t c = 0; c < cols; ++c) {
            out[r * cols + c] = row[c];
        }
    }

    for (size_t c = 0; c < cols; ++c) {
        std::vector<std::complex<double>> col(rows);
        for (size_t r = 0; r < rows; ++r) {
            col[r] = out[r * cols + c];
        }
        col = fft_transform(std::move(col));
        for (size_t r = 0; r < rows; ++r) {
            out[r * cols + c] = col[r];
        }
    }

    return out;
}

Result<std::vector<std::complex<double>>> ifft2(const std::vector<std::complex<double>>& data) {
    if (data.empty()) {
        return std::vector<std::complex<double>>{};
    }

    const size_t n = data.size();
    size_t cols = 1;
    const size_t root = static_cast<size_t>(std::sqrt(static_cast<double>(n)));
    for (size_t c = root; c >= 1; --c) {
        if (n % c == 0) {
            cols = c;
            break;
        }
    }
    const size_t rows = n / cols;

    std::vector<std::complex<double>> out = data;

    for (size_t r = 0; r < rows; ++r) {
        std::vector<std::complex<double>> row(cols);
        for (size_t c = 0; c < cols; ++c) {
            row[c] = out[r * cols + c];
        }
        row = ifft_transform(std::move(row));
        for (size_t c = 0; c < cols; ++c) {
            out[r * cols + c] = row[c];
        }
    }

    for (size_t c = 0; c < cols; ++c) {
        std::vector<std::complex<double>> col(rows);
        for (size_t r = 0; r < rows; ++r) {
            col[r] = out[r * cols + c];
        }
        col = ifft_transform(std::move(col));
        for (size_t r = 0; r < rows; ++r) {
            out[r * cols + c] = col[r];
        }
    }

    return out;
}

Result<std::vector<std::complex<double>>> dft(std::span<const double> data) {
    std::vector<double> copy(data.begin(), data.end());
    return fft(copy);
}

Result<std::vector<std::complex<double>>> rfft(const std::vector<double>& x) {
    if (x.empty()) {
        return std::vector<std::complex<double>>{};
    }

    const size_t full_len = next_power_of_two(x.size());
    return rfft_half_transform(x, full_len);
}

Result<std::vector<double>> irfft(const std::vector<std::complex<double>>& x, size_t n) {
    if (x.empty() || n == 0) {
        return std::vector<double>{};
    }
    const size_t full_len = next_power_of_two(n);
    return irfft_half_transform(x, full_len);
}

std::vector<std::complex<double>> fftshift(const std::vector<std::complex<double>>& x) {
    if (x.empty()) {
        return x;
    }
    const size_t n = x.size();
    const size_t half = n / 2;
    std::vector<std::complex<double>> out(n);
    for (size_t i = 0; i < n; ++i) {
        out[i] = x[(i + half) % n];
    }
    return out;
}

std::vector<std::complex<double>> ifftshift(const std::vector<std::complex<double>>& x) {
    if (x.empty()) {
        return x;
    }
    const size_t n = x.size();
    const size_t half = (n + 1) / 2;
    std::vector<std::complex<double>> out(n);
    for (size_t i = 0; i < n; ++i) {
        out[i] = x[(i + half) % n];
    }
    return out;
}

std::vector<double> fftfreq(size_t n, double d) {
    if (n == 0) {
        return std::vector<double>{};
    }
    std::vector<double> out(n);
    const double scale = 1.0 / (static_cast<double>(n) * d);
    const size_t positive_count = (n - 1) / 2 + 1;
    for (size_t i = 0; i < positive_count; ++i) {
        out[i] = static_cast<double>(i) * scale;
    }
    for (size_t i = positive_count; i < n; ++i) {
        out[i] = -static_cast<double>(n - i) * scale;
    }
    return out;
}

std::vector<double> rfftfreq(size_t n, double d) {
    if (n == 0) {
        return std::vector<double>{};
    }
    const size_t out_len = n / 2 + 1;
    std::vector<double> out(out_len);
    const double scale = 1.0 / (static_cast<double>(n) * d);
    for (size_t i = 0; i < out_len; ++i) {
        out[i] = static_cast<double>(i) * scale;
    }
    return out;
}

Result<std::vector<double>> dct2(const std::vector<double>& x) {
    const size_t n = x.size();
    if (n == 0) {
        return std::vector<double>{};
    }
    std::vector<double> out(n);
    const double scale0 = 1.0 / std::sqrt(static_cast<double>(n));
    const double scale = std::sqrt(2.0 / static_cast<double>(n));
    for (size_t k = 0; k < n; ++k) {
        double sum = 0.0;
        for (size_t t = 0; t < n; ++t) {
            sum += x[t] * std::cos(M_PI * static_cast<double>((t + 0.5) * k) / static_cast<double>(n));
        }
        out[k] = (k == 0 ? scale0 : scale) * sum;
    }
    return out;
}

Result<std::vector<double>> idct2(const std::vector<double>& x) {
    const size_t n = x.size();
    if (n == 0) {
        return std::vector<double>{};
    }
    std::vector<double> out(n);
    const double scale0 = 1.0 / std::sqrt(static_cast<double>(n));
    const double scale = std::sqrt(2.0 / static_cast<double>(n));
    for (size_t t = 0; t < n; ++t) {
        double sum = scale0 * x[0];
        for (size_t k = 1; k < n; ++k) {
            sum += scale * x[k] * std::cos(M_PI * static_cast<double>((t + 0.5) * k) / static_cast<double>(n));
        }
        out[t] = sum;
    }
    return out;
}

Result<std::vector<double>> dst2(const std::vector<double>& x) {
    const size_t n = x.size();
    if (n == 0) {
        return std::vector<double>{};
    }
    std::vector<double> out(n);
    const double scale = std::sqrt(2.0 / static_cast<double>(n + 1));
    for (size_t k = 0; k < n; ++k) {
        double sum = 0.0;
        for (size_t t = 0; t < n; ++t) {
            sum += x[t] * std::sin(M_PI * static_cast<double>((t + 1) * (k + 1)) / static_cast<double>(n + 1));
        }
        out[k] = scale * sum;
    }
    return out;
}

// Orthonormal DST-I (sin basis with sqrt(2/(n+1)) scaling) is symmetric and
// orthogonal, hence self-inverse: applying dst2 twice recovers the input.
Result<std::vector<double>> idst2(const std::vector<double>& x) {
    return dst2(x);
}

std::complex<double> goertzel(std::span<const double> x, double f, double fs) {
    if (x.empty() || fs <= 0.0 || f < 0.0 || f > fs / 2.0) {
        return std::complex<double>(0.0, 0.0);
    }

    const size_t n = x.size();
    const double k = std::round(f / fs * static_cast<double>(n));
    const double w = 2.0 * M_PI * k / static_cast<double>(n);
    const double coeff = 2.0 * std::cos(w);

    double s_prev = 0.0;
    double s_prev2 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double s = x[i] + coeff * s_prev - s_prev2;
        s_prev2 = s_prev;
        s_prev = s;
    }

    const double real = s_prev * std::cos(w) - s_prev2;
    const double imag = s_prev * std::sin(w);
    return std::complex<double>(real, imag);
}

} // namespace ms
