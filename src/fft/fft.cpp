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
    auto full = fft(x);
    if (!full) {
        return std::unexpected(full.error());
    }
    const size_t n = full->size();
    const size_t out_len = n / 2 + 1;
    std::vector<std::complex<double>> out(out_len);
    for (size_t i = 0; i < out_len; ++i) {
        out[i] = (*full)[i];
    }
    return out;
}

Result<std::vector<double>> irfft(const std::vector<std::complex<double>>& x, size_t n) {
    if (x.empty() || n == 0) {
        return std::vector<double>{};
    }
    const size_t full_len = next_power_of_two(n);
    std::vector<std::complex<double>> full(full_len);
    for (size_t i = 0; i < x.size() && i < full_len; ++i) {
        full[i] = x[i];
    }
    for (size_t i = x.size(); i < full_len; ++i) {
        const size_t k = full_len - i;
        full[i] = std::conj(full[k]);
    }
    return ifft(full);
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
