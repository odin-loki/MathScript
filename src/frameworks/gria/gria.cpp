#include "ms/frameworks/gria/gria.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <string>

namespace ms::gria {

namespace {

std::map<std::string, double>& hint_registry() {
    static std::map<std::string, double> hints;
    return hints;
}

} // namespace

double entropy(std::span<const double> data, size_t bins) {
    if (data.empty() || bins == 0) {
        return 0.0;
    }
    const auto [min_it, max_it] = std::minmax_element(data.begin(), data.end());
    const double vmin = *min_it;
    const double vmax = *max_it;
    const double width = (vmax - vmin) > 0.0 ? (vmax - vmin) / static_cast<double>(bins) : 1.0;

    std::vector<double> counts(bins, 0.0);
    for (double value : data) {
        size_t idx = static_cast<size_t>((value - vmin) / width);
        if (idx >= bins) {
            idx = bins - 1;
        }
        counts[idx] += 1.0;
    }

    const double total = static_cast<double>(data.size());
    double h = 0.0;
    for (double count : counts) {
        if (count <= 0.0) {
            continue;
        }
        const double p = count / total;
        h -= p * std::log2(p);
    }
    return h;
}

double matrix_alpha(const Matrix<double>& x, const Matrix<double>& fx) {
    std::vector<double> flat_x;
    std::vector<double> flat_fx;
    flat_x.reserve(x.rows() * x.cols());
    flat_fx.reserve(fx.rows() * fx.cols());
    for (size_t i = 0; i < x.rows(); ++i) {
        for (size_t j = 0; j < x.cols(); ++j) {
            flat_x.push_back(x(i, j));
        }
    }
    for (size_t i = 0; i < fx.rows(); ++i) {
        for (size_t j = 0; j < fx.cols(); ++j) {
            flat_fx.push_back(fx(i, j));
        }
    }
    const double h_in = entropy(flat_x);
    if (h_in <= 1e-12) {
        return 0.0;
    }
    const double alpha = 1.0 - entropy(flat_fx) / h_in;
    return std::clamp(alpha, 0.0, 1.0);
}

bool is_critical(double alpha, double tolerance) {
    return std::abs(alpha - 0.5) <= tolerance;
}

ComputeClass classify(double alpha) {
    if (alpha < 0.35) {
        return ComputeClass::Reversible;
    }
    if (alpha > 0.65) {
        return ComputeClass::Irreversible;
    }
    return ComputeClass::Critical;
}

namespace gf2n {

uint64_t mul(uint64_t a, uint64_t b, uint64_t poly) {
    uint64_t result = 0;
    while (b != 0) {
        if (b & 1ULL) {
            result ^= a;
        }
        b >>= 1ULL;
        const bool carry = (a & (1ULL << 63)) != 0;
        a <<= 1ULL;
        if (carry) {
            a ^= poly;
        }
    }
    return result;
}

uint64_t pow(uint64_t a, uint64_t exp, uint64_t poly) {
    uint64_t result = 1;
    uint64_t base = a;
    while (exp > 0) {
        if (exp & 1ULL) {
            result = mul(result, base, poly);
        }
        base = mul(base, base, poly);
        exp >>= 1ULL;
    }
    return result;
}

uint64_t inv(uint64_t a, uint64_t poly) {
    return pow(a, (1ULL << 16) - 2ULL, poly);
}

std::vector<uint64_t> generate_field(int n) {
    std::vector<uint64_t> field;
    if (n <= 0 || n > 16) {
        return field;
    }
    const uint64_t modulus = 1ULL << static_cast<unsigned>(n);
    field.reserve(static_cast<size_t>(modulus));
    for (uint64_t i = 0; i < modulus; ++i) {
        field.push_back(i);
    }
    return field;
}

} // namespace gf2n

namespace ca {

std::vector<uint8_t> step(std::span<const uint8_t> state, uint8_t rule) {
    std::vector<uint8_t> next(state.size(), 0);
    for (size_t i = 0; i < state.size(); ++i) {
        const size_t left = i == 0 ? state.size() - 1 : i - 1;
        const size_t right = (i + 1) % state.size();
        const uint8_t pattern =
            static_cast<uint8_t>((state[left] << 2) | (state[i] << 1) | state[right]);
        next[i] = static_cast<uint8_t>((rule >> pattern) & 1U);
    }
    return next;
}

double langton_lambda(uint8_t rule) {
    int zeros = 0;
    for (int i = 0; i < 8; ++i) {
        zeros += ((rule >> i) & 1U) == 0 ? 1 : 0;
    }
    return static_cast<double>(zeros) / 8.0;
}

double alpha_ca(uint8_t rule, size_t steps, size_t width) {
    std::vector<uint8_t> state(width, 0);
    state[width / 2] = 1;
    std::vector<double> bits;
    bits.reserve(steps * width);
    for (size_t s = 0; s < steps; ++s) {
        for (uint8_t bit : state) {
            bits.push_back(static_cast<double>(bit));
        }
        state = step(state, rule);
    }
    return compute_alpha<double>(bits, [](std::span<const double> input) {
        std::vector<double> out;
        out.reserve(input.size());
        for (double v : input) {
            out.push_back(v > 0.5 ? 1.0 : 0.0);
        }
        return out;
    });
}

size_t hamming_distance(std::span<const uint8_t> a, std::span<const uint8_t> b) {
    const size_t common = std::min(a.size(), b.size());
    size_t distance = 0;
    for (size_t i = 0; i < common; ++i) {
        if (a[i] != b[i]) {
            ++distance;
        }
    }
    distance += (a.size() > b.size()) ? (a.size() - b.size()) : (b.size() - a.size());
    return distance;
}

std::vector<size_t> divergence_trajectory(std::vector<uint8_t> config_a,
                                           std::vector<uint8_t> config_b,
                                           uint8_t rule,
                                           int n_steps) {
    std::vector<size_t> trajectory;
    trajectory.reserve(static_cast<size_t>(n_steps > 0 ? n_steps : 0) + 1);
    trajectory.push_back(hamming_distance(config_a, config_b));
    for (int i = 0; i < n_steps; ++i) {
        config_a = step(config_a, rule);
        config_b = step(config_b, rule);
        trajectory.push_back(hamming_distance(config_a, config_b));
    }
    return trajectory;
}

std::optional<size_t> settling_time(std::vector<uint8_t> config_a,
                                     std::vector<uint8_t> config_b,
                                     uint8_t rule,
                                     int n_steps) {
    if (hamming_distance(config_a, config_b) == 0) {
        return 0;
    }
    for (int i = 1; i <= n_steps; ++i) {
        config_a = step(config_a, rule);
        config_b = step(config_b, rule);
        if (hamming_distance(config_a, config_b) == 0) {
            return static_cast<size_t>(i);
        }
    }
    return std::nullopt;
}

} // namespace ca

namespace lfsr {

uint64_t step(uint64_t state, uint64_t poly) {
    const uint64_t bit = state & 1ULL;
    state >>= 1ULL;
    if (bit) {
        state ^= poly;
    }
    return state;
}

double alpha_lfsr(uint64_t poly, size_t steps) {
    uint64_t state = 1;
    std::vector<double> seq;
    seq.reserve(steps);
    for (size_t i = 0; i < steps; ++i) {
        seq.push_back(static_cast<double>(state & 1ULL));
        state = step(state, poly);
    }
    return compute_alpha<double>(seq, [](std::span<const double> input) {
        std::vector<double> out = {input.begin(), input.end()};
        std::reverse(out.begin(), out.end());
        return out;
    });
}

bool is_maximal(uint64_t poly, int n) {
    if (n <= 0 || n > 16) {
        return false;
    }
    const uint64_t mask = (1ULL << n) - 1ULL;
    uint64_t state = 1;
    for (uint64_t i = 0; i < mask; ++i) {
        state = step(state, poly) & mask;
        if (state == 1 && i > 0 && i < mask - 1) {
            return false;
        }
    }
    return state == 1;
}

} // namespace lfsr

void register_dispatch_hint(const std::string& op, double alpha) {
    hint_registry()[op] = alpha;
}

double dispatch_hint_alpha(const std::string& op) {
    const auto it = hint_registry().find(op);
    if (it == hint_registry().end()) {
        return -1.0;
    }
    return it->second;
}

namespace {

struct DefaultHints {
    DefaultHints() {
        register_dispatch_hint("matmul", 0.72);
        register_dispatch_hint("fft", 0.58);
        register_dispatch_hint("svd", 0.81);
    }
};

const DefaultHints kDefaultHints{};

} // namespace

} // namespace ms::gria
