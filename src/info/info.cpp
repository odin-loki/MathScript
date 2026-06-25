#define _USE_MATH_DEFINES
#include "ms/info/info.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_map>
#include <vector>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_E
#define M_E 2.71828182845904523536
#endif

namespace ms {
namespace info {

static double log_base(double x, double base) {
    return std::log(x) / std::log(base);
}

double entropy(std::span<const double> p, double base) {
    double h = 0.0;
    for (double pi : p)
        if (pi > 0.0) h -= pi * log_base(pi, base);
    return h;
}

double joint_entropy(std::span<const double> pxy, int rows, int cols,
                     double base) {
    double h = 0.0;
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j) {
            double v = pxy[static_cast<size_t>(i * cols + j)];
            if (v > 0.0) h -= v * log_base(v, base);
        }
    return h;
}

double conditional_entropy(std::span<const double> pxy, int rows, int cols,
                           double base) {
    // H(Y|X) = H(X,Y) - H(X)
    std::vector<double> px(rows, 0.0);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            px[i] += pxy[static_cast<size_t>(i * cols + j)];
    return joint_entropy(pxy, rows, cols, base) - entropy(px, base);
}

double mutual_info(std::span<const double> pxy, int rows, int cols,
                   double base) {
    // I(X;Y) = H(X) + H(Y) - H(X,Y)
    std::vector<double> px(rows, 0.0), py(cols, 0.0);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j) {
            double v = pxy[static_cast<size_t>(i * cols + j)];
            px[i] += v; py[j] += v;
        }
    return entropy(px, base) + entropy(py, base) -
           joint_entropy(pxy, rows, cols, base);
}

double cross_entropy(std::span<const double> p, std::span<const double> q,
                     double base) {
    double h = 0.0;
    for (size_t i = 0; i < p.size(); ++i)
        if (p[i] > 0.0 && q[i] > 0.0)
            h -= p[i] * log_base(q[i], base);
    return h;
}

double kl_divergence(std::span<const double> p, std::span<const double> q,
                     double base) {
    double kl = 0.0;
    for (size_t i = 0; i < p.size(); ++i)
        if (p[i] > 0.0 && q[i] > 0.0)
            kl += p[i] * log_base(p[i] / q[i], base);
    return kl;
}

double js_divergence(std::span<const double> p, std::span<const double> q,
                     double base) {
    std::vector<double> m(p.size());
    for (size_t i = 0; i < p.size(); ++i)
        m[i] = 0.5 * (p[i] + q[i]);
    return 0.5 * kl_divergence(p, m, base) + 0.5 * kl_divergence(q, m, base);
}

double renyi_entropy(std::span<const double> p, double alpha, double base) {
    if (std::abs(alpha - 1.0) < 1e-12) return entropy(p, base);
    double sum = 0.0;
    for (double pi : p) if (pi > 0.0) sum += std::pow(pi, alpha);
    return log_base(sum, base) / (1.0 - alpha);
}

double tsallis_entropy(std::span<const double> p, double q_param) {
    if (std::abs(q_param - 1.0) < 1e-12) {
        double h = 0.0;
        for (double pi : p) if (pi > 0.0) h -= pi * std::log(pi);
        return h;
    }
    double sum = 0.0;
    for (double pi : p) if (pi > 0.0) sum += std::pow(pi, q_param);
    return (1.0 - sum) / (q_param - 1.0);
}

double hellinger_dist(std::span<const double> p, std::span<const double> q) {
    double sum = 0.0;
    for (size_t i = 0; i < p.size(); ++i)
        sum += std::pow(std::sqrt(p[i]) - std::sqrt(q[i]), 2.0);
    return std::sqrt(sum) / std::sqrt(2.0);
}

double tv_distance(std::span<const double> p, std::span<const double> q) {
    double tv = 0.0;
    for (size_t i = 0; i < p.size(); ++i)
        tv += std::abs(p[i] - q[i]);
    return 0.5 * tv;
}

double channel_capacity_bsc(double p_error) {
    if (p_error <= 0.0 || p_error >= 1.0) return (p_error <= 0.0) ? 1.0 : 0.0;
    double h = -p_error * std::log2(p_error) -
               (1.0 - p_error) * std::log2(1.0 - p_error);
    return 1.0 - h;
}

double channel_capacity_bec(double epsilon) {
    return 1.0 - epsilon;
}

double shannon_hartley(double bandwidth_hz, double snr_linear) {
    return bandwidth_hz * std::log2(1.0 + snr_linear);
}

double rate_distortion_gaussian(double variance, double distortion) {
    if (distortion >= variance) return 0.0;
    return 0.5 * std::log2(variance / distortion);
}

double source_coding_rate(std::span<const double> p) {
    return entropy(p, 2.0); // Shannon's source coding theorem
}

double lz_complexity(std::span<const int> seq) {
    if (seq.empty()) return 0.0;
    size_t n = seq.size();
    size_t c = 1; // complexity counter
    size_t i = 0, k = 1, l = 1;
    size_t kmax = 1, k_init = 1;

    while (k + l <= n) {
        // Check if seq[k..k+l-1] exists in seq[0..k-1]
        bool found = false;
        for (size_t start = 0; start + l <= k; ++start) {
            bool match = true;
            for (size_t j = 0; j < l; ++j) {
                if (seq[start + j] != seq[k + j]) { match = false; break; }
            }
            if (match) { found = true; break; }
        }
        if (found) {
            ++l;
        } else {
            ++c;
            k += l;
            l = 1;
        }
        (void)i; (void)kmax; (void)k_init;
    }
    // Normalised complexity
    return static_cast<double>(c) / (static_cast<double>(n) / std::log2(static_cast<double>(n) + 1));
}

double redundancy(std::span<const double> p) {
    double h_max = std::log2(static_cast<double>(p.size()));
    return h_max - entropy(p, 2.0);
}

double efficiency(std::span<const double> p) {
    double h_max = std::log2(static_cast<double>(p.size()));
    if (h_max <= 0.0) return 1.0;
    return entropy(p, 2.0) / h_max;
}

double differential_entropy_gaussian(double sigma) {
    return 0.5 * std::log(2.0 * M_PI * M_E * sigma * sigma);
}

double differential_entropy_uniform(double a, double b) {
    return std::log(std::abs(b - a));
}

double sample_entropy(std::span<const double> x, int m, double r) {
    // ApEn / SampEn: count template matches
    size_t n = x.size();
    if (n < static_cast<size_t>(m + 1)) return 0.0;
    auto count_matches = [&](int len) -> double {
        double cnt = 0.0;
        for (size_t i = 0; i < n - len; ++i) {
            for (size_t j = i + 1; j < n - len; ++j) {
                double d = 0.0;
                for (int k = 0; k < len; ++k)
                    d = std::max(d, std::abs(x[i + k] - x[j + k]));
                if (d < r) cnt += 1.0;
            }
        }
        return cnt;
    };
    double A = count_matches(m + 1);
    double B = count_matches(m);
    if (B <= 0.0) return 0.0;
    return -std::log(A / B);
}

} // namespace info
} // namespace ms
