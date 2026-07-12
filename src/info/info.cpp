#define _USE_MATH_DEFINES
#include "ms/info/info.hpp"
#include <algorithm>
#include <cmath>
#include <map>
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

double normalized_entropy(std::span<const double> p) {
    if (p.size() <= 1) return 0.0;
    const double h_max = std::log2(static_cast<double>(p.size()));
    return entropy(p, 2.0) / h_max;
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

// Validates that W is a non-empty, rectangular channel matrix whose rows
// are (approximately) valid probability distributions.
static bool is_valid_channel_matrix(const std::vector<std::vector<double>>& W) {
    if (W.empty()) return false;
    const size_t n_outputs = W[0].size();
    if (n_outputs == 0) return false;
    for (const auto& row : W) {
        if (row.size() != n_outputs) return false;
        double sum = 0.0;
        for (double v : row) {
            if (v < 0.0) return false;
            sum += v;
        }
        if (std::abs(sum - 1.0) > 1e-6) return false;
    }
    return true;
}

ChannelCapacityResult channel_capacity(const std::vector<std::vector<double>>& W,
                                        double eps, int max_iter) {
    ChannelCapacityResult result;
    if (!is_valid_channel_matrix(W)) return result;

    const size_t n_inputs = W.size();
    const size_t n_outputs = W[0].size();

    // Single-input channel: capacity is trivially 0 bits.
    if (n_inputs == 1) {
        result.input_distribution = {1.0};
        return result;
    }

    std::vector<double> p(n_inputs, 1.0 / static_cast<double>(n_inputs));
    double capacity = 0.0;

    for (int iter = 0; iter < max_iter; ++iter) {
        // p(y) = sum_x p(x) W(y|x)
        std::vector<double> py(n_outputs, 0.0);
        for (size_t x = 0; x < n_inputs; ++x)
            for (size_t y = 0; y < n_outputs; ++y)
                py[y] += p[x] * W[x][y];

        // c(x) = sum_y W(y|x) log2( W(y|x) / p(y) )
        std::vector<double> c(n_inputs, 0.0);
        for (size_t x = 0; x < n_inputs; ++x) {
            double cx = 0.0;
            for (size_t y = 0; y < n_outputs; ++y) {
                if (W[x][y] > 0.0 && py[y] > 0.0)
                    cx += W[x][y] * log_base(W[x][y] / py[y], 2.0);
            }
            c[x] = cx;
        }

        // Capacity estimate: I(X;Y) = sum_x p(x) c(x)
        double new_capacity = 0.0;
        for (size_t x = 0; x < n_inputs; ++x) new_capacity += p[x] * c[x];

        // Update p(x) <- p(x) * 2^c(x), then normalise.
        std::vector<double> new_p(n_inputs);
        double norm = 0.0;
        for (size_t x = 0; x < n_inputs; ++x) {
            new_p[x] = p[x] * std::exp2(c[x]);
            norm += new_p[x];
        }
        if (norm > 0.0)
            for (size_t x = 0; x < n_inputs; ++x) new_p[x] /= norm;

        bool converged = std::abs(new_capacity - capacity) < eps;
        p = new_p;
        capacity = new_capacity;
        if (converged) break;
    }

    result.capacity = capacity;
    result.input_distribution = p;
    return result;
}

double blahut_arimoto(const std::vector<std::vector<double>>& W,
                       double eps, int max_iter) {
    return channel_capacity(W, eps, max_iter).capacity;
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

static int discretize_value(double value, double vmin, double vmax, int bins) {
    if (bins <= 0) return 0;
    const double width =
        (vmax > vmin) ? (vmax - vmin) / static_cast<double>(bins) : 1.0;
    int idx = static_cast<int>((value - vmin) / width);
    if (idx >= bins) idx = bins - 1;
    if (idx < 0) idx = 0;
    return idx;
}

static double range_min(std::span<const double> data) {
    return *std::min_element(data.begin(), data.end());
}

static double range_max(std::span<const double> data) {
    return *std::max_element(data.begin(), data.end());
}

double transfer_entropy(const std::vector<double>& x,
                        const std::vector<double>& y, int bins, int lag) {
    if (x.size() != y.size() || lag < 1 || bins < 1) return 0.0;

    const size_t n = x.size();
    if (n < static_cast<size_t>(lag) + 1) return 0.0;

    const double xmin = range_min(x);
    const double xmax = range_max(x);
    const double ymin = range_min(y);
    const double ymax = range_max(y);

    const size_t n_samples = n - static_cast<size_t>(lag);

    // Joint p(y_t, y_{t+lag}) for H(y_{t+lag}|y_t): rows=y_t, cols=y_{t+lag}.
    std::vector<double> p_yt_yfuture(static_cast<size_t>(bins * bins), 0.0);
    // Joint p(y_t, x_t) marginal for H(y_{t+lag}|y_t, x_t).
    std::vector<double> p_yt_xt(static_cast<size_t>(bins * bins), 0.0);
    // Joint p(y_t, x_t, y_{t+lag}): index = iy_past * bins * bins + ix * bins + iy_future.
    std::vector<double> p_yt_xt_yfuture(
        static_cast<size_t>(bins * bins * bins), 0.0);

    for (size_t t = 0; t < n_samples; ++t) {
        const int iy_past =
            discretize_value(y[t], ymin, ymax, bins);
        const int iy_future =
            discretize_value(y[t + static_cast<size_t>(lag)], ymin, ymax, bins);
        const int ix =
            discretize_value(x[t], xmin, xmax, bins);

        p_yt_yfuture[static_cast<size_t>(iy_past * bins + iy_future)] += 1.0;
        p_yt_xt[static_cast<size_t>(iy_past * bins + ix)] += 1.0;
        p_yt_xt_yfuture[static_cast<size_t>(
            iy_past * bins * bins + ix * bins + iy_future)] += 1.0;
    }

    const double inv_n = 1.0 / static_cast<double>(n_samples);
    for (double& v : p_yt_yfuture) v *= inv_n;
    for (double& v : p_yt_xt) v *= inv_n;
    for (double& v : p_yt_xt_yfuture) v *= inv_n;

    // H(y_{t+lag}|y_t) via conditional_entropy(rows=y_t, cols=y_{t+lag}).
    const double h_future_given_past =
        conditional_entropy(p_yt_yfuture, bins, bins, 2.0);

    // H(y_{t+lag}|y_t, x_t) = H(y_t, x_t, y_{t+lag}) - H(y_t, x_t).
    const double h_joint_3d = joint_entropy(p_yt_xt_yfuture, bins * bins, bins, 2.0);
    const double h_joint_yt_xt = joint_entropy(p_yt_xt, bins, bins, 2.0);
    const double h_future_given_past_and_x = h_joint_3d - h_joint_yt_xt;

    double te = h_future_given_past - h_future_given_past_and_x;
    if (te < 0.0) te = 0.0;
    return te;
}

double permutation_entropy(std::span<const double> x, int order, int delay,
                           bool normalize) {
    if (order < 2 || delay < 1) return 0.0;

    const size_t n = x.size();
    const size_t span_needed =
        static_cast<size_t>(order - 1) * static_cast<size_t>(delay) + 1;
    if (n < span_needed) return 0.0;

    const size_t n_windows = n - span_needed + 1;

    // Frequency count of each ordinal pattern ("position-of-rank" encoding:
    // pattern[k] = local window position holding the k-th smallest value).
    std::map<std::vector<int>, long long> counts;
    std::vector<int> idx(static_cast<size_t>(order));
    for (size_t start = 0; start < n_windows; ++start) {
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int a, int b) {
            double va = x[start + static_cast<size_t>(a) * static_cast<size_t>(delay)];
            double vb = x[start + static_cast<size_t>(b) * static_cast<size_t>(delay)];
            return va < vb;
        });
        counts[idx] += 1;
    }

    std::vector<double> probs;
    probs.reserve(counts.size());
    const double total = static_cast<double>(n_windows);
    for (const auto& kv : counts)
        probs.push_back(static_cast<double>(kv.second) / total);

    const double h = entropy(probs, 2.0);
    if (!normalize) return h;

    double log2_factorial = 0.0;
    for (int i = 2; i <= order; ++i) log2_factorial += std::log2(static_cast<double>(i));
    if (log2_factorial <= 0.0) return 0.0;
    return h / log2_factorial;
}

} // namespace info
} // namespace ms
