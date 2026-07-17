#include "ms/stats/stats.hpp"
#include "ms/prob/prob.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <unordered_map>
#include <vector>

namespace ms {

namespace {

double f_distribution_upper_tail_p(double f_stat, int df1, int df2) {
    if (f_stat <= 0.0 || df1 < 1 || df2 < 1) {
        return 1.0;
    }
    return 1.0 - f_cdf(f_stat,
                       static_cast<double>(df1),
                       static_cast<double>(df2));
}

std::vector<double> average_ranks(const std::vector<double>& values) {
    const size_t n = values.size();
    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), 0u);
    std::sort(order.begin(), order.end(),
              [&](size_t i, size_t j) { return values[i] < values[j]; });
    std::vector<double> ranks(n);
    size_t i = 0;
    while (i < n) {
        size_t j = i + 1;
        while (j < n && values[order[j]] == values[order[i]]) {
            ++j;
        }
        const double avg_rank =
            0.5 * (static_cast<double>(i + 1) + static_cast<double>(j));
        for (size_t k = i; k < j; ++k) {
            ranks[order[k]] = avg_rank;
        }
        i = j;
    }
    return ranks;
}

} // namespace

double mean(std::span<const double> data) {
    if (data.empty()) {
        return 0.0;
    }
    return std::accumulate(data.begin(), data.end(), 0.0) / static_cast<double>(data.size());
}

double var(std::span<const double> data) {
    if (data.size() < 2) {
        return 0.0;
    }
    const double m = mean(data);
    double sum_sq = 0.0;
    for (double v : data) {
        sum_sq += (v - m) * (v - m);
    }
    return sum_sq / static_cast<double>(data.size() - 1);
}

double stddev(std::span<const double> data) {
    return std::sqrt(var(data));
}

double median(std::span<const double> data) {
    if (data.empty()) {
        return 0.0;
    }
    std::vector<double> sorted(data.begin(), data.end());
    std::sort(sorted.begin(), sorted.end());
    const size_t n = sorted.size();
    if (n % 2 == 0) {
        return (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
    }
    return sorted[n / 2];
}

double min_value(std::span<const double> data) {
    return *std::min_element(data.begin(), data.end());
}

double max_value(std::span<const double> data) {
    return *std::max_element(data.begin(), data.end());
}

double mode(std::span<const double> data) {
    if (data.empty()) {
        return 0.0;
    }
    std::unordered_map<double, size_t> counts;
    for (double v : data) {
        ++counts[v];
    }
    double best = data[0];
    size_t best_count = 0;
    for (const auto& [value, count] : counts) {
        if (count > best_count) {
            best = value;
            best_count = count;
        }
    }
    return best;
}

double percentile(std::span<const double> data, double p) {
    if (data.empty()) {
        return 0.0;
    }
    const size_t idx = static_cast<size_t>(
        (p / 100.0) * static_cast<double>(data.size() - 1));
    std::vector<double> scratch(data.begin(), data.end());
    std::nth_element(scratch.begin(), scratch.begin() + static_cast<std::ptrdiff_t>(idx),
                     scratch.end());
    return scratch[idx];
}

double ttest(std::span<const double> sample, double mu) {
    if (sample.size() < 2) {
        return 0.0;
    }
    const double m = mean(sample);
    const double s = stddev(sample);
    if (s == 0.0) {
        return 0.0;
    }
    return (m - mu) / (s / std::sqrt(static_cast<double>(sample.size())));
}

namespace {

bool weighted_inputs_valid(const std::vector<double>& x,
                           const std::vector<double>& w,
                           double& sum_w) {
    sum_w = 0.0;
    if (x.empty() || x.size() != w.size()) {
        return false;
    }
    for (double wi : w) {
        if (wi < 0.0) {
            return false;
        }
        sum_w += wi;
    }
    return sum_w > 0.0;
}

} // namespace

double weighted_mean(const std::vector<double>& x, const std::vector<double>& w) {
    if (x.empty() || x.size() != w.size()) {
        return 0.0;
    }
    double sum_w = 0.0;
    double sum_wx = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        if (w[i] < 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        sum_w += w[i];
        sum_wx += w[i] * x[i];
    }
    if (sum_w == 0.0) {
        return 0.0;
    }
    return sum_wx / sum_w;
}

double weighted_variance(const std::vector<double>& x,
                         const std::vector<double>& w,
                         bool sample) {
    if (x.size() < 2 || x.size() != w.size()) {
        return 0.0;
    }
    double sum_w = 0.0;
    if (!weighted_inputs_valid(x, w, sum_w)) {
        for (double wi : w) {
            if (wi < 0.0) {
                return std::numeric_limits<double>::quiet_NaN();
            }
        }
        return 0.0;
    }
    const double wmean = weighted_mean(x, w);
    double sum_w_sq_dev = 0.0;
    double sum_w2 = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        const double dev = x[i] - wmean;
        sum_w_sq_dev += w[i] * dev * dev;
        sum_w2 += w[i] * w[i];
    }
    if (sum_w_sq_dev == 0.0) {
        return 0.0;
    }
    double denom = sum_w;
    if (sample) {
        denom = sum_w - sum_w2 / sum_w;
        if (denom <= 0.0) {
            return 0.0;
        }
    }
    return sum_w_sq_dev / denom;
}

double weighted_correlation(const std::vector<double>& x,
                            const std::vector<double>& y,
                            const std::vector<double>& w) {
    if (x.size() != y.size() || x.size() != w.size() || x.empty()) {
        return 0.0;
    }
    double sum_w = 0.0;
    if (!weighted_inputs_valid(x, w, sum_w)) {
        for (double wi : w) {
            if (wi < 0.0) {
                return std::numeric_limits<double>::quiet_NaN();
            }
        }
        return 0.0;
    }
    const double mx = weighted_mean(x, w);
    const double my = weighted_mean(y, w);
    double num = 0.0;
    double dx = 0.0;
    double dy = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        const double a = x[i] - mx;
        const double b = y[i] - my;
        num += w[i] * a * b;
        dx += w[i] * a * a;
        dy += w[i] * b * b;
    }
    if (dx == 0.0 || dy == 0.0) {
        return 0.0;
    }
    return num / std::sqrt(dx * dy);
}

double correlation(std::span<const double> x, std::span<const double> y) {
    if (x.size() != y.size() || x.empty()) {
        return 0.0;
    }
    const double mx = mean(x);
    const double my = mean(y);
    double num = 0.0;
    double dx = 0.0;
    double dy = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        const double a = x[i] - mx;
        const double b = y[i] - my;
        num += a * b;
        dx += a * a;
        dy += b * b;
    }
    if (dx == 0.0 || dy == 0.0) {
        return 0.0;
    }
    return num / std::sqrt(dx * dy);
}

double partial_correlation(std::span<const double> x,
                           std::span<const double> y,
                           std::span<const double> z) {
    if (x.size() != y.size() || x.size() != z.size() || x.empty()) {
        return 0.0;
    }
    const double r_xy = correlation(x, y);
    const double r_xz = correlation(x, z);
    const double r_yz = correlation(y, z);
    const double denom =
        std::sqrt((1.0 - r_xz * r_xz) * (1.0 - r_yz * r_yz));
    if (denom == 0.0) {
        return 0.0;
    }
    return (r_xy - r_xz * r_yz) / denom;
}

double skewness(std::span<const double> data) {
    if (data.size() < 3) {
        return 0.0;
    }
    const double m = mean(data);
    const double s = stddev(data);
    if (s == 0.0) {
        return 0.0;
    }
    double sum = 0.0;
    for (double v : data) {
        const double z = (v - m) / s;
        sum += z * z * z;
    }
    return sum / static_cast<double>(data.size());
}

double kurtosis(std::span<const double> data) {
    if (data.size() < 4) {
        return 0.0;
    }
    const double m = mean(data);
    const double s = stddev(data);
    if (s == 0.0) {
        return 0.0;
    }
    double sum = 0.0;
    for (double v : data) {
        const double z = (v - m) / s;
        sum += z * z * z * z;
    }
    return sum / static_cast<double>(data.size()) - 3.0;
}

double two_sample_ttest(std::span<const double> a, std::span<const double> b) {
    if (a.size() < 2 || b.size() < 2) {
        return 0.0;
    }
    const double ma = mean(a);
    const double mb = mean(b);
    const double va = var(a);
    const double vb = var(b);
    const double na = static_cast<double>(a.size());
    const double nb = static_cast<double>(b.size());
    const double se = std::sqrt(va / na + vb / nb);
    if (se == 0.0) {
        return 0.0;
    }
    return (ma - mb) / se;
}

double ztest(std::span<const double> sample, double mu, double sigma) {
    if (sample.empty() || sigma == 0.0) {
        return 0.0;
    }
    return (mean(sample) - mu) / (sigma / std::sqrt(static_cast<double>(sample.size())));
}

LinearRegressionResult linear_regression(std::span<const double> x, std::span<const double> y) {
    LinearRegressionResult result{};
    if (x.size() != y.size() || x.size() < 2) {
        return result;
    }
    const double mx = mean(x);
    const double my = mean(y);
    double sxx = 0.0;
    double sxy = 0.0;
    double syy = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        const double dx = x[i] - mx;
        const double dy = y[i] - my;
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
    }
    if (sxx == 0.0) {
        return result;
    }
    result.slope = sxy / sxx;
    result.intercept = my - result.slope * mx;
    result.r_squared = (sxy * sxy) / (sxx * syy);
    return result;
}

double geometric_mean(std::span<const double> data) {
    if (data.empty()) return 0.0;
    double log_sum = 0.0;
    for (double v : data) {
        if (v <= 0.0) return std::numeric_limits<double>::quiet_NaN();
        log_sum += std::log(v);
    }
    return std::exp(log_sum / static_cast<double>(data.size()));
}

double harmonic_mean(std::span<const double> data) {
    if (data.empty()) return 0.0;
    double inv_sum = 0.0;
    for (double v : data) {
        if (v == 0.0) return 0.0;
        inv_sum += 1.0 / v;
    }
    return static_cast<double>(data.size()) / inv_sum;
}

double rms(std::span<const double> data) {
    if (data.empty()) return 0.0;
    double sum_sq = 0.0;
    for (double v : data) sum_sq += v * v;
    return std::sqrt(sum_sq / static_cast<double>(data.size()));
}

double mad(const std::vector<double>& x, bool scale) {
    if (x.empty()) {
        return 0.0;
    }
    const double med = median(x);
    std::vector<double> abs_dev(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        abs_dev[i] = std::abs(x[i] - med);
    }
    double result = median(abs_dev);
    if (scale) {
        result *= 1.4826;
    }
    return result;
}

double iqr(std::span<const double> data) {
    if (data.size() < 4) return 0.0;
    return percentile(data, 75.0) - percentile(data, 25.0);
}

double trimmed_mean(std::span<const double> data, double frac) {
    if (data.empty()) return 0.0;
    std::vector<double> sorted(data.begin(), data.end());
    std::sort(sorted.begin(), sorted.end());
    size_t trim = static_cast<size_t>(frac * static_cast<double>(sorted.size()));
    if (2 * trim >= sorted.size()) return median(data);
    return mean(std::span<const double>(sorted.data() + trim,
                                         sorted.size() - 2 * trim));
}

double spearman(std::span<const double> x, std::span<const double> y) {
    if (x.size() != y.size() || x.empty()) return 0.0;
    const size_t n = x.size();
    // Rank x and y
    const auto rank_vec = [&](std::span<const double> v) {
        std::vector<size_t> idx(n);
        std::iota(idx.begin(), idx.end(), 0u);
        std::sort(idx.begin(), idx.end(),
                  [&](size_t a, size_t b) { return v[a] < v[b]; });
        std::vector<double> ranks(n);
        for (size_t i = 0; i < n; ++i) ranks[idx[i]] = static_cast<double>(i + 1);
        return ranks;
    };
    auto rx = rank_vec(x);
    auto ry = rank_vec(y);
    double d2 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = rx[i] - ry[i];
        d2 += d * d;
    }
    double nd = static_cast<double>(n);
    return 1.0 - 6.0 * d2 / (nd * (nd * nd - 1.0));
}

double kendall(std::span<const double> x, std::span<const double> y) {
    if (x.size() != y.size() || x.empty()) return 0.0;
    const size_t n = x.size();
    long long concordant = 0, discordant = 0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            double sx = x[i] - x[j];
            double sy = y[i] - y[j];
            if (sx * sy > 0.0) ++concordant;
            else if (sx * sy < 0.0) ++discordant;
        }
    }
    double nd = static_cast<double>(n);
    double denom = nd * (nd - 1.0) / 2.0;
    return (denom > 0.0) ? static_cast<double>(concordant - discordant) / denom : 0.0;
}

double chi2_gof(std::span<const double> observed, std::span<const double> expected) {
    if (observed.size() != expected.size() || observed.empty()) return 0.0;
    double chi2 = 0.0;
    for (size_t i = 0; i < observed.size(); ++i) {
        if (expected[i] > 0.0) {
            double d = observed[i] - expected[i];
            chi2 += d * d / expected[i];
        }
    }
    return chi2;
}

double ks_test(std::span<const double> x,
               std::function<double(double)> cdf) {
    if (x.empty()) return 0.0;
    std::vector<double> sorted(x.begin(), x.end());
    std::sort(sorted.begin(), sorted.end());
    const size_t n = sorted.size();
    double dn = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double f0 = cdf(sorted[i]);
        double ecdf_hi = static_cast<double>(i + 1) / static_cast<double>(n);
        double ecdf_lo = static_cast<double>(i) / static_cast<double>(n);
        dn = std::max(dn, std::abs(ecdf_hi - f0));
        dn = std::max(dn, std::abs(f0 - ecdf_lo));
    }
    return dn;
}

AnovaResult one_way_anova(const std::vector<std::vector<double>>& groups) {
    AnovaResult result{};
    if (groups.size() < 2) {
        return result;
    }
    int k = 0;
    int N = 0;
    double grand_sum = 0.0;
    for (const auto& group : groups) {
        if (group.empty()) {
            continue;
        }
        ++k;
        N += static_cast<int>(group.size());
        grand_sum += std::accumulate(group.begin(), group.end(), 0.0);
    }
    if (k < 2 || N <= k) {
        return result;
    }
    const double grand_mean = grand_sum / static_cast<double>(N);
    double ss_between = 0.0;
    double ss_within = 0.0;
    for (const auto& group : groups) {
        if (group.empty()) {
            continue;
        }
        const double m = mean(group);
        const double n = static_cast<double>(group.size());
        ss_between += n * (m - grand_mean) * (m - grand_mean);
        for (double v : group) {
            const double d = v - m;
            ss_within += d * d;
        }
    }
    result.df_between = k - 1;
    result.df_within = N - k;
    if (ss_within <= 0.0 || result.df_between < 1 || result.df_within < 1) {
        return result;
    }
    result.f_stat = (ss_between / static_cast<double>(result.df_between)) /
                    (ss_within / static_cast<double>(result.df_within));
    result.p_value = f_distribution_upper_tail_p(
        result.f_stat, result.df_between, result.df_within);
    return result;
}

MannWhitneyResult mann_whitney_u(std::span<const double> a, std::span<const double> b) {
    MannWhitneyResult result{};
    if (a.empty() || b.empty()) {
        return result;
    }
    const int n1 = static_cast<int>(a.size());
    const int n2 = static_cast<int>(b.size());
    const int N = n1 + n2;
    std::vector<double> combined(static_cast<size_t>(N));
    std::vector<int> labels(static_cast<size_t>(N));
    for (int i = 0; i < n1; ++i) {
        combined[static_cast<size_t>(i)] = a[static_cast<size_t>(i)];
        labels[static_cast<size_t>(i)] = 0;
    }
    for (int i = 0; i < n2; ++i) {
        combined[static_cast<size_t>(n1 + i)] = b[static_cast<size_t>(i)];
        labels[static_cast<size_t>(n1 + i)] = 1;
    }
    const auto ranks = average_ranks(combined);
    double r1 = 0.0;
    for (int i = 0; i < n1; ++i) {
        r1 += ranks[static_cast<size_t>(i)];
    }
    const double u1 = r1 - static_cast<double>(n1 * (n1 + 1)) / 2.0;
    const double u2 = static_cast<double>(n1 * n2) - u1;
    result.u_stat = std::min(u1, u2);
    const double mean_u = static_cast<double>(n1 * n2) / 2.0;
    double tie_sum = 0.0;
    std::unordered_map<double, int> tie_counts;
    for (double v : combined) {
        ++tie_counts[v];
    }
    for (const auto& [value, count] : tie_counts) {
        (void)value;
        if (count > 1) {
            tie_sum += static_cast<double>(count * count * count - count);
        }
    }
    double var_u = static_cast<double>(n1 * n2) *
                   (static_cast<double>(N + 1) -
                    tie_sum / (static_cast<double>(N) * static_cast<double>(N - 1))) /
                   12.0;
    if (var_u <= 0.0) {
        return result;
    }
    const double sd_u = std::sqrt(var_u);
    const double cc = (result.u_stat > mean_u) ? -0.5 : 0.5;
    const double z = (result.u_stat - mean_u + cc) / sd_u;
    result.p_value = 2.0 * (1.0 - norm_cdf(std::abs(z), 0.0, 1.0));
    return result;
}

KruskalWallisResult kruskal_wallis(const std::vector<std::vector<double>>& groups) {
    KruskalWallisResult result{};
    if (groups.size() < 2) {
        return result;
    }
    int k = 0;
    int N = 0;
    for (const auto& group : groups) {
        if (group.empty()) {
            continue;
        }
        ++k;
        N += static_cast<int>(group.size());
    }
    if (k < 2 || N <= k) {
        return result;
    }
    std::vector<double> pooled(static_cast<size_t>(N));
    std::vector<int> labels(static_cast<size_t>(N));
    size_t idx = 0;
    int group_idx = 0;
    for (const auto& group : groups) {
        if (group.empty()) {
            continue;
        }
        for (double v : group) {
            pooled[idx] = v;
            labels[idx] = group_idx;
            ++idx;
        }
        ++group_idx;
    }
    const auto ranks = average_ranks(pooled);
    std::vector<double> rank_sums(static_cast<size_t>(k), 0.0);
    std::vector<int> group_sizes(static_cast<size_t>(k), 0);
    for (int i = 0; i < N; ++i) {
        const size_t gi = static_cast<size_t>(labels[static_cast<size_t>(i)]);
        rank_sums[gi] += ranks[static_cast<size_t>(i)];
        ++group_sizes[gi];
    }
    double sum_term = 0.0;
    for (int gi = 0; gi < k; ++gi) {
        const double ni = static_cast<double>(group_sizes[static_cast<size_t>(gi)]);
        const double Ri = rank_sums[static_cast<size_t>(gi)];
        sum_term += (Ri * Ri) / ni;
    }
    const double N_d = static_cast<double>(N);
    double h = (12.0 / (N_d * (N_d + 1.0))) * sum_term - 3.0 * (N_d + 1.0);
    double tie_cubed_sum = 0.0;
    std::unordered_map<double, int> tie_counts;
    for (double v : pooled) {
        ++tie_counts[v];
    }
    for (const auto& [value, count] : tie_counts) {
        (void)value;
        if (count > 1) {
            tie_cubed_sum += static_cast<double>(count * count * count - count);
        }
    }
    const double tie_denom = N_d * N_d * N_d - N_d;
    if (tie_cubed_sum > 0.0 && tie_denom > 0.0) {
        const double correction = 1.0 - tie_cubed_sum / tie_denom;
        if (correction <= 0.0) {
            if (h <= 0.0) {
                result.df = k - 1;
                result.h_stat = 0.0;
                result.p_value = 1.0;
            }
            return result;
        }
        h /= correction;
    }
    result.df = k - 1;
    if (result.df < 1) {
        return result;
    }
    if (h < 0.0) {
        h = 0.0;
    }
    result.h_stat = h;
    result.p_value = 1.0 - chi2_cdf(h, static_cast<double>(result.df));
    return result;
}

FriedmanResult friedman(const std::vector<std::vector<double>>& data) {
    FriedmanResult result{};
    const int n = static_cast<int>(data.size());
    if (n < 2) {
        return result;
    }
    const int k = static_cast<int>(data[0].size());
    if (k < 2) {
        return result;
    }
    for (const auto& row : data) {
        if (static_cast<int>(row.size()) != k) {
            return result;
        }
    }
    std::vector<double> rank_sums(static_cast<size_t>(k), 0.0);
    double tie_cubed_sum = 0.0;
    for (const auto& row : data) {
        const auto ranks = average_ranks(row);
        for (int j = 0; j < k; ++j) {
            rank_sums[static_cast<size_t>(j)] += ranks[static_cast<size_t>(j)];
        }
        std::unordered_map<double, int> tie_counts;
        for (double v : row) {
            ++tie_counts[v];
        }
        for (const auto& [value, count] : tie_counts) {
            (void)value;
            if (count > 1) {
                tie_cubed_sum += static_cast<double>(count * count * count - count);
            }
        }
    }
    double sum_sq = 0.0;
    for (int j = 0; j < k; ++j) {
        const double Rj = rank_sums[static_cast<size_t>(j)];
        sum_sq += Rj * Rj;
    }
    const double n_d = static_cast<double>(n);
    const double k_d = static_cast<double>(k);
    double chi2 = (12.0 / (n_d * k_d * (k_d + 1.0))) * sum_sq - 3.0 * n_d * (k_d + 1.0);
    const double tie_denom = n_d * k_d * (k_d * k_d * k_d - k_d);
    if (tie_cubed_sum > 0.0 && tie_denom > 0.0) {
        const double correction = 1.0 - tie_cubed_sum / tie_denom;
        if (correction <= 0.0) {
            if (chi2 <= 0.0) {
                result.df = k - 1;
                result.chi2_stat = 0.0;
                result.p_value = 1.0;
            }
            return result;
        }
        chi2 /= correction;
    }
    result.df = k - 1;
    if (result.df < 1) {
        return result;
    }
    if (chi2 < 0.0) {
        chi2 = 0.0;
    }
    result.chi2_stat = chi2;
    result.p_value = 1.0 - chi2_cdf(chi2, static_cast<double>(result.df));
    return result;
}

KSTestResult ks_test_2sample(std::span<const double> a, std::span<const double> b) {
    KSTestResult result{};
    if (a.empty() || b.empty()) {
        return result;
    }
    const int n1 = static_cast<int>(a.size());
    const int n2 = static_cast<int>(b.size());
    std::vector<double> x1(a.begin(), a.end());
    std::vector<double> x2(b.begin(), b.end());
    std::sort(x1.begin(), x1.end());
    std::sort(x2.begin(), x2.end());
    size_t i = 0;
    size_t j = 0;
    double d_max = 0.0;
    while (i < x1.size() || j < x2.size()) {
        const double f1 = static_cast<double>(i) / static_cast<double>(n1);
        const double f2 = static_cast<double>(j) / static_cast<double>(n2);
        d_max = std::max(d_max, std::abs(f1 - f2));
        if (i < x1.size() && j < x2.size() && x1[i] == x2[j]) {
            ++i;
            ++j;
        } else if (i < x1.size() && (j >= x2.size() || x1[i] < x2[j])) {
            ++i;
        } else if (j < x2.size()) {
            ++j;
        } else {
            break;
        }
    }
    result.d_stat = d_max;
    if (d_max <= 0.0) {
        result.p_value = 1.0;
        return result;
    }
    const double n_e = static_cast<double>(n1 * n2) / static_cast<double>(n1 + n2);
    double p_sum = 0.0;
    for (int k = 1; k <= 100; ++k) {
        const double term =
            2.0 * std::exp(-2.0 * static_cast<double>(k * k) * n_e * d_max * d_max);
        p_sum += (k % 2 == 1) ? term : -term;
        if (term < 1e-14) {
            break;
        }
    }
    result.p_value = std::clamp(p_sum, 0.0, 1.0);
    return result;
}

JarqueBeraResult jarque_bera(std::span<const double> x) {
    JarqueBeraResult result{};
    const size_t n = x.size();
    if (n < 4) {
        return result;
    }
    const double s = skewness(x);
    const double k_excess = kurtosis(x);
    const double jb = (static_cast<double>(n) / 6.0) *
                      (s * s + (k_excess * k_excess) / 4.0);
    result.jb_stat = jb;
    result.df = 2;
    result.p_value = 1.0 - chi2_cdf(jb, 2.0);
    return result;
}

LjungBoxResult ljung_box(std::span<const double> x, int max_lag) {
    LjungBoxResult result{};
    const size_t n = x.size();
    if (n < 2 || max_lag < 1) {
        return result;
    }
    const int effective_max_lag =
        std::min(max_lag, static_cast<int>(n) - 1);
    if (effective_max_lag < 1) {
        return result;
    }
    const auto rhos = acf(x, effective_max_lag);
    const double n_d = static_cast<double>(n);
    double sum = 0.0;
    for (int k = 1; k <= effective_max_lag; ++k) {
        const double rho = rhos[static_cast<size_t>(k)];
        sum += (rho * rho) / (n_d - static_cast<double>(k));
    }
    const double q = n_d * (n_d + 2.0) * sum;
    result.q_stat = q;
    result.df = effective_max_lag;
    result.p_value =
        1.0 - chi2_cdf(q, static_cast<double>(effective_max_lag));
    return result;
}

LeveneResult levene_test(const std::vector<std::vector<double>>& groups) {
    LeveneResult result{};
    std::vector<std::vector<double>> transformed;
    transformed.reserve(groups.size());
    for (const auto& group : groups) {
        if (group.empty()) {
            transformed.emplace_back();
            continue;
        }
        const double med = median(group);
        std::vector<double> devs;
        devs.reserve(group.size());
        for (double v : group) {
            devs.push_back(std::abs(v - med));
        }
        transformed.push_back(std::move(devs));
    }
    const AnovaResult anova = one_way_anova(transformed);
    result.f_stat = anova.f_stat;
    result.p_value = anova.p_value;
    result.df_between = anova.df_between;
    result.df_within = anova.df_within;
    return result;
}

BartlettResult bartlett_test(const std::vector<std::vector<double>>& groups) {
    BartlettResult result{};
    int k = 0;
    int N = 0;
    for (const auto& group : groups) {
        if (group.empty()) {
            continue;
        }
        ++k;
        N += static_cast<int>(group.size());
    }
    if (k < 2 || N <= k) {
        return result;
    }
    double sum_ni_minus_1_ln_si2 = 0.0;
    double sum_inv_ni_minus_1 = 0.0;
    double sum_ni_minus_1_si2 = 0.0;
    for (const auto& group : groups) {
        if (group.empty()) {
            continue;
        }
        const int ni = static_cast<int>(group.size());
        if (ni < 2) {
            return result;
        }
        const double si2 = var(group);
        if (si2 <= 0.0) {
            return result;
        }
        const double ni_m1 = static_cast<double>(ni - 1);
        sum_ni_minus_1_ln_si2 += ni_m1 * std::log(si2);
        sum_inv_ni_minus_1 += 1.0 / ni_m1;
        sum_ni_minus_1_si2 += ni_m1 * si2;
    }
    const double sp2 = sum_ni_minus_1_si2 / static_cast<double>(N - k);
    if (sp2 <= 0.0) {
        return result;
    }
    const double M = static_cast<double>(N - k) * std::log(sp2) -
                     sum_ni_minus_1_ln_si2;
    const double C = 1.0 +
                     (1.0 / (3.0 * static_cast<double>(k - 1))) *
                         (sum_inv_ni_minus_1 - 1.0 / static_cast<double>(N - k));
    if (C <= 0.0) {
        return result;
    }
    double chi2 = M / C;
    if (chi2 < 0.0) {
        chi2 = 0.0;
    }
    result.chi2_stat = chi2;
    result.df = k - 1;
    result.p_value =
        1.0 - chi2_cdf(chi2, static_cast<double>(result.df));
    return result;
}

FlignerResult fligner_test(const std::vector<std::vector<double>>& groups) {
    FlignerResult result{};
    int k = 0;
    int N = 0;
    for (const auto& group : groups) {
        if (group.empty()) {
            continue;
        }
        ++k;
        N += static_cast<int>(group.size());
    }
    if (k < 2 || N <= k) {
        return result;
    }
    // Absolute deviations from each group's own median, pooled across groups but tracking
    // which (non-empty) group each deviation came from and each group's size.
    std::vector<double> pooled_devs;
    pooled_devs.reserve(static_cast<size_t>(N));
    std::vector<size_t> owner_group;
    owner_group.reserve(static_cast<size_t>(N));
    std::vector<int> group_sizes;
    group_sizes.reserve(static_cast<size_t>(k));
    for (const auto& group : groups) {
        if (group.empty()) {
            continue;
        }
        const double med = median(group);
        group_sizes.push_back(static_cast<int>(group.size()));
        const size_t gidx = group_sizes.size() - 1;
        for (double v : group) {
            pooled_devs.push_back(std::abs(v - med));
            owner_group.push_back(gidx);
        }
    }
    const auto ranks = average_ranks(pooled_devs);
    const double N_d = static_cast<double>(N);
    std::vector<double> scores(pooled_devs.size());
    for (size_t i = 0; i < pooled_devs.size(); ++i) {
        const double p = 0.5 + ranks[i] / (2.0 * (N_d + 1.0));
        scores[i] = norm_ppf(p, 0.0, 1.0);
    }
    const double a_bar = mean(scores);
    const double v = var(scores);
    if (v <= 0.0) {
        return result;
    }
    std::vector<double> group_score_sum(static_cast<size_t>(k), 0.0);
    for (size_t i = 0; i < scores.size(); ++i) {
        group_score_sum[owner_group[i]] += scores[i];
    }
    double numerator = 0.0;
    for (size_t g = 0; g < group_sizes.size(); ++g) {
        const double ni = static_cast<double>(group_sizes[g]);
        const double a_bar_i = group_score_sum[g] / ni;
        numerator += ni * (a_bar_i - a_bar) * (a_bar_i - a_bar);
    }
    double chi2_fk = numerator / v;
    if (chi2_fk < 0.0) {
        chi2_fk = 0.0;
    }
    result.chi2_stat = chi2_fk;
    result.df = k - 1;
    result.p_value = 1.0 - chi2_cdf(chi2_fk, static_cast<double>(result.df));
    return result;
}

ShapiroWilkResult shapiro_wilk(std::span<const double> x) {
    ShapiroWilkResult result{};
    const size_t n = x.size();
    if (n < 3) {
        return result;
    }
    std::vector<double> sorted(x.begin(), x.end());
    std::sort(sorted.begin(), sorted.end());

    const double n_d = static_cast<double>(n);
    std::vector<double> m(n);
    double sum_m2 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double p = (static_cast<double>(i + 1) - 0.375) / (n_d + 0.25);
        m[i] = norm_ppf(p, 0.0, 1.0);
        sum_m2 += m[i] * m[i];
    }
    if (sum_m2 <= 0.0) {
        return result;
    }
    const double norm_c = std::sqrt(sum_m2);

    const double xbar = mean(sorted);
    double denom = 0.0;
    for (double v : sorted) {
        const double d = v - xbar;
        denom += d * d;
    }
    if (denom <= 1e-14) {
        // Constant input: degenerate but not an error -- treat as trivially "normal".
        return result;
    }

    double numer = 0.0;
    for (size_t i = 0; i < n; ++i) {
        numer += (m[i] / norm_c) * sorted[i];
    }
    double w = (numer * numer) / denom;
    w = std::clamp(w, 1e-12, 1.0);
    result.w_stat = w;

    // Royston (1995) normalizing transformation of W (see the doc comment in stats.hpp for the
    // simplifications relative to the textbook algorithm).
    double w_trans;
    double mu;
    double sigma;
    if (n <= 11) {
        const double gamma = -2.273 + 0.459 * n_d;
        w_trans = -std::log(std::max(gamma - std::log(1.0 - w), 1e-300));
        mu = 0.5440 - 0.39978 * n_d + 0.025054 * n_d * n_d -
             0.0006714 * n_d * n_d * n_d;
        sigma = std::exp(1.3822 - 0.77857 * n_d + 0.062767 * n_d * n_d -
                         0.0020322 * n_d * n_d * n_d);
    } else {
        const double ln_n = std::log(n_d);
        w_trans = std::log(1.0 - w);
        mu = -1.5861 - 0.31082 * ln_n - 0.083751 * ln_n * ln_n +
             0.0038915 * ln_n * ln_n * ln_n;
        sigma = std::exp(-0.4803 - 0.082676 * ln_n + 0.0030302 * ln_n * ln_n);
    }
    if (sigma <= 0.0) {
        return result;
    }
    result.p_value = std::clamp(1.0 - norm_cdf(w_trans, mu, sigma), 0.0, 1.0);
    return result;
}

WilcoxonSignedRankResult wilcoxon_signed_rank(std::span<const double> x, std::span<const double> y) {
    WilcoxonSignedRankResult result{};
    if (x.size() != y.size() || x.empty()) {
        return result;
    }
    std::vector<double> abs_diffs;
    std::vector<double> signs;
    abs_diffs.reserve(x.size());
    signs.reserve(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        const double d = x[i] - y[i];
        if (d == 0.0) {
            continue;
        }
        abs_diffs.push_back(std::abs(d));
        signs.push_back(d > 0.0 ? 1.0 : -1.0);
    }
    const int n_eff = static_cast<int>(abs_diffs.size());
    if (n_eff < 1) {
        return result;
    }
    result.n_eff = n_eff;

    const auto ranks = average_ranks(abs_diffs);
    double w_plus = 0.0;
    double w_minus = 0.0;
    for (size_t i = 0; i < ranks.size(); ++i) {
        if (signs[i] > 0.0) {
            w_plus += ranks[i];
        } else {
            w_minus += ranks[i];
        }
    }
    result.w_stat = std::min(w_plus, w_minus);

    const double n_d = static_cast<double>(n_eff);
    const double mean_w = n_d * (n_d + 1.0) / 4.0;
    double tie_sum = 0.0;
    std::unordered_map<double, int> tie_counts;
    for (double v : abs_diffs) {
        ++tie_counts[v];
    }
    for (const auto& [value, count] : tie_counts) {
        (void)value;
        if (count > 1) {
            tie_sum += static_cast<double>(count * count * count - count);
        }
    }
    double var_w = n_d * (n_d + 1.0) * (2.0 * n_d + 1.0) / 24.0 - tie_sum / 48.0;
    if (var_w <= 0.0) {
        return result;
    }
    const double sd_w = std::sqrt(var_w);
    const double cc = (result.w_stat > mean_w) ? -0.5 : 0.5;
    result.z_stat = (result.w_stat - mean_w + cc) / sd_w;
    result.p_value = 2.0 * (1.0 - norm_cdf(std::abs(result.z_stat), 0.0, 1.0));
    return result;
}

std::vector<double> multiple_regression(
    const std::vector<std::vector<double>>& X,
    std::span<const double> y) {
    if (X.empty() || X[0].empty()) return {};
    const size_t m = X.size();
    const size_t p = X[0].size();
    // Normal equations: (X^T X) beta = X^T y
    std::vector<std::vector<double>> XtX(p, std::vector<double>(p, 0.0));
    std::vector<double> Xty(p, 0.0);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < p; ++j) {
            Xty[j] += X[i][j] * y[i];
            for (size_t k = 0; k < p; ++k)
                XtX[j][k] += X[i][j] * X[i][k];
        }
    }
    // Gaussian elimination
    auto A = XtX;
    auto b = Xty;
    for (size_t col = 0; col < p; ++col) {
        size_t pivot = col;
        for (size_t row = col + 1; row < p; ++row)
            if (std::abs(A[row][col]) > std::abs(A[pivot][col])) pivot = row;
        std::swap(A[col], A[pivot]);
        std::swap(b[col], b[pivot]);
        if (std::abs(A[col][col]) < 1e-14) continue;
        for (size_t row = col + 1; row < p; ++row) {
            double f = A[row][col] / A[col][col];
            for (size_t k = col; k < p; ++k) A[row][k] -= f * A[col][k];
            b[row] -= f * b[col];
        }
    }
    std::vector<double> beta(p, 0.0);
    for (int i = static_cast<int>(p) - 1; i >= 0; --i) {
        double s = b[static_cast<size_t>(i)];
        for (size_t j = static_cast<size_t>(i) + 1; j < p; ++j)
            s -= A[static_cast<size_t>(i)][j] * beta[j];
        beta[static_cast<size_t>(i)] =
            (std::abs(A[static_cast<size_t>(i)][static_cast<size_t>(i)]) > 1e-14)
                ? s / A[static_cast<size_t>(i)][static_cast<size_t>(i)]
                : 0.0;
    }
    return beta;
}

double variance_inflation_factor(const std::vector<std::vector<double>>& X, size_t j) {
    if (X.empty() || X[0].empty()) {
        return 1.0;
    }
    const size_t m = X.size();
    const size_t p = X[0].size();
    if (j >= p) {
        return 0.0;
    }
    for (const auto& row : X) {
        if (row.size() != p) {
            return 0.0;
        }
    }
    if (p < 2) {
        return 1.0;
    }

    std::vector<double> y(m);
    for (size_t i = 0; i < m; ++i) {
        y[i] = X[i][j];
    }

    std::vector<std::vector<double>> X_other(
        m, std::vector<double>(p - 1, 0.0));
    for (size_t i = 0; i < m; ++i) {
        size_t col = 0;
        for (size_t k = 0; k < p; ++k) {
            if (k == j) {
                continue;
            }
            X_other[i][col++] = X[i][k];
        }
    }

    const auto beta = multiple_regression(X_other, y);
    if (beta.size() != p - 1) {
        return 1.0;
    }

    const double my = mean(y);
    double ss_tot = 0.0;
    double ss_res = 0.0;
    for (size_t i = 0; i < m; ++i) {
        double y_hat = 0.0;
        for (size_t k = 0; k < beta.size(); ++k) {
            y_hat += beta[k] * X_other[i][k];
        }
        const double centered = y[i] - my;
        ss_tot += centered * centered;
        const double residual = y[i] - y_hat;
        ss_res += residual * residual;
    }
    if (ss_tot == 0.0) {
        return 1.0;
    }

    const double r_squared = 1.0 - ss_res / ss_tot;
    if (r_squared >= 1.0 - 1e-14) {
        return std::numeric_limits<double>::infinity();
    }
    return 1.0 / (1.0 - r_squared);
}

double vif(const std::vector<std::vector<double>>& X, size_t j) {
    return variance_inflation_factor(X, j);
}

namespace {

double kde_kernel(double u, const char* kernel) {
    if (std::strcmp(kernel, "epanechnikov") == 0) {
        if (std::abs(u) > 1.0) {
            return 0.0;
        }
        return 0.75 * (1.0 - u * u);
    }
    static constexpr double inv_sqrt_2pi = 0.3989422804014327;
    return inv_sqrt_2pi * std::exp(-0.5 * u * u);
}

bool kde_is_epanechnikov(const char* kernel) {
    return std::strcmp(kernel, "epanechnikov") == 0;
}

// Gaussian tail cutoff in sigma units: at least 4σ for large n·h, widened when
// needed so omitted per-sample mass stays below 1e-13 (safe for 1e-10 tests).
double kde_gaussian_support_sigmas(size_t n, double bandwidth) {
    static constexpr double inv_sqrt_2pi = 0.3989422804014327;
    static constexpr double k_min_sigmas = 4.0;
    static constexpr double k_omit_threshold = 1e-13;

    const double nh = static_cast<double>(n) * bandwidth;
    if (nh <= 0.0) {
        return k_min_sigmas;
    }
    const double log_thresh =
        std::log(k_omit_threshold * nh / inv_sqrt_2pi);
    if (log_thresh >= 0.0) {
        return 100.0;
    }
    const double needed = std::sqrt(-2.0 * log_thresh);
    return std::max(k_min_sigmas, needed);
}

double kde_support_radius(const char* kernel, size_t n, double bandwidth) {
    if (kde_is_epanechnikov(kernel)) {
        return bandwidth;
    }
    return kde_gaussian_support_sigmas(n, bandwidth) * bandwidth;
}

} // namespace

std::vector<double> kde(std::span<const double> samples,
                        std::span<const double> grid,
                        double bandwidth,
                        const char* kernel) {
    if (samples.empty() || bandwidth <= 0.0) {
        return {};
    }
    if (grid.empty()) {
        return {};
    }

    std::vector<double> sorted(samples.begin(), samples.end());
    std::sort(sorted.begin(), sorted.end());

    const double inv_nh =
        1.0 / (static_cast<double>(samples.size()) * bandwidth);
    const double support = kde_support_radius(kernel, samples.size(), bandwidth);

    std::vector<double> result(grid.size(), 0.0);
    for (size_t gi = 0; gi < grid.size(); ++gi) {
        const double x = grid[gi];
        const double lo_val = x - support;
        const double hi_val = x + support;

        const auto lo_it =
            std::lower_bound(sorted.begin(), sorted.end(), lo_val);
        const auto hi_it =
            std::upper_bound(sorted.begin(), sorted.end(), hi_val);

        double sum = 0.0;
        for (auto it = lo_it; it != hi_it; ++it) {
            sum += kde_kernel((x - *it) / bandwidth, kernel);
        }
        result[gi] = inv_nh * sum;
    }
    return result;
}

std::vector<double> acf(std::span<const double> x, int max_lag) {
    if (x.empty() || max_lag < 0) return {};
    const double m = mean(x);
    const size_t n = x.size();
    double var0 = 0.0;
    for (double v : x) var0 += (v - m) * (v - m);
    std::vector<double> result(static_cast<size_t>(max_lag + 1), 0.0);
    for (int lag = 0; lag <= max_lag; ++lag) {
        double sum = 0.0;
        for (size_t i = 0; i + static_cast<size_t>(lag) < n; ++i)
            sum += (x[i] - m) * (x[i + static_cast<size_t>(lag)] - m);
        result[static_cast<size_t>(lag)] = (var0 > 0.0) ? sum / var0 : 0.0;
    }
    return result;
}

std::vector<double> pacf(std::span<const double> x, int max_lag) {
    if (x.empty() || max_lag < 1) return {};
    // Yule-Walker equations
    auto ac = acf(x, max_lag);
    std::vector<double> result(static_cast<size_t>(max_lag + 1), 0.0);
    result[0] = 1.0;

    std::vector<std::vector<double>> phi(
        static_cast<size_t>(max_lag + 1),
        std::vector<double>(static_cast<size_t>(max_lag + 1), 0.0));

    for (int k = 1; k <= max_lag; ++k) {
        double num = ac[static_cast<size_t>(k)];
        for (int j = 1; j < k; ++j)
            num -= phi[static_cast<size_t>(k - 1)][static_cast<size_t>(j)] *
                   ac[static_cast<size_t>(k - j)];
        double den = 1.0;
        for (int j = 1; j < k; ++j)
            den -= phi[static_cast<size_t>(k - 1)][static_cast<size_t>(j)] *
                   ac[static_cast<size_t>(j)];
        phi[static_cast<size_t>(k)][static_cast<size_t>(k)] =
            (std::abs(den) > 1e-14) ? num / den : 0.0;
        result[static_cast<size_t>(k)] =
            phi[static_cast<size_t>(k)][static_cast<size_t>(k)];
        for (int j = 1; j < k; ++j)
            phi[static_cast<size_t>(k)][static_cast<size_t>(j)] =
                phi[static_cast<size_t>(k - 1)][static_cast<size_t>(j)] -
                phi[static_cast<size_t>(k)][static_cast<size_t>(k)] *
                phi[static_cast<size_t>(k - 1)][static_cast<size_t>(k - j)];
    }
    return result;
}

std::vector<double> arfit(std::span<const double> x, int p) {
    if (static_cast<int>(x.size()) <= p || p < 1) return {};
    // Yule-Walker method
    auto ac = acf(x, p);
    // Solve Toeplitz system R * phi = r
    std::vector<std::vector<double>> R(
        static_cast<size_t>(p),
        std::vector<double>(static_cast<size_t>(p), 0.0));
    std::vector<double> r(static_cast<size_t>(p));
    for (int i = 0; i < p; ++i) {
        r[static_cast<size_t>(i)] = ac[static_cast<size_t>(i + 1)];
        for (int j = 0; j < p; ++j)
            R[static_cast<size_t>(i)][static_cast<size_t>(j)] =
                ac[static_cast<size_t>(std::abs(i - j))];
    }
    // Gaussian elimination
    for (int col = 0; col < p; ++col) {
        int pivot = col;
        for (int row = col + 1; row < p; ++row)
            if (std::abs(R[static_cast<size_t>(row)][static_cast<size_t>(col)]) >
                std::abs(R[static_cast<size_t>(pivot)][static_cast<size_t>(col)]))
                pivot = row;
        std::swap(R[static_cast<size_t>(col)], R[static_cast<size_t>(pivot)]);
        std::swap(r[static_cast<size_t>(col)], r[static_cast<size_t>(pivot)]);
        if (std::abs(R[static_cast<size_t>(col)][static_cast<size_t>(col)]) < 1e-14)
            continue;
        for (int row = col + 1; row < p; ++row) {
            double f = R[static_cast<size_t>(row)][static_cast<size_t>(col)] /
                       R[static_cast<size_t>(col)][static_cast<size_t>(col)];
            for (int k = col; k < p; ++k)
                R[static_cast<size_t>(row)][static_cast<size_t>(k)] -=
                    f * R[static_cast<size_t>(col)][static_cast<size_t>(k)];
            r[static_cast<size_t>(row)] -= f * r[static_cast<size_t>(col)];
        }
    }
    std::vector<double> phi(static_cast<size_t>(p), 0.0);
    for (int i = p - 1; i >= 0; --i) {
        double s = r[static_cast<size_t>(i)];
        for (int j = i + 1; j < p; ++j)
            s -= R[static_cast<size_t>(i)][static_cast<size_t>(j)] *
                 phi[static_cast<size_t>(j)];
        double diag = R[static_cast<size_t>(i)][static_cast<size_t>(i)];
        phi[static_cast<size_t>(i)] = (std::abs(diag) > 1e-14) ? s / diag : 0.0;
    }
    return phi;
}

double bootstrap_mean(std::span<const double> data, int n_boot, unsigned seed) {
    if (data.empty()) return 0.0;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<size_t> idx_dist(0, data.size() - 1);
    double sum = 0.0;
    for (int b = 0; b < n_boot; ++b) {
        double s = 0.0;
        for (size_t i = 0; i < data.size(); ++i)
            s += data[idx_dist(rng)];
        sum += s / static_cast<double>(data.size());
    }
    return sum / static_cast<double>(n_boot);
}

std::pair<double, double> bootstrap_ci(std::span<const double> data,
                                        double level, int n_boot, unsigned seed) {
    if (data.empty()) return {0.0, 0.0};
    std::mt19937 rng(seed);
    std::uniform_int_distribution<size_t> idx_dist(0, data.size() - 1);
    std::vector<double> boot_means(static_cast<size_t>(n_boot));
    for (int b = 0; b < n_boot; ++b) {
        double s = 0.0;
        for (size_t i = 0; i < data.size(); ++i)
            s += data[idx_dist(rng)];
        boot_means[static_cast<size_t>(b)] = s / static_cast<double>(data.size());
    }
    std::sort(boot_means.begin(), boot_means.end());
    double alpha = (1.0 - level) / 2.0;
    size_t lo = static_cast<size_t>(alpha * static_cast<double>(n_boot));
    size_t hi = static_cast<size_t>((1.0 - alpha) * static_cast<double>(n_boot));
    hi = std::min(hi, boot_means.size() - 1);
    return {boot_means[lo], boot_means[hi]};
}

BootstrapResult bootstrap_ci(
    std::span<const double> data,
    const std::function<double(std::span<const double>)>& stat_fn,
    size_t n_resamples,
    double confidence_level,
    unsigned seed) {
    BootstrapResult result{};
    if (data.empty() || n_resamples == 0) {
        return result;
    }
    result.point_estimate = stat_fn(data);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<size_t> idx_dist(0, data.size() - 1);
    std::vector<double> boot_stats(n_resamples);
    std::vector<double> sample(data.size());
    for (size_t b = 0; b < n_resamples; ++b) {
        for (size_t i = 0; i < data.size(); ++i) {
            sample[i] = data[idx_dist(rng)];
        }
        boot_stats[b] = stat_fn(sample);
    }
    std::sort(boot_stats.begin(), boot_stats.end());
    const double alpha = (1.0 - confidence_level) / 2.0;
    size_t lo = static_cast<size_t>(alpha * static_cast<double>(n_resamples));
    size_t hi = static_cast<size_t>((1.0 - alpha) * static_cast<double>(n_resamples));
    hi = std::min(hi, boot_stats.size() - 1);
    result.lower = boot_stats[lo];
    result.upper = boot_stats[hi];
    if (n_resamples >= 2) {
        const double m = std::accumulate(boot_stats.begin(), boot_stats.end(), 0.0) /
                         static_cast<double>(n_resamples);
        double sum_sq = 0.0;
        for (double v : boot_stats) {
            const double d = v - m;
            sum_sq += d * d;
        }
        result.std_error =
            std::sqrt(sum_sq / static_cast<double>(n_resamples - 1));
    }
    return result;
}

} // namespace ms
