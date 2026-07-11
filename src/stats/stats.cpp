#include "ms/stats/stats.hpp"
#include "ms/prob/prob.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <unordered_map>
#include <vector>

namespace ms {

namespace {

double beta_continued_fraction(double a, double b, double x) {
    const double fpmin = 1e-300;
    const double qab = a + b;
    const double qap = a + 1.0;
    const double qam = a - 1.0;
    double c = 1.0;
    double d = 1.0 - qab * x / qap;
    if (std::abs(d) < fpmin) {
        d = fpmin;
    }
    d = 1.0 / d;
    double h = d;
    for (int m = 1; m <= 200; ++m) {
        const double m2 = 2.0 * static_cast<double>(m);
        double aa = static_cast<double>(m) * (b - static_cast<double>(m)) * x /
                    ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if (std::abs(d) < fpmin) {
            d = fpmin;
        }
        c = 1.0 + aa / c;
        if (std::abs(c) < fpmin) {
            c = fpmin;
        }
        d = 1.0 / d;
        h *= d * c;
        aa = -(a + static_cast<double>(m)) * (qab + static_cast<double>(m)) * x /
             ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if (std::abs(d) < fpmin) {
            d = fpmin;
        }
        c = 1.0 + aa / c;
        if (std::abs(c) < fpmin) {
            c = fpmin;
        }
        d = 1.0 / d;
        const double del = d * c;
        h *= del;
        if (std::abs(del - 1.0) < 1e-14) {
            break;
        }
    }
    return h;
}

double regularized_incomplete_beta(double a, double b, double x) {
    if (x <= 0.0) {
        return 0.0;
    }
    if (x >= 1.0) {
        return 1.0;
    }
    const double bt = std::exp(std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) +
                               a * std::log(x) + b * std::log(1.0 - x));
    if (x < (a + 1.0) / (a + b + 2.0)) {
        return bt * beta_continued_fraction(a, b, x) / a;
    }
    return 1.0 - bt * beta_continued_fraction(b, a, 1.0 - x) / b;
}

double f_distribution_upper_tail_p(double f_stat, int df1, int df2) {
    if (f_stat <= 0.0 || df1 < 1 || df2 < 1) {
        return 1.0;
    }
    const double x = static_cast<double>(df1) * f_stat /
                     (static_cast<double>(df1) * f_stat + static_cast<double>(df2));
    return 1.0 - regularized_incomplete_beta(
                     static_cast<double>(df1) / 2.0,
                     static_cast<double>(df2) / 2.0,
                     x);
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
    std::vector<double> sorted(data.begin(), data.end());
    std::sort(sorted.begin(), sorted.end());
    const size_t idx = static_cast<size_t>((p / 100.0) * static_cast<double>(sorted.size() - 1));
    return sorted[idx];
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

double mad(std::span<const double> data) {
    if (data.empty()) return 0.0;
    double med = median(data);
    std::vector<double> abs_dev(data.size());
    for (size_t i = 0; i < data.size(); ++i)
        abs_dev[i] = std::abs(data[i] - med);
    return median(abs_dev);
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
