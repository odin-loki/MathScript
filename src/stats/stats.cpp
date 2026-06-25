#include "ms/stats/stats.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <random>
#include <unordered_map>
#include <vector>

namespace ms {

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

} // namespace ms
