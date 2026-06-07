#include "ms/stats/stats.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
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

} // namespace ms
