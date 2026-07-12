#include "ms/frameworks/cypha/cypha.hpp"

#include "ms/frameworks/izaac/izaac.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ms::cypha {

namespace {

double normal_quantile(double p) {
    if (p <= 0.0) {
        return -8.0;
    }
    if (p >= 1.0) {
        return 8.0;
    }

    static constexpr double a1 = -3.969683028665376e+01;
    static constexpr double a2 = 2.209460984245205e+02;
    static constexpr double a3 = -2.759401190516105e+02;
    static constexpr double a4 = 1.383577518672690e+02;
    static constexpr double a5 = -3.066479806614716e+01;
    static constexpr double a6 = 2.506628277459239e+00;

    static constexpr double b1 = -5.447609879822406e+01;
    static constexpr double b2 = 1.615858368580409e+02;
    static constexpr double b3 = -1.556989798598866e+02;
    static constexpr double b4 = 6.680131188771972e+01;
    static constexpr double b5 = -1.328068155288572e+01;

    static constexpr double c1 = -7.784894002430293e-03;
    static constexpr double c2 = -3.223964580411365e-01;
    static constexpr double c3 = -2.400758227161838e+00;
    static constexpr double c4 = -2.549732539343734e+00;
    static constexpr double c5 = 4.374664141464968e+00;
    static constexpr double c6 = 2.938163982698783e+00;

    static constexpr double d1 = 7.784695709091636e-03;
    static constexpr double d2 = 3.224671290700398e-01;
    static constexpr double d3 = 2.445134137142996e+00;
    static constexpr double d4 = 3.754408661907416e+00;

    static constexpr double p_low = 0.02425;
    static constexpr double p_high = 1.0 - p_low;

    if (p < p_low) {
        const double q = std::sqrt(-2.0 * std::log(p));
        return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
               ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
    }
    if (p <= p_high) {
        const double q = p - 0.5;
        const double r = q * q;
        return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
               (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0);
    }
    const double q = std::sqrt(-2.0 * std::log(1.0 - p));
    return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
           ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
}

} // namespace

DifModel::DifModel(const DifConfig& cfg) : cfg_(cfg) {
    weights_ = Matrix<double>(cfg.output_dim, cfg.input_dim, 0.0);
    bias_ = Matrix<double>(cfg.output_dim, 1, 0.0);
    uncertainty_ = {0.0, 2.0, 0.0, 1.0};
}

Result<void> DifModel::update(const Matrix<double>& x, const Matrix<double>& y) {
    if (x.rows() != cfg_.input_dim || y.rows() != cfg_.output_dim || x.cols() != 1 || y.cols() != 1) {
        return std::unexpected(DimensionMismatch{x.rows(), y.rows()});
    }

    Matrix<double> prediction(cfg_.output_dim, 1, 0.0);
    for (size_t o = 0; o < cfg_.output_dim; ++o) {
        double sum = bias_(o, 0);
        for (size_t i = 0; i < cfg_.input_dim; ++i) {
            sum += weights_(o, i) * x(i, 0);
        }
        prediction(o, 0) = sum;
        const double err = y(o, 0) - sum;
        bias_(o, 0) += cfg_.learning_rate * err;
        for (size_t i = 0; i < cfg_.input_dim; ++i) {
            weights_(o, i) += cfg_.learning_rate * err * x(i, 0);
        }
    }

    uncertainty_.mu = prediction(0, 0);
    uncertainty_.delta = std::max(0.1, uncertainty_.delta * (1.0 - cfg_.learning_rate * 0.1));
    ++updates_;
    return {};
}

Result<Matrix<double>> DifModel::predict(const Matrix<double>& x) const {
    if (x.rows() != cfg_.input_dim || x.cols() != 1) {
        return std::unexpected(DimensionMismatch{x.rows(), cfg_.input_dim});
    }
    Matrix<double> out(cfg_.output_dim, 1, 0.0);
    for (size_t o = 0; o < cfg_.output_dim; ++o) {
        double sum = bias_(o, 0);
        for (size_t i = 0; i < cfg_.input_dim; ++i) {
            sum += weights_(o, i) * x(i, 0);
        }
        out(o, 0) = sum;
    }
    return out;
}

Result<PredictionInterval> DifModel::predict_interval(const Matrix<double>& x) const {
    const auto mean_result = predict(x);
    if (!mean_result) {
        return std::unexpected(mean_result.error());
    }

    // Symmetric credible band from the fitted NIG uncertainty proxy.
    // scale = delta / sqrt(alpha) approximates an effective posterior std-dev;
    // z matches gh_protection coverage (not a full Bayesian GH posterior).
    const double alpha = std::max(uncertainty_.alpha, 1e-6);
    const double scale = uncertainty_.delta / std::sqrt(alpha);
    const double z = normal_quantile(0.5 + 0.5 * cfg_.gh_protection);

    PredictionInterval interval;
    interval.mean = mean_result.value();
    interval.lower = Matrix<double>(cfg_.output_dim, 1, 0.0);
    interval.upper = Matrix<double>(cfg_.output_dim, 1, 0.0);
    for (size_t o = 0; o < cfg_.output_dim; ++o) {
        const double m = interval.mean(o, 0);
        interval.lower(o, 0) = m - z * scale;
        interval.upper(o, 0) = m + z * scale;
    }
    interval.nig_alpha = uncertainty_.alpha;
    interval.nig_beta = uncertainty_.beta;
    return interval;
}

Result<double> DifModel::ood_score(const Matrix<double>& x) const {
    if (x.rows() != cfg_.input_dim || x.cols() != 1) {
        return std::unexpected(DimensionMismatch{x.rows(), cfg_.input_dim});
    }
    double energy = 0.0;
    for (size_t i = 0; i < x.rows(); ++i) {
        energy += x(i, 0) * x(i, 0);
    }
    return std::sqrt(energy) / static_cast<double>(cfg_.input_dim);
}

bool DifModel::gh_gate(const Matrix<double>& x) const {
    const auto score_result = ood_score(x);
    if (!score_result) {
        return false;
    }

    // Simplified GH-posterior gate: treat ood_score^2 * d as a chi-squared(df=d) proxy
    // under iid N(0,1) inputs. Full GH-posterior coverage math is out of scope here;
    // Wilson–Hilferty gives a defensible chi-squared quantile at gh_protection.
    const double score = score_result.value();
    const double d = static_cast<double>(cfg_.input_dim);
    const double chi_stat = score * score * d;
    const double z = normal_quantile(cfg_.gh_protection);
    const double wilson = 1.0 - 2.0 / (9.0 * d) + z * std::sqrt(2.0 / (9.0 * d));
    const double chi_thresh = d * wilson * wilson * wilson;
    return chi_stat <= chi_thresh;
}

NIGParams nig_fit(const Matrix<double>& data) {
    NIGParams params;
    if (data.rows() == 0) {
        return params;
    }
    double sum = 0.0;
    for (size_t i = 0; i < data.rows(); ++i) {
        sum += data(i, 0);
    }
    params.mu = sum / static_cast<double>(data.rows());
    double var = 0.0;
    for (size_t i = 0; i < data.rows(); ++i) {
        const double d = data(i, 0) - params.mu;
        var += d * d;
    }
    params.delta = std::sqrt(var / static_cast<double>(data.rows()) + 1e-6);
    params.alpha = 2.0;
    return params;
}

double nig_pdf(double x, const NIGParams& params) {
    const double z =
        params.alpha * params.delta / std::sqrt(params.delta * params.delta + (x - params.mu) * (x - params.mu));
    return std::exp(-z) / (2.0 * params.delta + 1e-12);
}

namespace {

double nig_cdf_integrate(double a, double b, const NIGParams& params, int steps = 512) {
    if (a >= b) {
        return 0.0;
    }
    const double h = (b - a) / static_cast<double>(steps);
    double sum = 0.5 * (nig_pdf(a, params) + nig_pdf(b, params));
    for (int i = 1; i < steps; ++i) {
        sum += nig_pdf(a + static_cast<double>(i) * h, params);
    }
    return sum * h;
}

} // namespace

double nig_cdf(double x, const NIGParams& params) {
    const double delta = std::max(params.delta, 1e-6);
    const double left = params.mu - 80.0 * delta;
    const double right = params.mu + 80.0 * delta;
    const double total = nig_cdf_integrate(left, right, params);
    if (total <= 0.0) {
        return 0.5;
    }
    const double partial = nig_cdf_integrate(left, x, params);
    return std::clamp(partial / total, 0.0, 1.0);
}

namespace {

// Both closed-form moments require alpha^2 - beta^2 > 0 (a proper NIG tail-heaviness
// vs. asymmetry balance) and delta > 0 (a proper scale). Outside this domain the
// distribution is undefined, so callers get NaN rather than an inf/garbage value.
bool nig_moments_domain_valid(const NIGParams& params) {
    return params.delta > 0.0 && std::abs(params.beta) < params.alpha;
}

} // namespace

double nig_mean(const NIGParams& params) {
    if (!nig_moments_domain_valid(params)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double gamma = std::sqrt(params.alpha * params.alpha - params.beta * params.beta);
    return params.mu + params.delta * params.beta / gamma;
}

double nig_variance(const NIGParams& params) {
    if (!nig_moments_domain_valid(params)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double gamma = std::sqrt(params.alpha * params.alpha - params.beta * params.beta);
    return params.delta * params.alpha * params.alpha / (gamma * gamma * gamma);
}

Matrix<double> nig_sample(const NIGParams& params, size_t n, izaac::CSPRNG& rng) {
    Matrix<double> samples(n, 1, 0.0);
    const double alpha = std::max(params.alpha, 1.0);
    const double base_scale = params.delta / std::sqrt(alpha);

    // Variance-mixture heuristic consistent with the simplified NIG parameterisation:
    // X = mu + beta * U + base_scale * sqrt(G) * Z, G ~ Exp(1), Z ~ N(0,1).
    for (size_t i = 0; i < n; ++i) {
        const double z = rng.next_normal();
        const double g = std::max(1e-6, -std::log(std::max(rng.next_f64(), 1e-12)));
        const double skew = params.beta * (rng.next_f64() - 0.5);
        samples(i, 0) = params.mu + skew + base_scale * std::sqrt(g) * z;
    }
    return samples;
}

} // namespace ms::cypha
