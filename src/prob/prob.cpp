#include "ms/prob/prob.hpp"
#include "ms/special/special.hpp"
#include <cmath>
#include <functional>
#include <limits>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ms {

namespace {

constexpr double k_ppf_neg_tail = -1e100;
constexpr double k_ppf_pos_tail = 1e100;
constexpr double k_ppf_bracket_max = 1e300;

double bisect_increasing(const std::function<double(double)>& f, double lo, double hi,
                         double tol = 1e-12, int max_iter = 200) {
    for (int i = 0; i < max_iter; ++i) {
        const double mid = 0.5 * (lo + hi);
        if (hi - lo <= tol) {
            break;
        }
        if (f(mid) <= 0.0) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return 0.5 * (lo + hi);
}

double expand_upper_until_cdf(const std::function<double(double)>& cdf, double p, double hi) {
    while (cdf(hi) < p && hi < k_ppf_bracket_max) {
        hi *= 2.0;
    }
    return hi;
}

} // namespace

double norm_pdf(double x, double mu, double sigma) {
    if (sigma == 0) {
        return 0.0;
    }
    const double z = (x - mu) / sigma;
    return std::exp(-0.5 * z * z) / (sigma * std::sqrt(2.0 * M_PI));
}

double norm_cdf(double x, double mu, double sigma) {
    if (sigma == 0) {
        return 0.0;
    }
    const double z = (x - mu) / sigma;
    return 0.5 * (1.0 + std::erf(z / std::sqrt(2.0)));
}

double norm_ppf(double p, double mu, double sigma) {
    if (p <= 0.0 || p >= 1.0) return 0.0;  // Guard: boundary/invalid returns 0
    if (p == 0.5) return mu;
    // Rational approximation (Beasley-Springer-Moro) for standard normal PPF
    // then scale by sigma and shift by mu
    static const double a[] = {-3.969683028665376e+01, 2.209460984245205e+02,
                                -2.759285104469687e+02, 1.383577518672690e+02,
                                -3.066479806614716e+01, 2.506628277459239e+00};
    static const double b[] = {-5.447609879822406e+01, 1.615858368580409e+02,
                                -1.556989798598866e+02, 6.680131188771972e+01,
                                -1.328068155288572e+01};
    static const double c[] = {-7.784894002430293e-03, -3.223964580411365e-01,
                                -2.400758277161838e+00, -2.549732539343734e+00,
                                 4.374664141464968e+00,  2.938163982698783e+00};
    static const double d[] = { 7.784695709041462e-03,  3.224671290700398e-01,
                                 2.445134137142996e+00,  3.754408661907416e+00};
    const double plow = 0.02425, phigh = 1.0 - plow;
    double q, r, z;
    if (p < plow) {
        q = std::sqrt(-2.0 * std::log(p));
        z = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    } else if (p <= phigh) {
        q = p - 0.5;
        r = q * q;
        z = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
            (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0);
    } else {
        q = std::sqrt(-2.0 * std::log(1.0 - p));
        z = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
             ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    }
    return mu + z * sigma;
}

double exp_pdf(double x, double lambda) {
    if (x < 0.0 || lambda <= 0.0) {
        return 0.0;
    }
    return lambda * std::exp(-lambda * x);
}

double exp_cdf(double x, double lambda) {
    if (x < 0.0 || lambda <= 0.0) {
        return 0.0;
    }
    return 1.0 - std::exp(-lambda * x);
}

double exp_ppf(double p, double lambda) {
    if (lambda <= 0.0) {
        return 0.0;
    }
    if (p <= 0.0) {
        return 0.0;
    }
    if (p >= 1.0) {
        return k_ppf_pos_tail;
    }
    return -std::log(1.0 - p) / lambda;
}

double binom_pdf(int k, int n, double p) {
    if (k < 0 || k > n || p < 0.0 || p > 1.0) {
        return 0.0;
    }
    if (k == 0) {
        return std::pow(1.0 - p, static_cast<double>(n));
    }
    if (k == n) {
        return std::pow(p, static_cast<double>(n));
    }
    const double log_coef = std::lgamma(static_cast<double>(n + 1))
        - std::lgamma(static_cast<double>(k + 1))
        - std::lgamma(static_cast<double>(n - k + 1));
    return std::exp(log_coef + static_cast<double>(k) * std::log(p)
                    + static_cast<double>(n - k) * std::log1p(-p));
}

double binom_cdf(int k, int n, double p) {
    if (k < 0 || p < 0.0 || p > 1.0 || n < 0) {
        return 0.0;
    }
    if (k >= n) {
        return 1.0;
    }
    double sum = std::pow(1.0 - p, static_cast<double>(n));
    if (k == 0) {
        return sum;
    }
    double term = sum;
    for (int i = 0; i < k; ++i) {
        term *= static_cast<double>(n - i) / static_cast<double>(i + 1) * p / (1.0 - p);
        sum += term;
    }
    return sum;
}

double pois_pdf(double k, double lambda) {
    if (k < 0.0 || lambda <= 0.0) {
        return 0.0;
    }
    return std::pow(lambda, k) * std::exp(-lambda) / std::tgamma(k + 1.0);
}

double pois_cdf(double k, double lambda) {
    if (lambda <= 0.0) {
        return 0.0;
    }
    if (k < 0.0) {
        return 0.0;
    }
    const int k_int = static_cast<int>(std::floor(k));
    double sum = std::exp(-lambda);
    if (k_int == 0) {
        return sum;
    }
    double term = sum;
    for (int i = 0; i < k_int; ++i) {
        term *= lambda / static_cast<double>(i + 1);
        sum += term;
    }
    return sum;
}

double chi2_pdf(double x, double df) {
    if (x <= 0.0 || df <= 0.0) {
        return 0.0;
    }
    const double k = df / 2.0;
    return std::pow(x, k - 1.0) * std::exp(-x / 2.0) / (std::pow(2.0, k) * std::tgamma(k));
}

double chi2_cdf(double x, double df) {
    if (x <= 0.0 || df <= 0.0) {
        return 0.0;
    }
    return gamma_inc_reg(df / 2.0, x / 2.0);
}

double chi2_ppf(double p, double df) {
    if (df <= 0.0) {
        return 0.0;
    }
    if (p <= 0.0) {
        return 0.0;
    }
    if (p >= 1.0) {
        return k_ppf_pos_tail;
    }
    const auto residual = [&](double x) { return chi2_cdf(x, df) - p; };
    double hi = std::max(df + 10.0 * std::sqrt(2.0 * df), 1.0);
    hi = expand_upper_until_cdf([&](double x) { return chi2_cdf(x, df); }, p, hi);
    return bisect_increasing(residual, 0.0, hi);
}

double uniform_pdf(double x, double a, double b) {
    if (b <= a || x < a || x > b) {
        return 0.0;
    }
    return 1.0 / (b - a);
}

double uniform_cdf(double x, double a, double b) {
    if (b <= a) {
        return 0.0;
    }
    if (x <= a) {
        return 0.0;
    }
    if (x >= b) {
        return 1.0;
    }
    return (x - a) / (b - a);
}

double t_pdf(double x, double df) {
    if (df <= 0.0) {
        return 0.0;
    }
    const double num = std::tgamma((df + 1.0) / 2.0);
    const double den = std::sqrt(df * M_PI) * std::tgamma(df / 2.0);
    return num / den * std::pow(1.0 + x * x / df, -(df + 1.0) / 2.0);
}

double t_cdf(double x, double df) {
    if (df <= 0.0) {
        return 0.0;
    }
    if (x == 0.0) {
        return 0.5;  // By symmetry of the t-distribution
    }
    const double z = df / (df + x * x);
    const double ib = beta_inc_reg(z, df / 2.0, 0.5);
    return x > 0.0 ? 1.0 - 0.5 * ib : 0.5 * ib;
}

double t_ppf(double p, double df) {
    if (df <= 0.0) {
        return 0.0;
    }
    if (p <= 0.0) {
        return k_ppf_neg_tail;
    }
    if (p >= 1.0) {
        return k_ppf_pos_tail;
    }
    if (p == 0.5) {
        return 0.0;
    }
    const auto residual = [&](double x) { return t_cdf(x, df) - p; };
    double lo = -1.0;
    double hi = 1.0;
    while (t_cdf(lo, df) > p && lo > -k_ppf_bracket_max) {
        lo *= 2.0;
    }
    while (t_cdf(hi, df) < p && hi < k_ppf_bracket_max) {
        hi *= 2.0;
    }
    return bisect_increasing(residual, lo, hi);
}

double gamma_pdf(double x, double shape, double scale) {
    if (x <= 0.0 || shape <= 0.0 || scale <= 0.0) {
        return 0.0;
    }
    const double z = x / scale;
    return std::pow(z, shape - 1.0) * std::exp(-z) / (scale * std::tgamma(shape));
}

double gamma_cdf(double x, double shape, double scale) {
    if (x <= 0.0 || shape <= 0.0 || scale <= 0.0) {
        return 0.0;
    }
    return gamma_inc_reg(shape, x / scale);
}

double gamma_ppf(double p, double shape, double scale) {
    if (shape <= 0.0 || scale <= 0.0) {
        return 0.0;
    }
    if (p <= 0.0) {
        return 0.0;
    }
    if (p >= 1.0) {
        return k_ppf_pos_tail;
    }
    if (shape == 1.0) {
        return exp_ppf(p, 1.0 / scale);
    }
    const auto residual = [&](double x) { return gamma_cdf(x, shape, scale) - p; };
    double hi = std::max(shape * scale + 10.0 * std::sqrt(shape) * scale, scale);
    hi = expand_upper_until_cdf([&](double x) { return gamma_cdf(x, shape, scale); }, p, hi);
    return bisect_increasing(residual, 0.0, hi);
}

double beta_pdf(double x, double alpha, double beta) {
    if (x < 0.0 || x > 1.0 || alpha <= 0.0 || beta <= 0.0) {
        return 0.0;
    }
    return std::pow(x, alpha - 1.0) * std::pow(1.0 - x, beta - 1.0) / beta_func(alpha, beta);
}

double beta_cdf(double x, double alpha, double beta) {
    if (x <= 0.0 || alpha <= 0.0 || beta <= 0.0) {
        return 0.0;
    }
    if (x >= 1.0) {
        return 1.0;
    }
    return beta_inc_reg(x, alpha, beta);
}

double beta_ppf(double p, double alpha, double beta) {
    if (alpha <= 0.0 || beta <= 0.0) {
        return 0.0;
    }
    if (p <= 0.0) {
        return 0.0;
    }
    if (p >= 1.0) {
        return 1.0;
    }
    if (alpha == 1.0 && beta == 1.0) {
        return p;
    }
    const auto residual = [&](double x) { return beta_cdf(x, alpha, beta) - p; };
    return bisect_increasing(residual, 0.0, 1.0);
}

double f_pdf(double x, double d1, double d2) {
    if (x <= 0.0 || d1 <= 0.0 || d2 <= 0.0) {
        return 0.0;
    }
    const double a = d1 / 2.0;
    const double b = d2 / 2.0;
    const double z = d1 * x / d2;
    const double log_num = a * std::log(d1 / d2) + (a - 1.0) * std::log(x);
    const double log_den = std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b) +
                           (a + b) * std::log(1.0 + z);
    return std::exp(log_num - log_den);
}

double f_cdf(double x, double d1, double d2) {
    if (x <= 0.0 || d1 <= 0.0 || d2 <= 0.0) {
        return 0.0;
    }
    const double z = d1 * x / (d1 * x + d2);
    return beta_inc_reg(z, d1 / 2.0, d2 / 2.0);
}

double f_ppf(double p, double d1, double d2) {
    if (d1 <= 0.0 || d2 <= 0.0) {
        return 0.0;
    }
    if (p <= 0.0) {
        return 0.0;
    }
    if (p >= 1.0) {
        return k_ppf_pos_tail;
    }
    const auto residual = [&](double x) { return f_cdf(x, d1, d2) - p; };
    double hi = (d2 > 2.0) ? (d2 / (d2 - 2.0)) * (d1 / d2) : 1.0;
    hi = std::max(hi + 10.0 * std::sqrt(2.0 * std::max(hi, 1.0)), 1.0);
    hi = expand_upper_until_cdf([&](double x) { return f_cdf(x, d1, d2); }, p, hi);
    return bisect_increasing(residual, 0.0, hi);
}

double lognormal_pdf(double x, double mu, double sigma) {
    if (x <= 0.0 || sigma <= 0.0) {
        return 0.0;
    }
    return norm_pdf(std::log(x), mu, sigma) / x;
}

double lognormal_cdf(double x, double mu, double sigma) {
    if (x <= 0.0 || sigma <= 0.0) {
        return 0.0;
    }
    return norm_cdf(std::log(x), mu, sigma);
}

double lognormal_ppf(double p, double mu, double sigma) {
    if (sigma <= 0.0) {
        return 0.0;
    }
    if (p <= 0.0) {
        return 0.0;
    }
    if (p >= 1.0) {
        return k_ppf_pos_tail;
    }
    return std::exp(norm_ppf(p, mu, sigma));
}

double weibull_pdf(double x, double lambda, double k) {
    if (x < 0.0 || lambda <= 0.0 || k <= 0.0) {
        return 0.0;
    }
    const double z = x / lambda;
    return (k / lambda) * std::pow(z, k - 1.0) * std::exp(-std::pow(z, k));
}

double weibull_cdf(double x, double lambda, double k) {
    if (x < 0.0 || lambda <= 0.0 || k <= 0.0) {
        return 0.0;
    }
    return 1.0 - std::exp(-std::pow(x / lambda, k));
}

double weibull_ppf(double p, double lambda, double k) {
    if (lambda <= 0.0 || k <= 0.0) {
        return 0.0;
    }
    if (p <= 0.0) {
        return 0.0;
    }
    if (p >= 1.0) {
        return k_ppf_pos_tail;
    }
    return lambda * std::pow(-std::log(1.0 - p), 1.0 / k);
}

double laplace_pdf(double x, double mu, double b) {
    if (b <= 0.0) {
        return 0.0;
    }
    return std::exp(-std::abs(x - mu) / b) / (2.0 * b);
}

double laplace_cdf(double x, double mu, double b) {
    if (b <= 0.0) {
        return 0.0;
    }
    if (x < mu) {
        return 0.5 * std::exp((x - mu) / b);
    }
    return 1.0 - 0.5 * std::exp(-(x - mu) / b);
}

double laplace_ppf(double p, double mu, double b) {
    if (b <= 0.0) {
        return 0.0;
    }
    if (p <= 0.0) {
        return k_ppf_neg_tail;
    }
    if (p >= 1.0) {
        return k_ppf_pos_tail;
    }
    const double d = p - 0.5;
    const double sign = (d > 0.0) ? 1.0 : ((d < 0.0) ? -1.0 : 0.0);
    return mu - b * sign * std::log(1.0 - 2.0 * std::abs(d));
}

double logistic_pdf(double x, double mu, double s) {
    if (s <= 0.0) {
        return 0.0;
    }
    const double z = (x - mu) / s;
    if (z >= 0.0) {
        const double ez = std::exp(-z);
        const double denom = 1.0 + ez;
        return ez / (s * denom * denom);
    }
    const double ez = std::exp(z);
    const double denom = 1.0 + ez;
    return ez / (s * denom * denom);
}

double logistic_cdf(double x, double mu, double s) {
    if (s <= 0.0) {
        return 0.0;
    }
    const double z = (x - mu) / s;
    if (z >= 0.0) {
        return 1.0 / (1.0 + std::exp(-z));
    }
    const double ez = std::exp(z);
    return ez / (1.0 + ez);
}

double logistic_ppf(double p, double mu, double s) {
    if (s <= 0.0) {
        return 0.0;
    }
    if (p <= 0.0) {
        return k_ppf_neg_tail;
    }
    if (p >= 1.0) {
        return k_ppf_pos_tail;
    }
    return mu + s * std::log(p / (1.0 - p));
}

double gumbel_pdf(double x, double mu, double beta) {
    if (beta <= 0.0) {
        return 0.0;
    }
    const double z = (x - mu) / beta;
    return std::exp(-z - std::exp(-z)) / beta;
}

double gumbel_cdf(double x, double mu, double beta) {
    if (beta <= 0.0) {
        return 0.0;
    }
    const double z = (x - mu) / beta;
    return std::exp(-std::exp(-z));
}

double gumbel_ppf(double p, double mu, double beta) {
    if (beta <= 0.0) {
        return 0.0;
    }
    if (p <= 0.0) {
        return k_ppf_neg_tail;
    }
    if (p >= 1.0) {
        return k_ppf_pos_tail;
    }
    return mu - beta * std::log(-std::log(p));
}

double cauchy_pdf(double x, double x0, double gamma) {
    if (gamma <= 0.0) {
        return 0.0;
    }
    const double z = (x - x0) / gamma;
    return 1.0 / (M_PI * gamma * (1.0 + z * z));
}

double cauchy_cdf(double x, double x0, double gamma) {
    if (gamma <= 0.0) {
        return 0.0;
    }
    return std::atan((x - x0) / gamma) / M_PI + 0.5;
}

double cauchy_ppf(double p, double x0, double gamma) {
    if (gamma <= 0.0) {
        return 0.0;
    }
    if (p <= 0.0) {
        return k_ppf_neg_tail;
    }
    if (p >= 1.0) {
        return k_ppf_pos_tail;
    }
    return x0 + gamma * std::tan(M_PI * (p - 0.5));
}

double pareto_pdf(double x, double x_m, double alpha) {
    if (x_m <= 0.0 || alpha <= 0.0 || x < x_m) {
        return 0.0;
    }
    return alpha * std::pow(x_m, alpha) / std::pow(x, alpha + 1.0);
}

double pareto_cdf(double x, double x_m, double alpha) {
    if (x_m <= 0.0 || alpha <= 0.0 || x < x_m) {
        return 0.0;
    }
    return 1.0 - std::pow(x_m / x, alpha);
}

double pareto_ppf(double p, double x_m, double alpha) {
    if (x_m <= 0.0 || alpha <= 0.0) {
        return 0.0;
    }
    if (p <= 0.0) {
        return x_m;
    }
    if (p >= 1.0) {
        return k_ppf_pos_tail;
    }
    return x_m / std::pow(1.0 - p, 1.0 / alpha);
}

double rayleigh_pdf(double x, double sigma) {
    if (x < 0.0 || sigma <= 0.0) {
        return 0.0;
    }
    const double s2 = sigma * sigma;
    return (x / s2) * std::exp(-x * x / (2.0 * s2));
}

double rayleigh_cdf(double x, double sigma) {
    if (x < 0.0 || sigma <= 0.0) {
        return 0.0;
    }
    return 1.0 - std::exp(-x * x / (2.0 * sigma * sigma));
}

double rayleigh_ppf(double p, double sigma) {
    if (sigma <= 0.0) {
        return 0.0;
    }
    if (p <= 0.0) {
        return 0.0;
    }
    if (p >= 1.0) {
        return k_ppf_pos_tail;
    }
    return sigma * std::sqrt(-2.0 * std::log(1.0 - p));
}

} // namespace ms
