#include "ms/prob/prob.hpp"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ms {

namespace {

double regularized_lower_incomplete_gamma(double a, double x) {
    if (x <= 0.0) {
        return 0.0;
    }
    double sum = 0.0;
    double term = 1.0 / a;
    sum = term;
    for (int n = 1; n < 200; ++n) {
        term *= x / (a + static_cast<double>(n));
        sum += term;
        if (std::abs(term) < 1e-12 * std::abs(sum)) {
            break;
        }
    }
    return std::exp(-x + a * std::log(x) - std::lgamma(a)) * sum;
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
    if (p <= 0.0 || p >= 1.0) {
        return 0.0;
    }
    double z = 0.0;
    for (int i = 0; i < 100; ++i) {
        const double q = 0.5 * (1.0 + std::erf(z / std::sqrt(2.0)));
        if (q > p) {
            z -= 0.01;
        } else {
            z += 0.01;
        }
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

double binom_pdf(int k, int n, double p) {
    if (k < 0 || k > n || p < 0.0 || p > 1.0) {
        return 0.0;
    }
    double binom = 1.0;
    for (int i = 0; i < k; ++i) {
        binom *= (n - i);
    }
    for (int i = 0; i < k; ++i) {
        binom /= (i + 1);
    }
    return binom * std::pow(p, k) * std::pow(1.0 - p, n - k);
}

double binom_cdf(int k, int n, double p) {
    double sum = 0.0;
    for (int i = 0; i <= k; ++i) {
        sum += binom_pdf(i, n, p);
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
    double sum = 0.0;
    for (double i = 0.0; i <= k; i += 1.0) {
        sum += pois_pdf(i, lambda);
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
    return regularized_lower_incomplete_gamma(df / 2.0, x / 2.0);
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
    const int steps = 1000;
    const double dx = x / static_cast<double>(steps);
    double sum = 0.0;
    for (int i = 0; i < steps; ++i) {
        const double t = (static_cast<double>(i) + 0.5) * dx;
        sum += t_pdf(t, df);
    }
    return sum * dx;
}

double gamma_pdf(double x, double shape, double scale) {
    if (x <= 0.0 || shape <= 0.0 || scale <= 0.0) {
        return 0.0;
    }
    const double z = x / scale;
    return std::pow(z, shape - 1.0) * std::exp(-z) / (scale * std::tgamma(shape));
}

} // namespace ms
