#include "ms/prob/prob.hpp"
#include "ms/special/special.hpp"
#include <cmath>
#include <limits>

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
    if (x == 0.0) {
        return 0.5;  // By symmetry of the t-distribution
    }
    // Integrate t_pdf from 0 to |x| and use symmetry
    const double limit = std::abs(x);
    const int steps = 2000;
    const double dx = limit / static_cast<double>(steps);
    double sum = 0.0;
    for (int i = 0; i < steps; ++i) {
        const double t = (static_cast<double>(i) + 0.5) * dx;
        sum += t_pdf(t, df);
    }
    const double half_area = sum * dx;
    return x > 0.0 ? 0.5 + half_area : 0.5 - half_area;
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
    return regularized_lower_incomplete_gamma(shape, x / scale);
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
    return regularized_incomplete_beta(alpha, beta, x);
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
    return regularized_incomplete_beta(d1 / 2.0, d2 / 2.0, z);
}

} // namespace ms
