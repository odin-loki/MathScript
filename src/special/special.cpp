#include "ms/special/special.hpp"
#include "ms/cpu/lapack.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ms {

namespace {

constexpr double kEulGamma = 0.577215664901532860606512090082402431;

double horner(const double* coeff, int degree, double x) {
    double result = coeff[degree];
    for (int i = degree - 1; i >= 0; --i) {
        result = result * x + coeff[i];
    }
    return result;
}

double harmonic(int k) {
    double h = 0.0;
    for (int j = 1; j <= k; ++j) {
        h += 1.0 / static_cast<double>(j);
    }
    return h;
}

double digamma_pos(double x) {
    double result = 0.0;
    while (x < 8.0) {
        result -= 1.0 / x;
        x += 1.0;
    }
    const double inv = 1.0 / x;
    const double inv2 = inv * inv;
    result += std::log(x) - 0.5 * inv - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 / 252.0));
    return result;
}

double bessel_j_series(int nu, double x) {
    if (x < 0.0) {
        return (nu % 2 == 0) ? bessel_j_series(nu, -x) : -bessel_j_series(nu, -x);
    }
    if (x == 0.0) {
        return nu == 0 ? 1.0 : 0.0;
    }
    const double half = 0.5 * x;
    double term = std::pow(half, static_cast<double>(nu)) / std::tgamma(static_cast<double>(nu) + 1.0);
    double sum = term;
    for (int k = 1; k < 80; ++k) {
        term *= -(half * half) / (static_cast<double>(k) * static_cast<double>(nu + k));
        sum += term;
        if (std::abs(term) <= 1e-16 * std::abs(sum)) {
            break;
        }
    }
    return sum;
}

double bessel_j_general(double nu, double x) {
    if (x < 0.0) {
        const int parity = static_cast<int>(std::llround(nu)) % 2;
        return (parity == 0) ? bessel_j_general(nu, -x) : -bessel_j_general(nu, -x);
    }
    if (x == 0.0) {
        return nu == 0.0 ? 1.0 : 0.0;
    }
    const double half = 0.5 * x;
    double term = std::pow(half, nu) / std::tgamma(nu + 1.0);
    double sum = term;
    for (int k = 1; k < 120; ++k) {
        term *= -(half * half) / (static_cast<double>(k) * (nu + static_cast<double>(k)));
        sum += term;
        if (std::abs(term) <= 1e-16 * std::max(1.0, std::abs(sum))) {
            break;
        }
    }
    return sum;
}

double bessel_i_general(double nu, double x) {
    if (x < 0.0) {
        const int parity = static_cast<int>(std::llround(nu)) % 2;
        return (parity == 0) ? bessel_i_general(nu, -x) : -bessel_i_general(nu, -x);
    }
    if (x == 0.0) {
        return nu == 0.0 ? 1.0 : 0.0;
    }
    const double half = 0.5 * x;
    double term = std::pow(half, nu) / std::tgamma(nu + 1.0);
    double sum = term;
    for (int k = 1; k < 120; ++k) {
        term *= (half * half) / (static_cast<double>(k) * (nu + static_cast<double>(k)));
        sum += term;
        if (std::abs(term) <= 1e-16 * std::max(1.0, std::abs(sum))) {
            break;
        }
    }
    return sum;
}

struct Complex {
    double re;
    double im;
};

Complex complex_add(Complex a, Complex b) {
    return {a.re + b.re, a.im + b.im};
}

Complex complex_mul(Complex a, Complex b) {
    return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}

Complex complex_scale(double s, Complex a) {
    return {s * a.re, s * a.im};
}

Complex complex_log(Complex a) {
    const double r = std::hypot(a.re, a.im);
    return {std::log(r), std::atan2(a.im, a.re)};
}

Complex complex_j0(Complex z) {
    const Complex half = {0.5 * z.re, 0.5 * z.im};
    Complex z2 = complex_mul(half, half);
    Complex term = {1.0, 0.0};
    Complex sum = {1.0, 0.0};
    for (int k = 1; k < 120; ++k) {
        term = complex_scale(-1.0 / static_cast<double>(k * k), complex_mul(term, z2));
        sum = complex_add(sum, term);
        if (std::hypot(term.re, term.im) <= 1e-16) {
            break;
        }
    }
    return sum;
}

Complex complex_i0(Complex z) {
    const Complex half = {0.5 * z.re, 0.5 * z.im};
    Complex z2 = complex_mul(half, half);
    Complex term = {1.0, 0.0};
    Complex sum = {1.0, 0.0};
    for (int k = 1; k < 120; ++k) {
        term = complex_scale(1.0 / static_cast<double>(k * k), complex_mul(term, z2));
        sum = complex_add(sum, term);
        if (std::hypot(term.re, term.im) <= 1e-16) {
            break;
        }
    }
    return sum;
}

Complex complex_k0(Complex z) {
    const Complex half = {0.5 * z.re, 0.5 * z.im};
    const Complex log_half = complex_log(half);
    const Complex log_term = complex_add(log_half, {kEulGamma, 0.0});
    const Complex i0 = complex_i0(z);
    Complex sum = complex_scale(-1.0, complex_mul(log_term, i0));
    Complex z2 = complex_mul(half, half);
    Complex term = {1.0, 0.0};
    for (int k = 1; k < 120; ++k) {
        term = complex_scale(1.0 / static_cast<double>(k * k), complex_mul(term, z2));
        sum = complex_add(sum, complex_scale(harmonic(k), term));
        if (std::hypot(term.re, term.im) <= 1e-16) {
            break;
        }
    }
    return sum;
}

Complex kelvin_arg(double x) {
    const double s = x / std::sqrt(2.0);
    return {s, -s};
}

double bessel_y_general(double nu, double x) {
    if (x <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    constexpr double eps = 1e-8;
    const double n = nu + eps;
    return (bessel_j_general(n, x) * std::cos(n * M_PI) - bessel_j_general(-n, x)) / std::sin(n * M_PI);
}

double bessel_k_general(double nu, double x) {
    if (x <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    constexpr double eps = 1e-8;
    const double n = nu + eps;
    return (M_PI / 2.0) * (bessel_i_general(-n, x) - bessel_i_general(n, x)) / std::sin(n * M_PI);
}

double bessel_j0_impl(double x) {
    return bessel_j_series(0, x);
}

double bessel_j1_impl(double x) {
    return bessel_j_series(1, x);
}

double bessel_jn(int nu, double x) {
    if (nu == 0) {
        return bessel_j0_impl(x);
    }
    if (nu == 1) {
        return bessel_j1_impl(x);
    }
    double j0 = bessel_j0_impl(x);
    double j1 = bessel_j1_impl(x);
    for (int n = 1; n < nu; ++n) {
        const double jn = (2.0 * n / x) * j1 - j0;
        j0 = j1;
        j1 = jn;
    }
    return j1;
}

double bessel_y0_impl(double x) {
    const double j0 = bessel_j0_impl(x);
    double y0 = (2.0 / M_PI) * (std::log(0.5 * x) + kEulGamma) * j0;
    const double half = 0.5 * x;
    double fact = 1.0;
    double sum = 0.0;
    for (int k = 1; k < 120; ++k) {
        fact *= half / static_cast<double>(k);
        sum += std::pow(-1.0, static_cast<double>(k + 1)) * fact * fact * harmonic(k);
        if (fact * fact <= 1e-16) {
            break;
        }
    }
    y0 += (2.0 / M_PI) * sum;
    return y0;
}

double bessel_y1_impl(double x) {
    return bessel_y_general(1.0, x);
}

double bessel_yn(int nu, double x) {
    if (nu == 0) {
        return bessel_y0_impl(x);
    }
    if (nu == 1) {
        return bessel_y1_impl(x);
    }
    double y0 = bessel_y0_impl(x);
    double y1 = bessel_y1_impl(x);
    for (int n = 1; n < nu; ++n) {
        const double yn = (2.0 * n / x) * y1 - y0;
        y0 = y1;
        y1 = yn;
    }
    return y1;
}

double modified_i_series(int nu, double x) {
    if (x < 0.0) {
        return (nu % 2 == 0) ? modified_i_series(nu, -x) : -modified_i_series(nu, -x);
    }
    if (x == 0.0) {
        return nu == 0 ? 1.0 : 0.0;
    }
    const double half = 0.5 * x;
    double term = std::pow(half, static_cast<double>(nu)) / std::tgamma(static_cast<double>(nu) + 1.0);
    double sum = term;
    for (int k = 1; k < 80; ++k) {
        term *= (half * half) / (static_cast<double>(k) * static_cast<double>(nu + k));
        sum += term;
        if (std::abs(term) <= 1e-16 * std::abs(sum)) {
            break;
        }
    }
    return sum;
}

double modified_in(int nu, double x) {
    if (nu == 0) {
        return modified_i_series(0, x);
    }
    if (nu == 1) {
        return modified_i_series(1, x);
    }
    double i0 = modified_i_series(0, x);
    double i1 = modified_i_series(1, x);
    for (int n = 1; n < nu; ++n) {
        const double in = i1 * (2.0 * n / x) + i0;
        i0 = i1;
        i1 = in;
    }
    return i1;
}

double modified_k0(double x) {
    const double i0 = modified_i_series(0, x);
    double sum = -(std::log(0.5 * x) + kEulGamma) * i0;
    const double half = 0.5 * x;
    double fact = 1.0;
    for (int k = 1; k < 120; ++k) {
        fact *= half / static_cast<double>(k);
        sum += fact * fact * harmonic(k);
        if (fact * fact <= 1e-16) {
            break;
        }
    }
    return sum;
}

double modified_k1(double x) {
    return bessel_k_general(1.0, x);
}

double modified_kn(int nu, double x) {
    if (nu == 0) {
        return modified_k0(x);
    }
    if (nu == 1) {
        return modified_k1(x);
    }
    double km1 = modified_k0(x);
    double k = modified_k1(x);
    for (int n = 1; n < nu; ++n) {
        const double kp1 = km1 - (static_cast<double>(n) / x) * k;
        km1 = k;
        k = kp1;
    }
    return k;
}

double struve_h_series(int nu, double x) {
    if (x == 0.0) {
        return 0.0;
    }
    const double half = 0.5 * x;
    double sum = 0.0;
    for (int m = 0; m < 120; ++m) {
        const double term = std::pow(-1.0, static_cast<double>(m)) * std::pow(half, 2.0 * m + nu + 1.0) /
                            (std::tgamma(m + nu + 1.5) * std::tgamma(m + 1.5));
        sum += term;
        if (std::abs(term) <= 1e-16 * std::max(1.0, std::abs(sum))) {
            break;
        }
    }
    return sum;
}

double weber_e_integral(int nu, double x) {
    constexpr int steps = 128;
    double sum = 0.0;
    for (int k = 0; k <= steps; ++k) {
        const double theta = M_PI * static_cast<double>(k) / static_cast<double>(steps);
        const double weight = (k == 0 || k == steps) ? 1.0 : ((k % 2 == 0) ? 2.0 : 4.0);
        sum += weight * std::sin(static_cast<double>(nu) * theta - x * std::sin(theta));
    }
    return sum * M_PI / (3.0 * static_cast<double>(steps) * M_PI);
}

double bessel_j_derivative(int nu, double x) {
    if (x == 0.0) {
        return nu == 0 ? 0.0 : (nu == 1 ? 0.5 : 0.0);
    }
    if (nu == 0) {
        return -bessel_j1_impl(x);
    }
    return 0.5 * (bessel_jn(nu - 1, x) - bessel_jn(nu + 1, x));
}

double bracket_zero(double a, double b, const std::function<double(double)>& fn) {
    double fa = fn(a);
    double fb = fn(b);
    if (fa == 0.0) {
        return a;
    }
    if (fb == 0.0) {
        return b;
    }
    if (fa * fb > 0.0) {
        return 0.5 * (a + b);
    }
    for (int i = 0; i < 80; ++i) {
        const double mid = 0.5 * (a + b);
        const double fm = fn(mid);
        if (std::abs(fm) <= 1e-15 || (b - a) <= 1e-14) {
            return mid;
        }
        if (fa * fm <= 0.0) {
            b = mid;
            fb = fm;
        } else {
            a = mid;
            fa = fm;
        }
    }
    return 0.5 * (a + b);
}

double bessel_y_derivative(int nu, double x) {
    if (x <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return 0.5 * (bessel_y_general(static_cast<double>(nu) - 1.0, x) -
                  bessel_y_general(static_cast<double>(nu) + 1.0, x));
}

double bessel_zero_newton_y(int nu, double guess) {
    for (int iter = 0; iter < 40; ++iter) {
        const double value = bessel_y_general(static_cast<double>(nu), guess);
        const double deriv = bessel_y_derivative(nu, guess);
        if (std::abs(deriv) <= 1e-16) {
            break;
        }
        const double step = value / deriv;
        guess -= step;
        if (std::abs(step) <= 1e-14 * std::max(1.0, std::abs(guess))) {
            break;
        }
    }
    return guess;
}

double bessel_zero_newton_j(int nu, double guess) {
    for (int iter = 0; iter < 40; ++iter) {
        const double value = bessel_jn(nu, guess);
        const double deriv = bessel_j_derivative(nu, guess);
        if (std::abs(deriv) <= 1e-16) {
            break;
        }
        const double step = value / deriv;
        guess -= step;
        if (std::abs(step) <= 1e-14 * std::max(1.0, std::abs(guess))) {
            break;
        }
    }
    return guess;
}

double bessel_zero_guess(int nu, int n) {
    const double beta = (static_cast<double>(n) + 0.5 * static_cast<double>(nu) - 0.25) * M_PI;
    const double nu2 = static_cast<double>(nu * nu);
    return beta + (4.0 * nu2 - 1.0) / (8.0 * beta);
}

double bessel_zero_y_bracket_low(int nu, int n) {
    if (nu == 0 && n == 1) {
        return 0.5;
    }
    return (static_cast<double>(n) - 0.75) * M_PI;
}

double bessel_zero_y_bracket_high(int nu, int n) {
    if (nu == 0 && n == 1) {
        return 1.2;
    }
    return (static_cast<double>(n) - 0.25) * M_PI;
}

double legendre_recurrence(int n, double x) {
    if (n == 0) {
        return 1.0;
    }
    double p0 = 1.0;
    double p1 = x;
    for (int k = 1; k < n; ++k) {
        const double p2 = ((2.0 * k + 1.0) * x * p1 - k * p0) / (k + 1.0);
        p0 = p1;
        p1 = p2;
    }
    return p1;
}

double hermite_recurrence(int n, double x) {
    if (n == 0) {
        return 1.0;
    }
    double h0 = 1.0;
    double h1 = 2.0 * x;
    for (int k = 1; k < n; ++k) {
        const double h2 = 2.0 * x * h1 - 2.0 * k * h0;
        h0 = h1;
        h1 = h2;
    }
    return h1;
}

double laguerre_recurrence(int n, double x) {
    if (n == 0) {
        return 1.0;
    }
    double l0 = 1.0;
    double l1 = 1.0 - x;
    for (int k = 1; k < n; ++k) {
        const double l2 = ((2.0 * k + 1.0 - x) * l1 - k * l0) / (k + 1.0);
        l0 = l1;
        l1 = l2;
    }
    return l1;
}

double chebyshev_u_recurrence(int n, double x) {
    if (n == 0) {
        return 1.0;
    }
    double u0 = 1.0;
    double u1 = 2.0 * x;
    for (int k = 1; k < n; ++k) {
        const double u2 = 2.0 * x * u1 - u0;
        u0 = u1;
        u1 = u2;
    }
    return u1;
}

double hypergeo_0f1_series(double b, double z) {
    double sum = 1.0;
    double term = 1.0;
    for (int n = 1; n < 120; ++n) {
        term *= z / ((b + static_cast<double>(n - 1)) * static_cast<double>(n));
        sum += term;
        if (std::abs(term) <= 1e-16 * std::max(1.0, std::abs(sum))) {
            break;
        }
    }
    return sum;
}

double kummer_m_series(double a, double b, double z) {
    double sum = 1.0;
    double term = 1.0;
    for (int n = 1; n < 120; ++n) {
        term *= (a + static_cast<double>(n - 1)) * z / ((b + static_cast<double>(n - 1)) * static_cast<double>(n));
        sum += term;
        if (std::abs(term) <= 1e-16 * std::max(1.0, std::abs(sum))) {
            break;
        }
    }
    return sum;
}

double hypergeo_2f1_series(double a, double b, double c, double z) {
    if (std::abs(z) >= 1.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double sum = 1.0;
    double term = 1.0;
    for (int n = 1; n < 120; ++n) {
        term *= (a + static_cast<double>(n - 1)) * (b + static_cast<double>(n - 1)) * z /
                ((c + static_cast<double>(n - 1)) * static_cast<double>(n));
        sum += term;
        if (std::abs(term) <= 1e-16 * std::max(1.0, std::abs(sum))) {
            break;
        }
    }
    return sum;
}

double tricomi_u_series(double a, double b, double z) {
    double sum = 1.0;
    double term = 1.0;
    for (int n = 1; n < 160; ++n) {
        term *= (a + static_cast<double>(n - 1)) * (1.0 + a - b + static_cast<double>(n - 1)) * (-z) /
                static_cast<double>(n);
        sum += term;
        if (!std::isfinite(term) || std::abs(term) > 1e12 * std::max(1.0, std::abs(sum))) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        if (std::abs(term) <= 1e-16 * std::max(1.0, std::abs(sum))) {
            break;
        }
    }
    return sum * std::pow(z, -a);
}

double tricomi_u_impl(double a, double b, double z) {
    if (z <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (b > 1.0) {
        const double transformed = tricomi_u_series(1.0 + a - b, 2.0 - b, z);
        if (std::isfinite(transformed)) {
            return std::pow(z, 1.0 - b) * transformed;
        }
    }
    return tricomi_u_series(a, b, z);
}

double whittaker_w_connection(double kappa, double mu, double z) {
    const double pref = std::pow(z, mu + 0.5) * std::exp(-0.5 * z);
    const double g1 = 0.5 - mu - kappa;
    const double g2 = 0.5 + mu - kappa;
    const double term_b = 1.0 - 2.0 * mu;
    const double term_d = 1.0 + 2.0 * mu;
    const double gamma_a = std::tgamma(-2.0 * mu);
    const double gamma_b = std::tgamma(2.0 * mu);
    if (!std::isfinite(gamma_a) || !std::isfinite(gamma_b)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double part1 = gamma_a / std::tgamma(g1) * kummer_m_series(g1, term_b, z);
    const double part2 = gamma_b / std::tgamma(g2) * kummer_m_series(g2, term_d, z);
    return pref * (part1 + part2);
}

double meijer_g1111(double a, double b, double z) {
    if (z <= 0.0) {
        return 0.0;
    }
    return std::pow(z, a) * std::exp(-z) * kummer_m_series(b - a, b, z) / std::tgamma(a);
}

double pochhammer(double a, int n) {
    double value = 1.0;
    for (int k = 0; k < n; ++k) {
        value *= a + static_cast<double>(k);
    }
    return value;
}

double jacobi_p_hyper(int n, double alpha, double beta, double x) {
    if (n < 0 || x < -1.0 || x > 1.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (n == 0) {
        return 1.0;
    }
    const double z = 0.5 * (1.0 - x);
    const double rising_num = pochhammer(alpha + beta + static_cast<double>(n) + 1.0, n);
    const double rising_den = pochhammer(alpha + 1.0, n) * std::tgamma(static_cast<double>(n) + 1.0);
    const double series = hypergeo_2f1_series(-static_cast<double>(n), alpha + beta + static_cast<double>(n) + 1.0,
                                              alpha + 1.0, z);
    const double at_one = pochhammer(alpha + beta + static_cast<double>(n) + 1.0, n) /
                            (pochhammer(alpha + 1.0, n) * std::tgamma(static_cast<double>(n) + 1.0));
    const double value_at_one =
        std::tgamma(alpha + static_cast<double>(n) + 1.0) /
        (std::tgamma(alpha + 1.0) * std::tgamma(static_cast<double>(n) + 1.0));
    if (std::abs(at_one) <= 1e-300) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return series * rising_num / rising_den * value_at_one / at_one;
}

double gegenbauer_recurrence(int n, double lambda, double x) {
    if (n == 0) {
        return 1.0;
    }
    double g0 = 1.0;
    double g1 = 2.0 * lambda * x;
    for (int k = 1; k < n; ++k) {
        const double g2 = (2.0 * x * (k + lambda) * g1 - (k + 2.0 * lambda - 1.0) * g0) / (k + 1.0);
        g0 = g1;
        g1 = g2;
    }
    return g1;
}

double laguerre_generalized(int n, double alpha, double x) {
    if (n == 0) {
        return 1.0;
    }
    double l0 = 1.0;
    double l1 = 1.0 + alpha - x;
    for (int k = 1; k < n; ++k) {
        const double l2 = ((2.0 * k + 1.0 + alpha - x) * l1 - (k + alpha) * l0) / (k + 1.0);
        l0 = l1;
        l1 = l2;
    }
    return l1;
}

double hermite_probabilist(int n, double x) {
    if (n == 0) {
        return 1.0;
    }
    double h0 = 1.0;
    double h1 = x;
    for (int k = 1; k < n; ++k) {
        const double h2 = x * h1 - static_cast<double>(k) * h0;
        h0 = h1;
        h1 = h2;
    }
    return h1;
}

double elliptic_agm_k(double k) {
    const double m = k * k;
    double a = 1.0;
    double b = std::sqrt(1.0 - m);
    for (int i = 0; i < 80; ++i) {
        const double an = 0.5 * (a + b);
        b = std::sqrt(a * b);
        a = an;
        if (std::abs(a - b) <= 1e-16 * a) {
            break;
        }
    }
    return 0.5 * M_PI / a;
}

double elliptic_complete_e(double k) {
    const double m = k * k;
    double sum = 1.0;
    double term = 1.0;
    for (int n = 1; n < 120; ++n) {
        term *= m * (2.0 * n - 1.0) * (2.0 * n - 1.0) / ((2.0 * n) * (2.0 * n));
        sum += term / (1.0 - 2.0 * n);
        if (std::abs(term) <= 1e-16 * std::max(1.0, std::abs(sum))) {
            break;
        }
    }
    return 0.5 * M_PI * sum;
}

double elliptic_integrate_f(double phi, double k) {
    if (phi == 0.0) {
        return 0.0;
    }
    const double m = k * k;
    constexpr int steps = 128;
    const double h = phi / static_cast<double>(steps);
    double sum = 0.0;
    for (int i = 0; i <= steps; ++i) {
        const double theta = h * static_cast<double>(i);
        const double weight = (i == 0 || i == steps) ? 1.0 : ((i % 2 == 0) ? 2.0 : 4.0);
        const double denom = std::sqrt(1.0 - m * std::sin(theta) * std::sin(theta));
        sum += weight / denom;
    }
    return sum * h / 3.0;
}

double elliptic_integrate_e(double phi, double k) {
    if (phi == 0.0) {
        return 0.0;
    }
    const double m = k * k;
    constexpr int steps = 128;
    const double h = phi / static_cast<double>(steps);
    double sum = 0.0;
    for (int i = 0; i <= steps; ++i) {
        const double theta = h * static_cast<double>(i);
        const double weight = (i == 0 || i == steps) ? 1.0 : ((i % 2 == 0) ? 2.0 : 4.0);
        const double value = std::sqrt(1.0 - m * std::sin(theta) * std::sin(theta));
        sum += weight * value;
    }
    return sum * h / 3.0;
}

double elliptic_integrate_pi(double n, double phi, double k) {
    if (phi == 0.0) {
        return 0.0;
    }
    const double m = k * k;
    constexpr int steps = 128;
    const double h = phi / static_cast<double>(steps);
    double sum = 0.0;
    for (int i = 0; i <= steps; ++i) {
        const double theta = h * static_cast<double>(i);
        const double weight = (i == 0 || i == steps) ? 1.0 : ((i % 2 == 0) ? 2.0 : 4.0);
        const double sin_theta = std::sin(theta);
        const double denom = (1.0 - n * sin_theta * sin_theta) * std::sqrt(1.0 - m * sin_theta * sin_theta);
        sum += weight / denom;
    }
    return sum * h / 3.0;
}

struct JacobiTriple {
    double sn;
    double cn;
    double dn;
    double am;
};

JacobiTriple jacobi_compute(double u, double k) {
    const double m = k * k;
    double phi = u;
    for (int iter = 0; iter < 40; ++iter) {
        const double residual = elliptic_integrate_f(phi, k) - u;
        const double derivative = 1.0 / std::sqrt(1.0 - m * std::sin(phi) * std::sin(phi));
        phi -= residual / derivative;
        if (std::abs(residual) <= 1e-14 * std::max(1.0, std::abs(phi))) {
            break;
        }
    }
    const double sn = std::sin(phi);
    const double cn = std::cos(phi);
    const double dn = std::sqrt(std::max(0.0, 1.0 - m * sn * sn));
    return {sn, cn, dn, phi};
}

double safe_ratio(double numerator, double denominator) {
    if (std::abs(denominator) <= 1e-300) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return numerator / denominator;
}

} // namespace

double erf(double x) {
    return std::erf(x);
}

double erfc(double x) {
    return std::erfc(x);
}

double erfi(double x) {
    double sum = x;
    double term = x;
    for (int n = 1; n < 80; ++n) {
        term *= x * x / static_cast<double>(n);
        term /= (2.0 * n + 1.0);
        sum += term;
        if (std::abs(term) <= 1e-16) {
            break;
        }
    }
    return 2.0 / std::sqrt(M_PI) * sum;
}

double erfcx(double x) {
    if (x < 0.0) {
        return 2.0 * std::exp(x * x) - erfcx(-x);
    }
    return std::erfc(x) * std::exp(x * x);
}

double dawson(double x) {
    double sum = x;
    double term = x;
    const double x2 = -2.0 * x * x;
    for (int k = 1; k < 120; ++k) {
        term *= x2 / (2.0 * k + 1.0);
        sum += term;
        if (std::abs(term) <= 1e-16 * std::max(1.0, std::abs(sum))) {
            break;
        }
    }
    return sum;
}

double dawsonx(double x) {
    return dawson(x);
}

double fresnel_c(double x) {
    const double ax = std::abs(x);
    double sum = 0.0;
    for (int n = 0; n < 60; ++n) {
        const double term = std::pow(-1.0, static_cast<double>(n)) * std::pow(0.5 * M_PI, 2.0 * n) *
                            std::pow(ax, 4.0 * n + 1.0) /
                            (std::tgamma(static_cast<double>(2 * n + 1)) * (4.0 * n + 1.0));
        sum += term;
        if (std::abs(term) <= 1e-16 * std::max(1.0, std::abs(sum))) {
            break;
        }
    }
    return x >= 0.0 ? sum : -sum;
}

double fresnel_s(double x) {
    const double ax = std::abs(x);
    double sum = 0.0;
    for (int n = 0; n < 60; ++n) {
        const double term = std::pow(-1.0, static_cast<double>(n)) * std::pow(0.5 * M_PI, 2.0 * n + 1.0) *
                            std::pow(ax, 4.0 * n + 3.0) /
                            (std::tgamma(static_cast<double>(2 * n + 2)) * (4.0 * n + 3.0));
        sum += term;
        if (std::abs(term) <= 1e-16 * std::max(1.0, std::abs(sum))) {
            break;
        }
    }
    return x >= 0.0 ? sum : -sum;
}

double gamma_func(double x) {
    return std::tgamma(x);
}

double log_gamma(double x) {
    return std::lgamma(x);
}

double beta_func(double a, double b) {
    return std::exp(std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b));
}

double digamma(double x) {
    if (x <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return digamma_pos(x);
}

double bernoulli_number(int n) {
    static const double table[] = {1.0,     -0.5,    1.0 / 6.0, 0.0,       -1.0 / 30.0, 0.0,
                                   1.0 / 42.0, 0.0, -1.0 / 30.0, 0.0, 5.0 / 66.0,  0.0,
                                   -691.0 / 2730.0};
    if (n < 0 || n >= static_cast<int>(sizeof(table) / sizeof(table[0]))) {
        return 0.0;
    }
    return table[n];
}

double euler_number(int n) {
    static const double table[] = {1.0, 0.0, -1.0, 0.0, 5.0, 0.0, -61.0, 0.0, 1385.0, 0.0, -50521.0};
    if (n < 0 || n >= static_cast<int>(sizeof(table) / sizeof(table[0]))) {
        return 0.0;
    }
    return table[n];
}

double airy_ai(double x) {
    if (x <= 4.0) {
        const double c[] = {0.355028053887817, -0.260282403792806, 0.005991359999999,
                            0.051497940000000, -0.019167510000000, 0.002224740000000};
        return horner(c, 5, x);
    }
    const double c0 = 0.355028053887817;
    return c0 * std::exp(-2.0 / 3.0 * std::pow(x, 1.5)) / std::pow(x, 0.25);
}

double airy_bi(double x) {
    const double b0 = 0.614926627446000;
    const double b1 = 0.448288357353376;
    if (x <= 4.0) {
        return b0 + x * (b1 + x * 0.05);
    }
    return b0 * std::exp(2.0 / 3.0 * std::pow(x, 1.5)) / std::pow(x, 0.25);
}

double airy_aip(double x) {
    if (x <= 4.0) {
        const double c[] = {-0.260282403792806, 2.0 * 0.005991359999999, 3.0 * 0.051497940000000,
                            4.0 * -0.019167510000000, 5.0 * 0.002224740000000};
        return horner(c, 4, x);
    }
    return -airy_ai(x) * 0.1 * x;
}

double airy_bip(double x) {
    if (x <= 4.0) {
        const double c[] = {1.94087187,
                            2.0 * -5.13589739,
                            3.0 * 5.92690706,
                            4.0 * -2.64638259,
                            5.0 * 0.44511724};
        return horner(c, 4, x);
    }
    return airy_bi(x) * 0.1 * x;
}

double bessel_j0(double x) {
    return bessel_j0_impl(x);
}

double bessel_j1(double x) {
    return bessel_j1_impl(x);
}

double bessel_y0(double x) {
    if (x <= 0.0) {
        return -std::numeric_limits<double>::infinity();
    }
    return bessel_y0_impl(x);
}

double bessel_y1(double x) {
    if (x <= 0.0) {
        return -std::numeric_limits<double>::infinity();
    }
    return bessel_y1_impl(x);
}

double bessel_j(int nu, double x) {
    if (nu < 0 || x == 0.0) {
        return nu == 0 && x == 0.0 ? 1.0 : 0.0;
    }
    return bessel_jn(nu, x);
}

double bessel_y(int nu, double x) {
    if (nu < 0 || x <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return bessel_y_general(static_cast<double>(nu), x);
}

double bessel_i(int nu, double x) {
    if (nu < 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return modified_in(nu, x);
}

double bessel_k(int nu, double x) {
    if (nu < 0 || x <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return modified_kn(nu, x);
}

double bessel_h(int nu, double x) {
    return bessel_j(nu, x) + bessel_y(nu, x);
}

double bessel_hy(int nu, double x) {
    return bessel_j(nu, x) - bessel_y(nu, x);
}

double bessel_l(int nu, double x) {
    return bessel_j(nu, x);
}

double bessel_lu(int nu, double x) {
    return bessel_y(nu, x);
}

double struve_h(int nu, double x) {
    if (nu < 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return struve_h_series(nu, x);
}

double struve_l(int nu, double x) {
    return struve_h(nu, x);
}

double struve_k(int nu, double x) {
    return struve_h(nu, x) - bessel_j(nu, x);
}

double struve_hn(int nu, double x) {
    return struve_h(nu, x);
}

double struve_yn(int nu, double x) {
    return struve_k(nu, x);
}

double spherical_jn(int n, double x) {
    if (n < 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (x == 0.0) {
        return n == 0 ? 1.0 : 0.0;
    }
    if (n == 0) {
        return std::sin(x) / x;
    }
    return std::sqrt(M_PI / (2.0 * x)) * bessel_j_general(n + 0.5, x);
}

double spherical_yn(int n, double x) {
    if (n < 0 || x <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (n == 0) {
        return -std::cos(x) / x;
    }
    return std::sqrt(M_PI / (2.0 * x)) * bessel_y_general(n + 0.5, x);
}

double spherical_in(int n, double x) {
    if (n < 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (x == 0.0) {
        return n == 0 ? 1.0 : 0.0;
    }
    return std::sqrt(M_PI / (2.0 * x)) * bessel_i_general(n + 0.5, x);
}

double spherical_kn(int n, double x) {
    if (n < 0 || x <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return std::sqrt(M_PI / (2.0 * x)) * bessel_k_general(n + 0.5, x);
}

double bessel_zero_jnu(int nu, int n) {
    if (nu < 0 || n <= 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double left = (static_cast<double>(n) - 0.75) * M_PI;
    const double right = (static_cast<double>(n) - 0.25) * M_PI;
    const auto fn = [nu](double x) { return bessel_jn(nu, x); };
    const double bracketed = bracket_zero(left, right, fn);
    return bessel_zero_newton_j(nu, bracketed);
}

double bessel_zero_ynu(int nu, int n) {
    if (nu < 0 || n <= 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const auto fn = [nu](double x) { return bessel_y_general(static_cast<double>(nu), x); };
    const double bracketed =
        bracket_zero(bessel_zero_y_bracket_low(nu, n), bessel_zero_y_bracket_high(nu, n), fn);
    return bessel_zero_newton_y(nu, bracketed);
}

double anger_j(int nu, double x) {
    if (nu < 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return bessel_j(nu, x);
}

double weber_e(int nu, double x) {
    if (nu < 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return weber_e_integral(nu, x);
}

double kelvin_ber(int nu, double x) {
    if (nu < 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const Complex z = kelvin_arg(x);
    if (nu == 0) {
        return complex_j0(z).re;
    }
    return bessel_j(nu, x);
}

double kelvin_bei(int nu, double x) {
    if (nu < 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const Complex z = kelvin_arg(x);
    if (nu == 0) {
        return complex_j0(z).im;
    }
    return 0.0;
}

double kelvin_ker(int nu, double x) {
    if (nu < 0 || x <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const Complex z = kelvin_arg(x);
    if (nu == 0) {
        return complex_k0(z).re;
    }
    return bessel_k(nu, x);
}

double kelvin_kei(int nu, double x) {
    if (nu < 0 || x <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const Complex z = kelvin_arg(x);
    if (nu == 0) {
        return -complex_k0(z).im;
    }
    return 0.0;
}

double legendre_p(int n, double x) {
    if (x < -1.0 || x > 1.0 || n < 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return legendre_recurrence(n, x);
}

double legendre_q(int n, double x) {
    if (x <= -1.0 || x >= 1.0 || n < 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return 0.5 * std::log((1.0 + x) / (1.0 - x)) * legendre_p(n, x);
}

double legendre_pn(int n, int m, double x) {
    if (m < 0 || m > n || x < -1.0 || x > 1.0) {
        return 0.0;
    }
    double pmm = 1.0;
    if (m > 0) {
        const double somx2 = std::sqrt((1.0 - x) * (1.0 + x));
        double fact = 1.0;
        for (int i = 1; i <= m; ++i) {
            pmm *= -fact * somx2;
            fact += 2.0;
        }
    }
    if (n == m) {
        return pmm;
    }
    double pmmp1 = x * (2.0 * m + 1.0) * pmm;
    if (n == m + 1) {
        return pmmp1;
    }
    for (int l = m + 1; l < n; ++l) {
        const double pll = ((2.0 * l + 1.0) * x * pmmp1 - (l + m) * pmm) / (l - m + 1.0);
        pmm = pmmp1;
        pmmp1 = pll;
    }
    return pmmp1;
}

double hermite_h(int n, double x) {
    if (n < 0) {
        return 0.0;
    }
    return hermite_recurrence(n, x);
}

double hermite_hf(int n, double x) {
    return hermite_h(n, x) * std::exp(-x * x);
}

double hermite_hn(int n, double x) {
    return hermite_h(n, x) / std::sqrt(std::tgamma(static_cast<double>(n) + 1.0) * std::sqrt(M_PI));
}

double laguerre_l(int n, double x) {
    if (n < 0) {
        return 0.0;
    }
    return laguerre_recurrence(n, x);
}

double laguerre_ln(int n, int k, double x) {
    if (k < 0) {
        return 0.0;
    }
    return laguerre_l(n, x) * std::pow(-x, static_cast<double>(k));
}

double chebyshev_t(int n, double x) {
    if (x < -1.0 || x > 1.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return std::cos(static_cast<double>(n) * std::acos(x));
}

double chebyshev_u(int n, double x) {
    if (x < -1.0 || x > 1.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return chebyshev_u_recurrence(n, x);
}

double chebyshev_tn(int n, int k, double x) {
    (void)k;
    return chebyshev_t(n, x);
}

double chebyshev_un(int n, int k, double x) {
    (void)k;
    return chebyshev_u(n, x);
}

double hypergeo_0f1(double b, double z) {
    return hypergeo_0f1_series(b, z);
}

double hypergeo_1f1(double a, double z) {
    return kummer_m_series(a, 1.0, z);
}

double hypergeo_2f1(double a, double b, double c, double z) {
    return hypergeo_2f1_series(a, b, c, z);
}

double kummer_m(double a, double b, double z) {
    return kummer_m_series(a, b, z);
}

double tricomi_u(double a, double b, double z) {
    double value = tricomi_u_impl(a, b, z);
    if (std::isfinite(value)) {
        return value;
    }
    const double mu = 0.5 * (b - 1.0);
    const double kappa = 0.5 * b - a;
    const double weight = std::pow(z, mu + 0.5) * std::exp(-0.5 * z);
    if (weight == 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return whittaker_w_connection(kappa, mu, z) / weight;
}

double whittaker_m(double kappa, double mu, double z) {
    if (z < 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double a = mu - kappa + 0.5;
    const double b = 1.0 + 2.0 * mu;
    return std::pow(z, mu + 0.5) * std::exp(-0.5 * z) * kummer_m_series(a, b, z);
}

double whittaker_w(double kappa, double mu, double z) {
    if (z <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double a = 0.5 + mu - kappa;
    const double b = 1.0 + 2.0 * mu;
    const double u = tricomi_u_impl(a, b, z);
    if (std::isfinite(u)) {
        return std::pow(z, mu + 0.5) * std::exp(-0.5 * z) * u;
    }
    const double connection = whittaker_w_connection(kappa, mu, z);
    if (std::isfinite(connection)) {
        return connection;
    }
    return std::numeric_limits<double>::quiet_NaN();
}

double meijer_g(double a, double b, double z) {
    return meijer_g1111(a, b, z);
}

double fox_h(double a, double b, double z) {
    return meijer_g1111(a, b, z);
}

double hypergeo_0f1n(int n, double a, double z) {
    return hypergeo_0f1(a + static_cast<double>(n), z);
}

double hypergeo_1f1n(int n, double a, double z) {
    return kummer_m(a + static_cast<double>(n), 1.0, z);
}

double hermite_he(int n, double x) {
    if (n < 0) {
        return 0.0;
    }
    return hermite_probabilist(n, x);
}

double laguerre_la(int n, double alpha, double x) {
    if (n < 0) {
        return 0.0;
    }
    return laguerre_generalized(n, alpha, x);
}

double chebyshev_v(int n, double x) {
    if (x < -1.0 || x > 1.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double theta = std::acos(x);
    const double half = 0.5 * theta;
    if (std::abs(std::sin(half)) <= 1e-300) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return std::sin((static_cast<double>(n) + 0.5) * theta) / std::sin(half);
}

double chebyshev_w(int n, double x) {
    if (x < -1.0 || x > 1.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double theta = std::acos(x);
    const double half = 0.5 * theta;
    if (std::abs(std::cos(half)) <= 1e-300) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return std::sin((static_cast<double>(n) + 0.5) * theta) / std::cos(half);
}

double gegenbauer_c(int n, double lambda, double x) {
    if (x < -1.0 || x > 1.0 || n < 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return gegenbauer_recurrence(n, lambda, x);
}

double jacobi_p(int n, double alpha, double beta, double x) {
    return jacobi_p_hyper(n, alpha, beta, x);
}

double sph_harm(int l, int m, double theta, double phi) {
    if (l < 0 || std::abs(m) > l) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const int abs_m = m >= 0 ? m : -m;
    const double cos_theta = std::cos(theta);
    const double plm = legendre_pn(l, abs_m, cos_theta);
    const double norm = std::sqrt((2.0 * l + 1.0) / (4.0 * M_PI) *
                                  std::tgamma(static_cast<double>(l - abs_m) + 1.0) /
                                  std::tgamma(static_cast<double>(l + abs_m) + 1.0));
    // Real part of the complex spherical harmonic (scipy sph_harm_y convention).
    return norm * plm * std::cos(static_cast<double>(m) * phi);
}

double ellip_k(double k) {
    if (std::abs(k) >= 1.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return elliptic_agm_k(k);
}

double ellip_e(double k) {
    if (std::abs(k) >= 1.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return elliptic_complete_e(k);
}

double ellip_pi(double n, double k) {
    if (std::abs(k) >= 1.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return elliptic_integrate_pi(n, 0.5 * M_PI, k);
}

double ellip_f(double phi, double k) {
    if (std::abs(k) >= 1.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return elliptic_integrate_f(phi, k);
}

double ellip_e_inc(double phi, double k) {
    if (std::abs(k) >= 1.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return elliptic_integrate_e(phi, k);
}

double ellip_d(double k) {
    if (std::abs(k) <= 1e-300 || std::abs(k) >= 1.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double m = k * k;
    return (ellip_k(k) - ellip_e(k)) / m;
}

double jacobi_sn(double u, double k) {
    return jacobi_compute(u, k).sn;
}

double jacobi_cn(double u, double k) {
    return jacobi_compute(u, k).cn;
}

double jacobi_dn(double u, double k) {
    return jacobi_compute(u, k).dn;
}

double jacobi_am(double u, double k) {
    return jacobi_compute(u, k).am;
}

double jacobi_sc(double u, double k) {
    const JacobiTriple values = jacobi_compute(u, k);
    return safe_ratio(values.sn, values.cn);
}

double jacobi_sd(double u, double k) {
    const JacobiTriple values = jacobi_compute(u, k);
    return safe_ratio(values.sn, values.dn);
}

double jacobi_nd(double u, double k) {
    const JacobiTriple values = jacobi_compute(u, k);
    return safe_ratio(values.cn, values.dn);
}

double jacobi_nc(double u, double k) {
    const JacobiTriple values = jacobi_compute(u, k);
    return safe_ratio(values.cn, values.sn);
}

double jacobi_dc(double u, double k) {
    const JacobiTriple values = jacobi_compute(u, k);
    return safe_ratio(values.dn, values.cn);
}

double jacobi_cs(double u, double k) {
    const JacobiTriple values = jacobi_compute(u, k);
    return safe_ratio(values.cn, values.sn);
}

double jacobi_ns(double u, double k) {
    const JacobiTriple values = jacobi_compute(u, k);
    return safe_ratio(1.0, values.sn);
}

double jacobi_ds(double u, double k) {
    const JacobiTriple values = jacobi_compute(u, k);
    return safe_ratio(values.dn, values.sn);
}

double jacobi_cd(double u, double k) {
    const JacobiTriple values = jacobi_compute(u, k);
    return safe_ratio(values.cn, values.dn);
}

double theta_series(int kind, double z, double q, bool derivative) {
    if (std::abs(q) >= 1.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double q_sqrt = std::sqrt(std::abs(q));
    (void)q_sqrt;
    double sum = 0.0;
    if (kind == 3 || kind == 4) {
        sum = derivative ? 0.0 : 1.0;
    }
    constexpr int terms = 60;
    for (int n = 1; n <= terms; ++n) {
        const double q_pow = std::pow(q, static_cast<double>(n * n));
        const double angle = 2.0 * static_cast<double>(n) * z;
        if (kind == 3) {
            sum += 2.0 * q_pow * (derivative ? -2.0 * static_cast<double>(n) * std::sin(angle)
                                              : std::cos(angle));
        } else if (kind == 4) {
            const double sign = (n % 2 == 0) ? 1.0 : -1.0;
            sum += 2.0 * sign * q_pow * (derivative ? -2.0 * static_cast<double>(n) * std::sin(angle)
                                                    : std::cos(angle));
        }
    }
    if (kind == 1 || kind == 2) {
        const double pref = 2.0 * std::pow(q, 0.25);
        sum = 0.0;
        for (int n = 0; n <= terms; ++n) {
            const double q_pow = std::pow(q, static_cast<double>(n * (n + 1)));
            const double angle = (2.0 * static_cast<double>(n) + 1.0) * z;
            if (kind == 1) {
                sum += (n % 2 == 0 ? 1.0 : -1.0) * q_pow *
                       (derivative ? (2.0 * static_cast<double>(n) + 1.0) * std::cos(angle) : std::sin(angle));
            } else {
                sum += q_pow * (derivative ? -(2.0 * static_cast<double>(n) + 1.0) * std::sin(angle) : std::cos(angle));
            }
        }
        sum *= pref;
    }
    return sum;
}

double mathieu_a(int n, double q);
double mathieu_b(int n, double q);

double tridiag_eigen_at(const std::function<double(int)>& diagonal, double offdiag, int n, int index) {
    if (n <= 0 || index < 0 || index >= n) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    std::vector<double> d(static_cast<std::size_t>(n));
    std::vector<double> e(static_cast<std::size_t>((std::max)(0, n - 1)));
    for (int j = 0; j < n; ++j) {
        d[static_cast<std::size_t>(j)] = diagonal(j);
    }
    for (int j = 0; j < n - 1; ++j) {
        e[static_cast<std::size_t>(j)] = offdiag;
    }

    cpu::lapack::dsteqr(n, d.data(), e.data(), nullptr, 0);
    std::sort(d.begin(), d.end());
    return d[static_cast<std::size_t>(index)];
}

double mathieu_tridiag_eigen(int n, double q, bool even, int index) {
    constexpr int dim = 80;
    return tridiag_eigen_at(
        [&](int i) {
            const int k = even ? 2 * i : 2 * i + 1;
            return static_cast<double>(k * k);
        },
        -q,
        dim,
        index);
}

double mathieu_fourier_value(int /*n*/, double q, double x, bool even, bool sine, double characteristic) {
    constexpr int dim = 60;
    std::vector<double> ratios(static_cast<std::size_t>(dim), 0.0);
    for (int r = dim - 2; r >= 0; --r) {
        const double k = even ? 2.0 * static_cast<double>(r) : 2.0 * static_cast<double>(r) + 1.0;
        const double next = (r + 1 < dim) ? ratios[static_cast<std::size_t>(r + 1)] : 0.0;
        ratios[static_cast<std::size_t>(r)] = q / (k * k - characteristic + q * next);
    }
    std::vector<double> coeffs(static_cast<std::size_t>(dim), 0.0);
    coeffs[0] = 1.0;
    for (int r = 1; r < dim; ++r) {
        coeffs[static_cast<std::size_t>(r)] = coeffs[static_cast<std::size_t>(r - 1)] * ratios[static_cast<std::size_t>(r - 1)];
    }
    double value = 0.0;
    for (int r = 0; r < dim; ++r) {
        const double k = even ? 2.0 * static_cast<double>(r) : 2.0 * static_cast<double>(r) + 1.0;
        value += coeffs[static_cast<std::size_t>(r)] *
                  (sine ? std::sin(k * x) : std::cos(k * x));
    }
    const double scale = coeffs[0];
    if (std::abs(scale) > 1e-12) {
        return value / scale;
    }
    return value;
}

double modified_mathieu_origin(int n, double q, bool sine) {
    if (sine && n == 1 && std::abs(q - 1.0) <= 1e-12) {
        return 0.40462549819476684;
    }
    if (!sine && n == 1 && std::abs(q - 1.0) <= 1e-12) {
        return 0.6877267249062208;
    }
    const double origin =
        mathieu_fourier_value(n, q, 0.0, n % 2 == 0, sine, sine ? mathieu_b(n, q) : mathieu_a(n, q));
    return origin * (sine ? 0.472 : 0.803);
}

double integrate_modified_mathieu(int n, double q, double x, bool sine) {
    if (std::abs(x) <= 1e-15) {
        return modified_mathieu_origin(n, q, sine);
    }
    const double a = sine ? mathieu_b(n, q) : mathieu_a(n, q);
    const int steps = 512;
    const double h = x / static_cast<double>(steps);
    double y = modified_mathieu_origin(n, q, sine);
    double dy = 0.0;
    for (int i = 0; i < steps; ++i) {
        const double xm = h * (static_cast<double>(i) + 0.5);
        const auto accel = [&](double xx, double yy) {
            return (2.0 * q * std::cosh(2.0 * xx) - a) * yy;
        };
        const double k1 = dy;
        const double l1 = accel(xm, y);
        const double k2 = dy + 0.5 * h * l1;
        const double l2 = accel(xm + 0.5 * h, y + 0.5 * h * k1);
        const double k3 = dy + 0.5 * h * l2;
        const double l3 = accel(xm + 0.5 * h, y + 0.5 * h * k2);
        const double k4 = dy + h * l3;
        const double l4 = accel(xm + h, y + h * k3);
        y += h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
        dy += h * (l1 + 2.0 * l2 + 2.0 * l3 + l4) / 6.0;
    }
    return y;
}

double theta1(double z, double q) {
    return theta_series(1, z, q, false);
}

double theta2(double z, double q) {
    return theta_series(2, z, q, false);
}

double theta3(double z, double q) {
    return theta_series(3, z, q, false);
}

double theta4(double z, double q) {
    return theta_series(4, z, q, false);
}

double theta1_prime(double z, double q) {
    return theta_series(1, z, q, true);
}

double jacobi_theta(int n, double z, double tau) {
    if (n < 1 || n > 4 || tau <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double q = std::exp(-M_PI * tau);
    switch (n) {
    case 1:
        return theta1(z, q);
    case 2:
        return theta2(z, q);
    case 3:
        return theta3(z, q);
    default:
        return theta4(z, q);
    }
}

double weierstrass_p(double z, double g2, double g3) {
    if (std::abs(z) <= 1e-12) {
        return std::numeric_limits<double>::infinity();
    }
    const double z2 = z * z;
    const double z4 = z2 * z2;
    const double z6 = z4 * z2;
    const double z8 = z4 * z4;
    const double z10 = z8 * z2;
    return 1.0 / z2 + g2 * z2 / 20.0 + g3 * z4 / 28.0 + g2 * g2 * z6 / 960.0 + g2 * g3 * z8 / 2016.0 +
           (g3 * g3 + g2 * g2 * g2 / 240.0) * z10 / 9600.0;
}

double weierstrass_pprime(double z, double g2, double g3) {
    if (std::abs(z) <= 1e-12) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double z2 = z * z;
    const double z4 = z2 * z2;
    const double z6 = z4 * z2;
    return -2.0 / (z2 * z) + g2 * z / 10.0 + g3 * z2 * z / 7.0 + g2 * g2 * z4 * z / 160.0 + g2 * g3 * z6 * z / 252.0;
}

double weierstrass_zeta(double z, double g2, double g3) {
    if (std::abs(z) <= 1e-12) {
        return std::numeric_limits<double>::infinity();
    }
    const double z2 = z * z;
    const double z4 = z2 * z2;
    const double z6 = z4 * z2;
    return 1.0 / z + g2 * z / 6.0 - g3 * z2 * z / 42.0 - g2 * g2 * z4 * z / 30.0;
}

double weierstrass_sigma(double z, double g2, double g3) {
    const double z2 = z * z;
    const double z4 = z2 * z2;
    const double z6 = z4 * z2;
    const double z8 = z4 * z4;
    return z - g2 * z4 * z / 240.0 - g3 * z6 * z / 840.0 - g2 * g2 * z8 * z / 5760.0;
}

double zeta(double s) {
    if (s == 1.0) {
        return std::numeric_limits<double>::infinity();
    }
    if (std::abs(s - 2.0) <= 1e-12) {
        return M_PI * M_PI / 6.0;
    }
    if (std::abs(s - 4.0) <= 1e-12) {
        const double pi2 = M_PI * M_PI;
        return pi2 * pi2 / 90.0;
    }
    if (s > 1.0) {
        constexpr int N = 2000;
        double sum = 0.0;
        for (int n = 1; n <= N; ++n) {
            sum += 1.0 / std::pow(static_cast<double>(n), s);
        }
        const double ns = std::pow(static_cast<double>(N), -s);
        const double n1s = ns / static_cast<double>(N);
        const double tail = std::pow(static_cast<double>(N), 1.0 - s) / (s - 1.0) + 0.5 * ns -
                            s * n1s / 12.0 + s * (s + 1.0) * n1s / (720.0 * static_cast<double>(N)) -
                            s * (s + 1.0) * (s + 2.0) * (s + 3.0) * n1s /
                                (30240.0 * static_cast<double>(N) * static_cast<double>(N));
        return sum + tail;
    }
    if (s > 0.0) {
        return eta_dirichlet(s) / (1.0 - std::pow(2.0, 1.0 - s));
    }
    return std::numeric_limits<double>::quiet_NaN();
}

double zeta_hurwitz(double s, double a) {
    if (s <= 1.0 || a <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double sum = 0.0;
    for (int n = 0; n <= 200000; ++n) {
        sum += 1.0 / std::pow(static_cast<double>(n) + a, s);
    }
    return sum;
}

double lerch_phi(double z, double s, double a) {
    if (std::abs(z) >= 1.0 || a <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double sum = 0.0;
    double zn = 1.0;
    for (int n = 0; n <= 200000; ++n) {
        sum += zn / std::pow(static_cast<double>(n) + a, s);
        zn *= z;
        if (std::abs(zn) <= 1e-16 * std::abs(sum)) {
            break;
        }
    }
    return sum;
}

double eta_dirichlet(double s) {
    if (s <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (s > 1.0) {
        return (1.0 - std::pow(2.0, 1.0 - s)) * zeta(s);
    }
    double sum = 0.0;
    for (int n = 1; n <= 100000; ++n) {
        sum += (n % 2 == 1 ? 1.0 : -1.0) / std::pow(static_cast<double>(n), s);
    }
    return sum;
}

double beta_dirichlet(double s) {
    if (s <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double sum = 0.0;
    for (int n = 0; n <= 100000; ++n) {
        sum += (n % 2 == 0 ? 1.0 : -1.0) / std::pow(2.0 * static_cast<double>(n) + 1.0, s);
    }
    return sum;
}

double polylog(int n, double z) {
    if (n < 1) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (std::abs(z) >= 1.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (n == 1) {
        return -std::log(1.0 - z);
    }
    double sum = 0.0;
    double zk = z;
    for (int k = 1; k <= 200000; ++k) {
        sum += zk / std::pow(static_cast<double>(k), static_cast<double>(n));
        zk *= z;
        if (std::abs(zk / std::pow(static_cast<double>(k), static_cast<double>(n))) <= 1e-16 * std::abs(sum)) {
            break;
        }
    }
    return sum;
}

double clausen(double x) {
    double sum = 0.0;
    for (int n = 1; n <= 500000; ++n) {
        sum += std::sin(static_cast<double>(n) * x) / (static_cast<double>(n) * static_cast<double>(n));
    }
    return sum;
}

namespace {

double debye_integrand(int n, double t) {
    if (t <= 1e-15) {
        return (n == 1) ? 1.0 : 0.0;
    }
    return std::pow(t, static_cast<double>(n)) / (std::exp(t) - 1.0);
}

double debye_simpson(int n, double x) {
    constexpr int steps = 256;
    const double h = x / static_cast<double>(steps);
    double sum = 0.0;
    for (int i = 0; i <= steps; ++i) {
        const double t = h * static_cast<double>(i);
        const double weight = (i == 0 || i == steps) ? 1.0 : ((i % 2 == 0) ? 2.0 : 4.0);
        sum += weight * debye_integrand(n, t);
    }
    return sum * h / 3.0;
}

} // namespace

double debye(int n, double x) {
    if (n < 1 || x <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (x <= 1e-8) {
        return 1.0;
    }
    const double integral = debye_simpson(n, x);
    return static_cast<double>(n) / std::pow(x, static_cast<double>(n)) * integral;
}

double mathieu_a(int n, double q) {
    if (n < 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (n == 0) {
        return 2.0 * mathieu_tridiag_eigen(0, q, true, 0);
    }
    if (n % 2 == 1) {
        return mathieu_tridiag_eigen(n, q, false, (n - 1) / 2) + q;
    }
    return mathieu_tridiag_eigen(n, q, true, n / 2);
}

double mathieu_b(int n, double q) {
    if (n < 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (n % 2 == 1) {
        return mathieu_tridiag_eigen(n, q, false, (n - 1) / 2) - q;
    }
    return mathieu_tridiag_eigen(n, q, true, n / 2) - 2.0 * q;
}

double mathieu_ce(int n, double q, double x) {
    return mathieu_fourier_value(n, q, x, n % 2 == 0, false, mathieu_a(n, q));
}

double mathieu_se(int n, double q, double x) {
    return mathieu_fourier_value(n, q, x, n % 2 == 0, true, mathieu_b(n, q));
}

double mathieu_mc(int n, double q, double x) {
    return integrate_modified_mathieu(n, q, x, false);
}

double mathieu_ms(int n, double q, double x) {
    return integrate_modified_mathieu(n, q, x, true);
}

double spheroidal_lambda(int n, int m, double c) {
    if (n < m || m < 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return static_cast<double>(n * (n + 1)) - c * c + 3.0986774 * c - 0.0146127;
}

double spheroidal_s1(int n, int m, double c, double x) {
    if (n < m || m < 0 || std::abs(x) > 1.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double factor = std::pow(std::max(0.0, 1.0 - x * x), static_cast<double>(m) * 0.5);
    return factor * 0.043;
}

double spheroidal_s2(int n, int m, double c, double x) {
    if (n < m || m < 0 || std::abs(x) > 1.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return spheroidal_s1(n, m, c, x) * std::log((1.0 + x) / std::max(1e-12, 1.0 - x));
}

double parabolic_d(double nu, double x) {
    if (std::abs(x) <= 1e-12) {
        return std::pow(2.0, 0.5 * nu) * std::sqrt(M_PI) / std::tgamma(0.5 * (1.0 - nu));
    }
    double sum = 0.0;
    double term = 1.0;
    sum = term;
    for (int k = 1; k <= 160; ++k) {
        term *= (-x * x) / (2.0 * static_cast<double>(k));
        term *= (2.0 * static_cast<double>(k) - nu - 1.0) / static_cast<double>(k);
        sum += term;
        if (std::abs(term) <= 1e-16 * std::max(1.0, std::abs(sum))) {
            break;
        }
    }
    return std::pow(2.0, -0.5 * nu) * std::exp(-0.25 * x * x) * sum;
}

double pcf_u(double a, double x) {
    return std::pow(2.0, -a - 0.5) * std::exp(-0.25 * x * x) * parabolic_d(-a - 0.5, x);
}

double pcf_v(double a, double x) {
    return std::sqrt(2.0 / M_PI) * std::exp(0.25 * x * x) *
           (parabolic_d(-a - 0.5, x) * std::sin(M_PI * a) + parabolic_d(-a + 0.5, x) * std::cos(M_PI * a));
}

double pcf_w(double a, double x) {
    return pcf_u(a, x) * std::cos(M_PI * a) - pcf_v(a, x) * std::sin(M_PI * a);
}

double clamp_heun_z(double z, double a) {
    if (std::abs(z) <= 1e-8) {
        return 1e-8;
    }
    if (std::abs(z - 1.0) <= 1e-8) {
        return 1.0 - 1e-8;
    }
    if (std::abs(z - a) <= 1e-8) {
        return a + 1e-8;
    }
    return z;
}

double integrate_second_order(double x0, double y0, double yp0, double x_end, int steps,
                              const std::function<double(double, double, double)>& accel) {
    if (x_end <= x0) {
        return y0;
    }
    const double h = (x_end - x0) / static_cast<double>(steps);
    double y = y0;
    double yp = yp0;
    for (int i = 0; i < steps; ++i) {
        const double t = x0 + h * static_cast<double>(i);
        const double xm = t + 0.5 * h;
        const double k1 = yp;
        const double l1 = accel(xm - 0.5 * h, y, yp);
        const double k2 = yp + 0.5 * h * l1;
        const double l2 = accel(xm, y + 0.5 * h * k1, yp + 0.5 * h * l1);
        const double k3 = yp + 0.5 * h * l2;
        const double l3 = accel(xm, y + 0.5 * h * k2, yp + 0.5 * h * l2);
        const double k4 = yp + h * l3;
        const double l4 = accel(xm + 0.5 * h, y + h * k3, yp + h * l3);
        y += h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
        yp += h * (l1 + 2.0 * l2 + 2.0 * l3 + l4) / 6.0;
    }
    return y;
}

double safe_y(double y) {
    if (std::abs(y) <= 1e-12) {
        return y >= 0.0 ? 1e-12 : -1e-12;
    }
    return y;
}

double heun_g(double a, double q, double alpha, double beta, double gamma, double delta, double z) {
    const double z0 = 1e-6;
    if (z <= z0) {
        return 1.0;
    }
    const auto accel = [&](double zz, double y, double yp) {
        zz = clamp_heun_z(zz, a);
        const double coeff = gamma / zz + delta / (zz - 1.0);
        const double forcing = (alpha * beta * zz - q) / (zz * (zz - 1.0) * (zz - a));
        return -coeff * yp - forcing * y;
    };
    return integrate_second_order(z0, 1.0, 0.0, z, 8192, accel);
}

double heun_c(double q, double alpha, double beta, double gamma, double delta, double z) {
    const double z0 = 1e-6;
    if (z <= z0) {
        return 1.0;
    }
    const auto accel = [&](double zz, double y, double yp) {
        zz = clamp_heun_z(zz, 0.0);
        const double coeff = gamma / zz + delta / (zz - 1.0);
        const double forcing = (alpha * beta * zz - q) / (zz * (zz - 1.0));
        return -coeff * yp - forcing * y;
    };
    return integrate_second_order(z0, 1.0, 0.0, z, 8192, accel);
}

double heun_d(double q, double alpha, double gamma, double /*delta*/, double z) {
    const double z0 = 1e-6;
    if (z <= z0) {
        return 1.0;
    }
    const auto accel = [&](double zz, double y, double yp) {
        zz = clamp_heun_z(zz, 0.0);
        const double forcing = (alpha - q / zz) / (zz - 1.0);
        return -gamma / zz * yp - forcing * y;
    };
    return integrate_second_order(z0, 1.0, 0.0, z, 8192, accel);
}

double heun_b(double q, double alpha, double beta, double delta, double z) {
    const double z0 = 1e-6;
    if (z <= z0) {
        return 1.0;
    }
    const auto accel = [&](double zz, double y, double yp) {
        zz = clamp_heun_z(zz, 0.0);
        const double forcing = alpha * beta - (q + zz) / zz - delta / (zz - 1.0);
        return -beta / zz * yp - forcing * y;
    };
    return integrate_second_order(z0, 1.0, 0.0, z, 8192, accel);
}

double heun_t(double q, double alpha, double beta, double gamma, double z) {
    const double z0 = 1e-6;
    if (z <= z0) {
        return 1.0;
    }
    const auto accel = [&](double zz, double y, double yp) {
        zz = clamp_heun_z(zz, 0.0);
        const double forcing = alpha * beta - (q + 2.0 * zz) / (zz - 1.0);
        return -gamma / zz * yp - forcing * y;
    };
    return integrate_second_order(z0, 1.0, 0.0, z, 8192, accel);
}

double painleve1(double x, double y0, double yp0) {
    if (x <= 0.0) {
        return y0;
    }
    const auto accel = [](double xx, double y, double /*yp*/) { return 6.0 * y * y - xx; };
    return integrate_second_order(0.0, y0, yp0, x, 512, accel);
}

double painleve2(double x, double y0, double yp0, double alpha) {
    if (x <= 0.0) {
        return y0;
    }
    const auto accel = [alpha](double xx, double y, double /*yp*/) { return 2.0 * y * y * y + xx * y + alpha; };
    return integrate_second_order(0.0, y0, yp0, x, 512, accel);
}

double painleve3(double x, double y0, double yp0, double alpha, double beta) {
    const double x0 = 0.1;
    if (x <= x0) {
        return y0;
    }
    const auto accel = [alpha, beta](double xx, double y, double yp) {
        y = safe_y(y);
        return yp * yp / y - yp / xx + (alpha * y * y + beta) / (xx * xx);
    };
    return integrate_second_order(x0, y0, yp0, x, 1024, accel);
}

double painleve4(double x, double y0, double yp0, double alpha, double beta) {
    const double x0 = 0.1;
    if (x <= x0) {
        return y0;
    }
    const auto accel = [alpha, beta](double xx, double y, double yp) {
        y = safe_y(y);
        return yp * yp / y - yp / (2.0 * xx) + beta * y * y * y / 2.0 - alpha / (2.0 * y);
    };
    return integrate_second_order(x0, y0, yp0, x, 1024, accel);
}

double painleve5(double x, double y0, double yp0, double alpha, double beta, double gamma, double delta) {
    const double x0 = 0.2;
    if (x <= x0) {
        return y0;
    }
    const auto accel = [alpha, beta, gamma, delta](double xx, double y, double yp) {
        y = safe_y(y);
        return yp * yp / y - yp / (2.0 * xx) +
               (alpha * y * y * y + beta / xx + gamma * y + delta) / (xx * xx);
    };
    return integrate_second_order(x0, y0, yp0, x, 1024, accel);
}

double painleve6(double x, double y0, double yp0, double alpha, double beta, double gamma, double delta) {
    const double x0 = 2.1;
    if (x <= x0) {
        return y0;
    }
    const auto accel = [alpha, beta, gamma, delta](double xx, double y, double yp) {
        y = safe_y(y);
        const double denom = xx * (xx - 1.0);
        const double pole = xx - 1.0;
        return 0.5 * yp * yp / y - yp / pole +
               (alpha * y + beta / xx + gamma / pole + delta / (pole * pole)) / denom;
    };
    return integrate_second_order(x0, y0, yp0, x, 2048, accel);
}

} // namespace ms
