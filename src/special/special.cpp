#include "ms/special/special.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>
#include <limits>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ms {

namespace {

constexpr double kEulGamma = 0.577215664901532860606512090082402431;

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

// Threshold above which the ascending Bessel series loses too many digits to cancellation and
// the Hankel (large-argument) asymptotic expansion is used instead.  The series error grows
// like e^{x} eps and the optimally truncated asymptotic error falls like e^{-2x}, so the two
// curves cross near x = 12-13 at about 4e-11 -- the worst relative error of either branch.
constexpr double kBesselAsymptoticX = 13.0;

// Hankel asymptotic expansion (DLMF 10.17.3-10.17.4) for J_nu(x) and Y_nu(x), x large:
//   J_nu(x) = sqrt(2/(pi x)) [P(nu,x) cos(chi) - Q(nu,x) sin(chi)]
//   Y_nu(x) = sqrt(2/(pi x)) [P(nu,x) sin(chi) + Q(nu,x) cos(chi)],  chi = x - (nu/2 + 1/4) pi
// with P collecting the even and Q the odd terms of t_k = t_{k-1} (mu - (2k-1)^2)/(8 x k),
// mu = 4 nu^2.  The series is divergent, so it is truncated at its smallest term.
struct BesselHankel {
    double j;
    double y;
};

BesselHankel bessel_hankel(double nu, double x) {
    const double mu = 4.0 * nu * nu;
    double term = 1.0;
    double p = 1.0;
    double q = 0.0;
    double previous = std::numeric_limits<double>::infinity();
    for (int k = 1; k < 64; ++k) {
        const double odd = 2.0 * static_cast<double>(k) - 1.0;
        term *= (mu - odd * odd) / (8.0 * x * static_cast<double>(k));
        const double magnitude = std::abs(term);
        if (magnitude >= previous || !std::isfinite(magnitude)) {
            break;  // optimal truncation: the expansion has started to diverge
        }
        previous = magnitude;
        const double sign = ((k / 2) % 2 == 0) ? 1.0 : -1.0;
        if (k % 2 == 0) {
            p += sign * term;
        } else {
            q += sign * term;
        }
        if (magnitude <= 1e-18) {
            break;
        }
    }
    const double chi = x - (0.5 * nu + 0.25) * M_PI;
    const double amp = std::sqrt(2.0 / (M_PI * x));
    const double c = std::cos(chi);
    const double s = std::sin(chi);
    return {amp * (p * c - q * s), amp * (p * s + q * c)};
}

double bessel_j_general(double nu, double x) {
    if (x < 0.0) {
        const int parity = static_cast<int>(std::llround(nu)) % 2;
        return (parity == 0) ? bessel_j_general(nu, -x) : -bessel_j_general(nu, -x);
    }
    if (x == 0.0) {
        return nu == 0.0 ? 1.0 : 0.0;
    }
    if (x >= kBesselAsymptoticX && x >= 2.0 * std::abs(nu)) {
        return bessel_hankel(nu, x).j;
    }
    const double half = 0.5 * x;
    double term = std::pow(half, nu) / std::tgamma(nu + 1.0);
    double sum = term;
    for (int k = 1; k < 400; ++k) {
        term *= -(half * half) / (static_cast<double>(k) * (nu + static_cast<double>(k)));
        sum += term;
        if (std::abs(term) <= 1e-17 * std::max(1.0, std::abs(sum))) {
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
    for (int k = 1; k < 900; ++k) {
        term *= (half * half) / (static_cast<double>(k) * (nu + static_cast<double>(k)));
        sum += term;
        if (std::abs(term) <= 1e-17 * std::max(1.0, std::abs(sum))) {
            break;
        }
    }
    return sum;
}

// K_nu(x) from the integral representation K_nu(x) = int_0^inf e^{-x cosh t} cosh(nu t) dt
// (DLMF 10.32.9) evaluated with the infinite trapezoidal rule.  The integrand extends to an
// even, analytic function of t on the strip |Im t| < pi/2, so the trapezoid error decays like
// exp(-pi^2/h) -- about 7e-18 at h = 1/4 -- uniformly in x.  This is used instead of the
// ascending log-series (which loses ~e^{2x} digits to cancellation: at x = 15, I_0(x) ~ 1.9e5
// while K_0(x) ~ 1.1e-7) and instead of the divergent large-x expansion, both of which are only
// accurate at the opposite ends of the range.
double modified_k_quadrature(double nu, double x) {
    // The trapezoid error is ~exp(-pi^2/h) in absolute terms while K_nu(x) itself is ~e^{-x},
    // so the step is tightened with x to keep the RELATIVE error near machine epsilon.
    const double h = std::min(0.25, M_PI * M_PI / (x + 45.0));
    const double a = std::abs(nu);
    double sum = 0.5 * std::exp(-x);
    for (int i = 1; i < 1000; ++i) {
        const double t = h * static_cast<double>(i);
        const double arg = -x * std::cosh(t);
        if (arg + a * t < -740.0) {
            break;
        }
        sum += 0.5 * (std::exp(arg + a * t) + std::exp(arg - a * t));
    }
    return sum * h;
}

// K_nu(x) for arbitrary real order.  The quadrature above is used directly for |nu| < 5; its
// cosh(nu t) factor makes the trapezoid lose accuracy at higher orders (measured: ~6e-8 at
// nu = 20), so beyond that the stable upward recurrence of DLMF 10.29.1,
// K_{nu+1}(x) = K_{nu-1}(x) + (2 nu / x) K_nu(x), carries it up from a quadrature seed.
double modified_k_order(double nu, double x) {
    const double a = std::abs(nu);
    if (a < 5.0) {
        return modified_k_quadrature(a, x);
    }
    const int steps = static_cast<int>(std::floor(a - 4.0));
    const double nu0 = a - static_cast<double>(steps);
    double km1 = modified_k_quadrature(nu0 - 1.0, x);
    double k = modified_k_quadrature(nu0, x);
    for (int i = 0; i < steps; ++i) {
        const double kp1 = km1 + (2.0 * (nu0 + static_cast<double>(i)) / x) * k;
        km1 = k;
        k = kp1;
    }
    return k;
}

// ber_nu(x) + i bei_nu(x) = J_nu(x e^{3 pi i / 4}) (DLMF 10.61.1), evaluated from the ascending
// series DLMF 10.65.1:
//   (x/2)^nu sum_k [cos + i sin]((3nu/4 + k/2) pi) (x^2/4)^k / (k! Gamma(nu+k+1)).
// Valid for every real order nu >= 0.  The terms have the magnitudes of the I_nu series while
// the sum has the magnitude of |J_nu(x e^{3 i pi/4})| ~ e^{x/sqrt 2}, so about 0.29 x / ln(10)
// significant digits are lost to cancellation: ~1e-13 at x = 10, ~1e-10 at x = 30.
std::complex<double> kelvin_ber_bei(double nu, double x) {
    if (x == 0.0) {
        return (nu == 0.0) ? std::complex<double>(1.0, 0.0) : std::complex<double>(0.0, 0.0);
    }
    const double half = 0.5 * std::abs(x);
    double coefficient = std::pow(half, nu) / std::tgamma(nu + 1.0);
    std::complex<double> sum(0.0, 0.0);
    double largest = 0.0;
    for (int k = 0; k < 400; ++k) {
        if (k > 0) {
            coefficient *= (half * half) / (static_cast<double>(k) * (nu + static_cast<double>(k)));
        }
        const double angle = (0.75 * nu + 0.5 * static_cast<double>(k)) * M_PI;
        sum += coefficient * std::complex<double>(std::cos(angle), std::sin(angle));
        largest = std::max(largest, coefficient);
        if (k > 2 && coefficient <= 1e-18 * largest) {
            break;
        }
    }
    return sum;
}

// ker_nu(x) + i kei_nu(x) = e^{-nu pi i / 2} K_nu(x e^{i pi / 4}) (DLMF 10.61.5).  The complex
// K is taken straight from the same integral representation used on the real axis,
// K_nu(z) = int_0^inf e^{-z cosh t} cosh(nu t) dt, which is valid for Re z > 0 -- and
// Re(x e^{i pi/4}) = x/sqrt(2) > 0 -- with the same exponentially convergent trapezoid rule.
std::complex<double> kelvin_ker_kei(double nu, double x) {
    const std::complex<double> z(x / std::sqrt(2.0), x / std::sqrt(2.0));
    const double a = std::abs(nu);
    // For complex z the integrand only decays inside |Im t| < arctan(Re z / Im z) = pi/4 here,
    // half the real-axis strip, so the step is halved twice over; at h = 1/16 the trapezoid
    // error exp(-pi^2/(2h)) ~ e^{-79} is far below the e^{-Re z} size of K itself.
    constexpr double h = 0.0625;
    std::complex<double> sum = 0.5 * std::exp(-z);
    for (int i = 1; i < 2000; ++i) {
        const double t = h * static_cast<double>(i);
        const std::complex<double> arg = -z * std::cosh(t);
        if (arg.real() + a * t < -740.0) {
            break;
        }
        sum += 0.5 * (std::exp(arg + a * t) + std::exp(arg - a * t));
    }
    sum *= h;
    const double phase = -0.5 * nu * M_PI;
    return sum * std::complex<double>(std::cos(phase), std::sin(phase));
}

double bessel_y_general(double nu, double x) {
    if (x <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (x >= kBesselAsymptoticX && x >= 2.0 * std::abs(nu)) {
        return bessel_hankel(nu, x).y;
    }
    constexpr double eps = 1e-8;
    const double n = nu + eps;
    return (bessel_j_general(n, x) * std::cos(n * M_PI) - bessel_j_general(-n, x)) / std::sin(n * M_PI);
}

double bessel_k_general(double nu, double x) {
    if (x <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return modified_k_order(nu, x);
}

double bessel_j0_impl(double x) {
    return bessel_j_general(0.0, x);
}

double bessel_j1_impl(double x) {
    return bessel_j_general(1.0, x);
}

// Miller's backward recurrence for J_n(x) in the region n > x, where the upward recurrence
// J_{n+1} = (2n/x) J_n - J_{n-1} is exponentially unstable (J_n is the minimal solution there).
// Seeded far above the turning point with an arbitrary tiny value and normalised by the
// identity J_0(x) + 2 sum_{k>=1} J_{2k}(x) = 1 (DLMF 10.12.4 at theta = 0).
double bessel_j_miller(int nu, double x) {
    const double ax = std::abs(x);
    const int start = 2 * ((nu + static_cast<int>(std::sqrt(40.0 * static_cast<double>(nu))) +
                            static_cast<int>(ax) + 4) / 2 + 1);
    double fp1 = 0.0;
    double f = 1e-280;
    double norm = 0.0;
    double answer = 0.0;
    for (int k = start; k >= 1; --k) {
        const double fm1 = (2.0 * static_cast<double>(k) / ax) * f - fp1;
        fp1 = f;
        f = fm1;
        if (std::abs(f) > 1e250) {
            f *= 1e-250;
            fp1 *= 1e-250;
            norm *= 1e-250;
            answer *= 1e-250;
        }
        if ((k - 1) % 2 == 0 && k - 1 > 0) {
            norm += 2.0 * f;
        }
        if (k - 1 == nu) {
            answer = f;
        }
    }
    norm += f;  // f now holds the (unnormalised) J_0
    const double value = answer / norm;
    return (x < 0.0 && nu % 2 == 1) ? -value : value;
}

double bessel_jn(int nu, double x) {
    if (nu == 0) {
        return bessel_j0_impl(x);
    }
    if (nu == 1) {
        return bessel_j1_impl(x);
    }
    if (x == 0.0) {
        return 0.0;
    }
    const double ax = std::abs(x);
    if (ax >= static_cast<double>(nu)) {
        // Below the turning point the upward recurrence is the stable direction.
        double j0 = bessel_j0_impl(ax);
        double j1 = bessel_j1_impl(ax);
        for (int n = 1; n < nu; ++n) {
            const double jn = (2.0 * static_cast<double>(n) / ax) * j1 - j0;
            j0 = j1;
            j1 = jn;
        }
        return (x < 0.0 && nu % 2 == 1) ? -j1 : j1;
    }
    if (ax * ax <= 4.0 * (static_cast<double>(nu) + 1.0)) {
        // Ascending series with monotonically decreasing terms: no cancellation at all.
        return bessel_j_series(nu, x);
    }
    return bessel_j_miller(nu, x);
}

double bessel_y0_impl(double x) {
    if (x >= kBesselAsymptoticX) {
        return bessel_hankel(0.0, x).y;
    }
    const double j0 = bessel_j0_impl(x);
    double y0 = (2.0 / M_PI) * (std::log(0.5 * x) + kEulGamma) * j0;
    const double half = 0.5 * x;
    double fact = 1.0;
    double sum = 0.0;
    double sign = 1.0;
    for (int k = 1; k < 120; ++k) {
        fact *= half / static_cast<double>(k);
        sum += sign * fact * fact * harmonic(k);
        sign = -sign;
        if (fact * fact <= 1e-16) {
            break;
        }
    }
    y0 += (2.0 / M_PI) * sum;
    return y0;
}

// Ascending series for Y_1 (A&S 9.1.11 at n = 1):
//   Y_1(x) = -2/(pi x) + (2/pi) ln(x/2) J_1(x)
//            - (x/(2 pi)) sum_k [psi(k+1)+psi(k+2)] (-x^2/4)^k / (k! (k+1)!)
// with psi(1) = -gamma, psi(k+1) = -gamma + H_k.  Replaces the previous route through
// bessel_y_general's finite-difference-in-order trick, which only reached ~1e-8.
double bessel_y1_impl(double x) {
    if (x >= kBesselAsymptoticX) {
        return bessel_hankel(1.0, x).y;
    }
    const double half = 0.5 * x;
    const double quarter = half * half;
    double power = 1.0;       // (-x^2/4)^k
    double factorials = 1.0;  // 1 / (k! (k+1)!)
    double sum = 0.0;
    for (int k = 0; k < 300; ++k) {
        if (k > 0) {
            power *= -quarter;
            factorials /= static_cast<double>(k) * static_cast<double>(k + 1);
        }
        const double psi_sum = -2.0 * kEulGamma + harmonic(k) + harmonic(k + 1);
        const double term = psi_sum * power * factorials;
        sum += term;
        if (k > 2 && std::abs(term) <= 1e-18 * std::max(1.0, std::abs(sum))) {
            break;
        }
    }
    return -2.0 / (M_PI * x) + (2.0 / M_PI) * std::log(half) * bessel_j_general(1.0, x) -
           half * sum / M_PI;
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
    for (int k = 1; k < 400; ++k) {
        term *= (half * half) / (static_cast<double>(k) * static_cast<double>(nu + k));
        sum += term;
        if (std::abs(term) <= 1e-17 * std::abs(sum)) {
            break;
        }
    }
    return sum;
}

// I_nu(x) for every integer order.  The upward recurrence I_{n+1} = I_{n-1} - (2n/x) I_n is
// the UNSTABLE direction (I_n is the minimal solution: it decays with n), so it is not used at
// all; the ascending series has strictly positive terms and therefore suffers no cancellation
// whatsoever, which makes it both stable and accurate for every order.
double modified_in(int nu, double x) {
    return modified_i_series(nu, x);
}

// K_nu(x) for every integer order.  DLMF 10.29.1 gives the upward recurrence
// K_{n+1}(x) = K_{n-1}(x) + (2n/x) K_n(x), which IS the stable direction for K (K_n grows with
// n, unlike I_n); it is used only when the order is high enough that the direct quadrature's
// cosh(n t) factor would need an inconveniently long range, and seeded from the quadrature.
double modified_kn(int nu, double x) {
    return modified_k_order(static_cast<double>(nu), x);
}

// Ascending series shared by the Struve H and the modified Struve L (DLMF 11.2.1 / 11.2.2):
//   H_nu(x) = sum_m (-1)^m (x/2)^{2m+nu+1} / (Gamma(m+3/2) Gamma(m+nu+3/2))
//   L_nu(x) = sum_m        (x/2)^{2m+nu+1} / (Gamma(m+3/2) Gamma(m+nu+3/2))
// -- the SAME series, differing only in the alternating sign.  `alternating` selects which.
double struve_series(int nu, double x, bool alternating) {
    if (x == 0.0) {
        return 0.0;
    }
    const double half = 0.5 * x;
    double term = std::pow(half, static_cast<double>(nu) + 1.0) /
                  (std::tgamma(1.5) * std::tgamma(static_cast<double>(nu) + 1.5));
    double sum = term;
    double largest = std::abs(term);
    for (int m = 1; m < 400; ++m) {
        const double md = static_cast<double>(m);
        term *= (alternating ? -1.0 : 1.0) * half * half /
                ((md + 0.5) * (md + static_cast<double>(nu) + 0.5));
        sum += term;
        largest = std::max(largest, std::abs(term));
        if (std::abs(term) <= 1e-18 * largest) {
            break;
        }
    }
    return sum;
}

// Large-argument expansion of H_nu - Y_nu (DLMF 11.6.1), which is exactly the Struve K:
//   H_nu(z) - Y_nu(z) ~ (1/pi) sum_k Gamma(k+1/2) (z/2)^{nu-2k-1} / Gamma(nu+1/2-k)
// truncated at its smallest term.  The ascending series above loses ~e^{x} digits for H at
// large x (its terms grow like I_nu while H_nu = O(1)), so this branch takes over there.
double struve_h_minus_y_asymptotic(int nu, double x) {
    const double half = 0.5 * x;
    const double nud = static_cast<double>(nu);
    double sum = 0.0;
    double previous = std::numeric_limits<double>::infinity();
    for (int k = 0; k < 64; ++k) {
        const double kd = static_cast<double>(k);
        const double denom = std::tgamma(nud + 0.5 - kd);
        double term = std::tgamma(kd + 0.5) * std::pow(half, nud - 2.0 * kd - 1.0);
        term = std::isfinite(denom) ? term / denom : 0.0;
        if (std::abs(term) >= previous) {
            break;
        }
        previous = std::abs(term);
        sum += term;
    }
    return sum / M_PI;
}

double struve_h_series(int nu, double x) {
    if (std::abs(x) > 20.0) {
        return bessel_y_general(static_cast<double>(nu), std::abs(x)) +
               struve_h_minus_y_asymptotic(nu, std::abs(x));
    }
    return struve_series(nu, x, true);
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
    // Y_{-1} = -Y_1 (DLMF 10.4.1), so the nu = 0 case folds into the same relation.
    const double lower = (nu == 0) ? -bessel_yn(1, x) : bessel_yn(nu - 1, x);
    return 0.5 * (lower - bessel_yn(nu + 1, x));
}

double bessel_zero_newton_y(int nu, double guess) {
    for (int iter = 0; iter < 40; ++iter) {
        const double value = bessel_yn(nu, guess);
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

// Locate the n-th sign change of fn on (start, inf) by marching with a step strictly smaller
// than the minimum spacing of consecutive Bessel zeros (which exceeds 3 for every nu >= 0),
// so no zero can be skipped and the index is exact.  Returns the bracket [lo, hi].
struct ZeroBracket {
    double lo;
    double hi;
    bool found;
};

ZeroBracket bessel_scan_bracket(int n, double start, double step,
                                const std::function<double(double)>& fn) {
    double lo = start;
    double flo = fn(lo);
    int count = 0;
    for (int i = 1; i <= 200000; ++i) {
        const double hi = start + step * static_cast<double>(i);
        const double fhi = fn(hi);
        if (!std::isfinite(fhi)) {
            return {lo, hi, false};
        }
        if (fhi == 0.0) {
            ++count;
            if (count == n) {
                return {hi, hi, true};
            }
        } else if (flo * fhi < 0.0) {
            ++count;
            if (count == n) {
                return {lo, hi, true};
            }
        }
        lo = hi;
        flo = fhi;
    }
    return {lo, lo, false};
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
    for (int n = 1; n < 600; ++n) {
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

double kummer_u_connection(double a, double b, double z) {
    const double log_c1 = log_gamma(1.0 - b) - log_gamma(a - b + 1.0);
    const double log_c2 = log_gamma(b - 1.0) - log_gamma(a);
    if (!std::isfinite(log_c1) || !std::isfinite(log_c2)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double c1 = std::exp(log_c1);
    const double c2 = std::exp(log_c2);
    const double m1 = kummer_m_series(a, b, z);
    const double m2 = kummer_m_series(a - b + 1.0, 2.0 - b, z);
    return c1 * m1 - c2 * std::pow(z, 1.0 - b) * m2;
}

// Large-z asymptotic expansion U(a,b,z) ~ z^{-a} 2F0(a, a-b+1; ; -1/z) (DLMF 13.7.3), truncated
// at its smallest term.  The previous version multiplied each term by (-z) instead of (-1/z),
// i.e. it evaluated the divergent series at the RECIPROCAL of the intended argument, and was
// therefore exact only in the terminating case b = a+1; elsewhere it returned finite garbage
// (U(1, 0.5, 1) came out as -4.18e272 against a true 0.4843).
double tricomi_u_asymptotic(double a, double b, double z) {
    double sum = 1.0;
    double term = 1.0;
    double previous = std::numeric_limits<double>::infinity();
    for (int n = 1; n < 200; ++n) {
        const double nd = static_cast<double>(n);
        term *= (a + nd - 1.0) * (a - b + nd) * (-1.0 / z) / nd;
        if (!std::isfinite(term) || std::abs(term) >= previous) {
            break;  // optimal truncation of a divergent expansion
        }
        previous = std::abs(term);
        sum += term;
        if (previous <= 1e-18 * std::abs(sum)) {
            break;
        }
    }
    return sum * std::pow(z, -a);
}

// U(a,b,z) = (1/Gamma(a)) int_0^inf e^{-z t} t^{a-1} (1+t)^{b-a-1} dt   (DLMF 13.4.4, a > 0,
// z > 0), evaluated with the double-exponential (tanh-sinh style) substitution
// t = exp((pi/2) sinh u).  The transformed integrand decays doubly exponentially at both ends,
// so the trapezoidal rule converges geometrically; the algebraic endpoint behaviour t^{a-1} is
// absorbed by the substitution.  This branch is uniformly valid -- it needs no restriction on b
// and handles integer b, where the Kummer connection formula degenerates.
double tricomi_u_quadrature(double a, double b, double z) {
    constexpr double h = 0.02;
    const double exponent = b - a - 1.0;
    double sum = 0.0;
    for (int direction = 0; direction < 2; ++direction) {
        for (int i = (direction == 0 ? 0 : 1); i < 6000; ++i) {
            const double u = h * static_cast<double>(direction == 0 ? i : -i);
            const double sh = std::sinh(u);
            const double log_t = 0.5 * M_PI * sh;
            if (log_t > 700.0) {
                break;
            }
            const double t = std::exp(log_t);
            const double log_f = -z * t + a * log_t + exponent * std::log1p(t) +
                                 std::log(0.5 * M_PI * std::cosh(u));
            if (log_f < -745.0) {
                if (i > 4) {
                    break;
                }
                continue;
            }
            sum += std::exp(log_f);
        }
    }
    return sum * h / std::tgamma(a);
}

double tricomi_u_impl(double a, double b, double z) {
    if (z <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    // U(a, a+1, z) = z^{-a} exactly (DLMF 13.6.4).
    if (std::abs(b - a - 1.0) <= 1e-14 * std::max(1.0, std::abs(a))) {
        return std::pow(z, -a);
    }
    if (a > 0.0) {
        if (z > std::max(30.0, 4.0 * (std::abs(a) + std::abs(a - b + 1.0)))) {
            return tricomi_u_asymptotic(a, b, z);
        }
        return tricomi_u_quadrature(a, b, z);
    }
    // a <= 0: shift up into the integral's domain and come back down with DLMF 13.3.7,
    //   U(a-1,b,z) = (2a + z - b) U(a,b,z) - a (a-b+1) U(a+1,b,z).
    const int shift = static_cast<int>(std::ceil(1.0 - a)) + 1;
    double upper = tricomi_u_impl(a + static_cast<double>(shift) + 1.0, b, z);
    double current = tricomi_u_impl(a + static_cast<double>(shift), b, z);
    for (int k = shift; k > 0; --k) {
        const double ak = a + static_cast<double>(k);
        const double lower = (2.0 * ak + z - b) * current - ak * (ak - b + 1.0) * upper;
        upper = current;
        current = lower;
    }
    return current;
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

// Closed form of the Meijer G-function G^{1,1}_{1,1}(z | a ; b).  Summing the residues of its
// Mellin-Barnes integrand Gamma(b+s) Gamma(1-a-s) z^{-s} at the poles s = -b-k gives
//   G^{1,1}_{1,1}(z | a; b) = sum_k (-1)^k/k! Gamma(1-a+b+k) z^{b+k}
//                           = Gamma(1-a+b) z^b (1+z)^{a-b-1}.
// The previous body returned z^a e^{-z} M(b-a,b,z)/Gamma(a), a different function entirely
// (0.393469 rather than 0.111111 at a=1, b=2, z=0.5).
double meijer_g1111(double a, double b, double z) {
    if (z <= 0.0) {
        return 0.0;
    }
    const double order = 1.0 - a + b;
    if (order <= 0.0 && order == std::floor(order)) {
        return std::numeric_limits<double>::quiet_NaN();  // Gamma pole
    }
    return std::tgamma(order) * std::pow(z, b) * std::pow(1.0 + z, a - b - 1.0);
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

double gamma_inc_reg_impl(double a, double x) {
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
        double aa = static_cast<double>(m) * (b - static_cast<double>(m)) * x / ((qam + m2) * (a + m2));
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
        aa = -(a + static_cast<double>(m)) * (qab + static_cast<double>(m)) * x / ((a + m2) * (qap + m2));
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

double beta_inc_reg_impl(double a, double b, double x) {
    if (x <= 0.0) {
        return 0.0;
    }
    if (x >= 1.0) {
        return 1.0;
    }
    const double bt = std::exp(std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) + a * std::log(x) +
                               b * std::log(1.0 - x));
    if (x < (a + 1.0) / (a + b + 2.0)) {
        return bt * beta_continued_fraction(a, b, x) / a;
    }
    return 1.0 - bt * beta_continued_fraction(b, a, 1.0 - x) / b;
}

// Humlicek (1982) w4 rational approximation of the Faddeeva function w(z) = exp(-z^2)
// erfc(-iz), specialized to the (v, a) parametrization standard in Voigt-profile work: given
// real "frequency" v and non-negative "damping" a, this returns w(v + i*a) accurate to ~1e-4
// relative error (region I/II asymptotics) or better (region III/IV rational forms) over the
// whole half-plane a >= 0. Region split follows s = |v| + a exactly as in the original paper.
std::complex<double> humlicek_w4(double v, double a) {
    const std::complex<double> z(a, -v);
    const double s = std::abs(v) + a;
    if (s >= 15.0) {
        return (z * 0.5641896) / (0.5 + z * z);
    }
    if (s >= 5.5) {
        const std::complex<double> u = z * z;
        return (z * (1.410474 + u * 0.5641896)) / (0.75 + u * (3.0 + u));
    }
    if (a >= 0.195 * std::abs(v) - 0.176) {
        return (16.4955 + z * (20.20933 + z * (11.96482 + z * (3.778987 + 0.5642236 * z)))) /
               (16.4955 + z * (38.82363 + z * (39.27121 + z * (21.69274 + z * (6.699398 + z)))));
    }
    const std::complex<double> u = z * z;
    const std::complex<double> numerator =
        z * (36183.31 - u * (3321.9905 - u * (1540.787 - u * (219.0313 - u * (35.76683 -
             u * (1.320522 - u * 0.56419))))));
    const std::complex<double> denominator =
        32066.6 - u * (24322.84 - u * (9022.228 - u * (2186.181 - u * (364.2191 -
                  u * (61.57037 - u * (1.841439 - u))))));
    return std::exp(u) - numerator / denominator;
}

// Thompson-Cox-Hastings shared FWHM for the pseudo-Voigt approximation, combining the
// Gaussian FWHM f_g = 2 sigma sqrt(2 ln 2) and Lorentzian FWHM f_l = 2 gamma.
struct PseudoVoigtFwhm {
    double f_g;
    double f_l;
    double f;
};

PseudoVoigtFwhm pseudo_voigt_fwhm(double sigma, double gamma) {
    const double s = std::max(sigma, 0.0);
    const double g = std::max(gamma, 0.0);
    const double f_g = 2.0 * s * std::sqrt(2.0 * std::log(2.0));
    const double f_l = 2.0 * g;
    const double f5 = std::pow(f_g, 5.0) + 2.69269 * std::pow(f_g, 4.0) * f_l +
                       2.42843 * std::pow(f_g, 3.0) * f_l * f_l + 4.47163 * f_g * f_g * std::pow(f_l, 3.0) +
                       0.07842 * f_g * std::pow(f_l, 4.0) + std::pow(f_l, 5.0);
    return {f_g, f_l, std::pow(f5, 0.2)};
}


// ---------------------------------------------------------------------------------------
// Airy functions.  Two exact representations are combined:
//
//  * |x| < 1: the convergent Maclaurin series (DLMF 9.4.1-9.4.4)
//        Ai(x) = c1 f(x) - c2 g(x),      Bi(x) = sqrt(3) (c1 f(x) + c2 g(x))
//    where f and g solve y'' = x y with f(0)=1, f'(0)=0 and g(0)=0, g'(0)=1, generated from
//    a_{n+3} = a_n/((n+3)(n+2)), and c1 = Ai(0) = 3^{-2/3}/Gamma(2/3),
//    c2 = -Ai'(0) = 3^{-1/3}/Gamma(1/3).  Derivatives come from the same series, differentiated
//    term by term.
//
//  * |x| >= 1: the exact Bessel connection formulas (DLMF 9.6.2-9.6.9) in zeta = (2/3)|x|^{3/2},
//        Ai(x)  = (1/pi) sqrt(x/3) K_{1/3}(zeta),    Ai'(x)  = -(x/(pi sqrt 3)) K_{2/3}(zeta)
//        Bi(x)  = sqrt(x/3) (I_{-1/3} + I_{1/3}),    Bi'(x)  = (x/sqrt 3) (I_{-2/3} + I_{2/3})
//        Ai(-y) = (sqrt y/3)(J_{1/3} + J_{-1/3}),    Ai'(-y) = (y/3)(J_{2/3} - J_{-2/3})
//        Bi(-y) = sqrt(y/3)(J_{-1/3} - J_{1/3}),     Bi'(-y) = (y/sqrt 3)(J_{-2/3} + J_{2/3}),
//    evaluated with the modified-Bessel routines above, which are themselves accurate to
//    ~1e-15 over the whole range (K by exponentially convergent quadrature, J by the ascending
//    series below the turning point and the Hankel expansion above it).
//
// This replaces a least-squares polynomial fit that was quantitatively wrong at the top of its
// own branch (Ai(4) = 0.0769 against a true 0.000952) and qualitatively wrong for every x < 0,
// where Ai and Bi oscillate.  Measured worst-case relative error: ~1e-15 (see the AiryReference
// regression test).
constexpr double kAiryC1 = 0.355028053887817239;   // 3^{-2/3} / Gamma(2/3) = Ai(0)
constexpr double kAiryC2 = 0.258819403792806798;   // 3^{-1/3} / Gamma(1/3) = -Ai'(0)
constexpr double kAiryS3 = 1.732050807568877294;   // sqrt(3)

struct AiryValues {
    double ai;
    double bi;
    double aip;
    double bip;
};

AiryValues airy_series(double x) {
    const double x3 = x * x * x;
    double f = 1.0;
    double g = x;
    double fp = 0.0;
    double gp = 1.0;
    double tf = 1.0;
    double tg = x;
    for (int k = 1; k < 60; ++k) {
        const double kd = static_cast<double>(k);
        tf *= x3 / ((3.0 * kd) * (3.0 * kd - 1.0));
        tg *= x3 / ((3.0 * kd + 1.0) * (3.0 * kd));
        f += tf;
        g += tg;
        if (x != 0.0) {
            fp += 3.0 * kd * tf / x;
            gp += (3.0 * kd + 1.0) * tg / x;
        }
        if (std::abs(tf) + std::abs(tg) <= 1e-19 * (std::abs(f) + std::abs(g))) {
            break;
        }
    }
    return {kAiryC1 * f - kAiryC2 * g,
            kAiryS3 * (kAiryC1 * f + kAiryC2 * g),
            kAiryC1 * fp - kAiryC2 * gp,
            kAiryS3 * (kAiryC1 * fp + kAiryC2 * gp)};
}

AiryValues airy_all(double x) {
    if (std::abs(x) < 1.0) {
        return airy_series(x);
    }
    const double y = std::abs(x);
    const double zeta = (2.0 / 3.0) * y * std::sqrt(y);
    if (x > 0.0) {
        const double k13 = modified_k_order(1.0 / 3.0, zeta);
        const double k23 = modified_k_order(2.0 / 3.0, zeta);
        const double i13 = bessel_i_general(1.0 / 3.0, zeta);
        const double im13 = bessel_i_general(-1.0 / 3.0, zeta);
        const double i23 = bessel_i_general(2.0 / 3.0, zeta);
        const double im23 = bessel_i_general(-2.0 / 3.0, zeta);
        const double root = std::sqrt(y / 3.0);
        return {root * k13 / M_PI,
                root * (im13 + i13),
                -y * k23 / (M_PI * kAiryS3),
                y * (im23 + i23) / kAiryS3};
    }
    const double j13 = bessel_j_general(1.0 / 3.0, zeta);
    const double jm13 = bessel_j_general(-1.0 / 3.0, zeta);
    const double j23 = bessel_j_general(2.0 / 3.0, zeta);
    const double jm23 = bessel_j_general(-2.0 / 3.0, zeta);
    return {std::sqrt(y) * (j13 + jm13) / 3.0,
            std::sqrt(y / 3.0) * (jm13 - j13),
            y * (j23 - jm23) / 3.0,
            y * (jm23 + j23) / kAiryS3};
}

} // namespace

double erf(double x) {
    return std::erf(x);
}

double erfc(double x) {
    return std::erfc(x);
}

double erfinv(double x) {
    if (x < -1.0 || x > 1.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (x == 0.0) {
        return 0.0;
    }
    if (x == 1.0) {
        return std::numeric_limits<double>::infinity();
    }
    if (x == -1.0) {
        return -std::numeric_limits<double>::infinity();
    }
    constexpr double a = 0.147;
    const double sign = x < 0.0 ? -1.0 : 1.0;
    const double x_abs = std::abs(x);
    const double ln1mx2 = std::log(1.0 - x_abs * x_abs);
    const double first = 2.0 / (M_PI * a) + 0.5 * ln1mx2;
    const double second = ln1mx2 / a;
    double w = sign * std::sqrt(std::sqrt(first * first - second) - first);
    for (int i = 0; i < 3; ++i) {
        const double err = erf(w) - x;
        w -= err / (2.0 / std::sqrt(M_PI) * std::exp(-w * w));
    }
    return w;
}

double erfcinv(double x) {
    if (x <= 0.0 || x >= 2.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return erfinv(1.0 - x);
}

// erfi(x) = (2/sqrt(pi)) sum_{n>=0} x^{2n+1} / (n! (2n+1)).  The n-th term is carried as
// u_n = x^{2n+1}/n! (u_n = u_{n-1} x^2 / n) and divided by (2n+1) -- the previous code divided
// by n(2n+1), i.e. it dropped the (2n-1) factor of the true ratio x^2(2n-1)/(n(2n+1)) and so
// summed x^{2n+1}/(n!(2n+1)!!) instead.  Every term is positive for x > 0, so there is no
// cancellation; the series just needs enough terms (it peaks near n = x^2).
double erfi(double x) {
    const double x2 = x * x;
    double u = x;
    double sum = x;
    for (int n = 1; n < 1200; ++n) {
        u *= x2 / static_cast<double>(n);
        const double term = u / (2.0 * static_cast<double>(n) + 1.0);
        sum += term;
        if (!std::isfinite(sum)) {
            break;
        }
        if (static_cast<double>(n) > x2 && std::abs(term) <= 1e-18 * std::abs(sum)) {
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

double voigt(double x, double sigma, double gamma) {
    const double g = std::max(gamma, 0.0);
    if (sigma <= 0.0) {
        if (g > 0.0) {
            return g / (M_PI * (x * x + g * g));
        }
        return x == 0.0 ? std::numeric_limits<double>::infinity() : 0.0;
    }
    const double scale = sigma * std::sqrt(2.0);
    const double v = x / scale;
    const double a = g / scale;
    const std::complex<double> w = humlicek_w4(v, a);
    return w.real() / (sigma * std::sqrt(2.0 * M_PI));
}

double pseudo_voigt(double x, double sigma, double gamma, double eta) {
    const PseudoVoigtFwhm fwhm = pseudo_voigt_fwhm(sigma, gamma);
    if (fwhm.f <= 0.0) {
        return x == 0.0 ? std::numeric_limits<double>::infinity() : 0.0;
    }
    const double eta_c = std::clamp(eta, 0.0, 1.0);
    const double gamma_pv = fwhm.f / 2.0;
    const double sigma_pv = fwhm.f / (2.0 * std::sqrt(2.0 * std::log(2.0)));
    const double lorentzian = gamma_pv / (M_PI * (x * x + gamma_pv * gamma_pv));
    const double gaussian = std::exp(-x * x / (2.0 * sigma_pv * sigma_pv)) / (sigma_pv * std::sqrt(2.0 * M_PI));
    return eta_c * lorentzian + (1.0 - eta_c) * gaussian;
}

double pseudo_voigt_auto(double x, double sigma, double gamma) {
    const PseudoVoigtFwhm fwhm = pseudo_voigt_fwhm(sigma, gamma);
    if (fwhm.f <= 0.0) {
        return x == 0.0 ? std::numeric_limits<double>::infinity() : 0.0;
    }
    const double r = fwhm.f_l / fwhm.f;
    const double eta = 1.36603 * r - 0.47719 * r * r + 0.11116 * r * r * r;
    return pseudo_voigt(x, sigma, gamma, eta);
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

double pochhammer(double a, int n) {
    if (n < 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double value = 1.0;
    for (int k = 0; k < n; ++k) {
        value *= a + static_cast<double>(k);
    }
    return value;
}

double falling_factorial(double a, int n) {
    if (n < 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double value = 1.0;
    for (int k = 0; k < n; ++k) {
        value *= a - static_cast<double>(k);
    }
    return value;
}

double rgamma(double x) {
    if (x <= 0.0 && x == std::floor(x)) {
        return 0.0;
    }
    const double g = gamma_func(x);
    if (!std::isfinite(g) || std::abs(g) <= 1e-300) {
        return 0.0;
    }
    return 1.0 / g;
}

double gamma_inc_reg(double a, double x) {
    return gamma_inc_reg_impl(a, x);
}

double gamma_inc_reg_upper(double a, double x) {
    if (x <= 0.0) {
        return 1.0;
    }
    return 1.0 - gamma_inc_reg_impl(a, x);
}

double gamma_inc(double a, double x) {
    return gamma_inc_reg(a, x) * gamma_func(a);
}

double beta_inc_reg(double x, double a, double b) {
    return beta_inc_reg_impl(a, b, x);
}

double beta_inc(double x, double a, double b) {
    return beta_inc_reg(x, a, b) * beta_func(a, b);
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
    return airy_all(x).ai;
}

double airy_bi(double x) {
    return airy_all(x).bi;
}

double airy_aip(double x) {
    return airy_all(x).aip;
}

double airy_bip(double x) {
    return airy_all(x).bip;
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
    return bessel_yn(nu, x);
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
    if (nu < 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return struve_series(nu, x, false);
}

double struve_k(int nu, double x) {
    // K_nu = H_nu - Y_nu, so it inherits Y's domain: x <= 0 is a singularity, not a value.
    if (nu < 0 || x <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (x > 20.0) {
        return struve_h_minus_y_asymptotic(nu, x);
    }
    return struve_h(nu, x) - bessel_y(nu, x);
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

double sph_bessel_j(int n, double x) {
    if (n < 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (x == 0.0) {
        return n == 0 ? 1.0 : 0.0;
    }
    const double j0 = std::sin(x) / x;
    if (n == 0) {
        return j0;
    }
    const double j1 = std::sin(x) / (x * x) - std::cos(x) / x;
    double jPrev = j0;
    double jCurr = j1;
    for (int k = 1; k < n; ++k) {
        const double jNext = ((2.0 * k + 1.0) / x) * jCurr - jPrev;
        jPrev = jCurr;
        jCurr = jNext;
    }
    return jCurr;
}

double sph_bessel_y(int n, double x) {
    if (n < 0 || x <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double y0 = -std::cos(x) / x;
    if (n == 0) {
        return y0;
    }
    const double y1 = -std::cos(x) / (x * x) - std::sin(x) / x;
    double yPrev = y0;
    double yCurr = y1;
    for (int k = 1; k < n; ++k) {
        const double yNext = ((2.0 * k + 1.0) / x) * yCurr - yPrev;
        yPrev = yCurr;
        yCurr = yNext;
    }
    return yCurr;
}

double bessel_zero_jnu(int nu, int n) {
    if (nu < 0 || n <= 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const auto fn = [nu](double x) { return bessel_jn(nu, x); };
    // J_nu is strictly positive on (0, j_{nu,1}), so a scan started just above the origin sees
    // exactly the positive zeros, in order.
    const ZeroBracket bracket = bessel_scan_bracket(n, 1e-8, 0.5, fn);
    if (!bracket.found) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double bisected = bracket_zero(bracket.lo, bracket.hi, fn);
    const double polished = bessel_zero_newton_j(nu, bisected);
    // Newton may leave the bracket for a nearly flat derivative; fall back to the bisection.
    if (!(polished > bracket.lo - 1e-6) || !(polished < bracket.hi + 1e-6)) {
        return bisected;
    }
    return polished;
}

double bessel_zero_ynu(int nu, int n) {
    if (nu < 0 || n <= 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const auto fn = [nu](double x) { return bessel_yn(nu, x); };
    // Y_nu(x) -> -inf as x -> 0+, so the scan again starts just above the origin.
    const ZeroBracket bracket = bessel_scan_bracket(n, 1e-6, 0.5, fn);
    if (!bracket.found) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double bisected = bracket_zero(bracket.lo, bracket.hi, fn);
    const double polished = bessel_zero_newton_y(nu, bisected);
    if (!(polished > bracket.lo - 1e-6) || !(polished < bracket.hi + 1e-6)) {
        return bisected;
    }
    return polished;
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
    return kelvin_ber_bei(static_cast<double>(nu), x).real();
}

double kelvin_bei(int nu, double x) {
    if (nu < 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return kelvin_ber_bei(static_cast<double>(nu), x).imag();
}

double kelvin_ker(int nu, double x) {
    if (nu < 0 || x <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return kelvin_ker_kei(static_cast<double>(nu), x).real();
}

double kelvin_kei(int nu, double x) {
    if (nu < 0 || x <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return kelvin_ker_kei(static_cast<double>(nu), x).imag();
}

double legendre_p(int n, double x) {
    if (x < -1.0 || x > 1.0 || n < 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return legendre_recurrence(n, x);
}

// Legendre function of the second kind on (-1, 1).  Q_n(x) = (1/2) P_n(x) ln((1+x)/(1-x))
// - W_{n-1}(x); rather than accumulating W explicitly, Q is generated from Q_0 = atanh(x) and
// Q_1 = x Q_0 - 1 by the same three-term recurrence P and Q share (DLMF 14.10.3):
//   (k+1) Q_{k+1}(x) = (2k+1) x Q_k(x) - k Q_{k-1}(x).
double legendre_q(int n, double x) {
    if (x <= -1.0 || x >= 1.0 || n < 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double q0 = 0.5 * std::log((1.0 + x) / (1.0 - x));
    if (n == 0) {
        return q0;
    }
    double previous = q0;
    double current = x * q0 - 1.0;
    for (int k = 1; k < n; ++k) {
        const double kd = static_cast<double>(k);
        const double next = ((2.0 * kd + 1.0) * x * current - kd * previous) / (kd + 1.0);
        previous = current;
        current = next;
    }
    return current;
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

double assoc_legendre_p(int l, int m, double x) {
    if (l < 0 || std::abs(m) > l || x < -1.0 || x > 1.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (m >= 0) {
        return legendre_pn(l, m, x);
    }
    const int abs_m = -m;
    const double plm = legendre_pn(l, abs_m, x);
    const double ratio = std::tgamma(static_cast<double>(l - abs_m) + 1.0) /
                         std::tgamma(static_cast<double>(l + abs_m) + 1.0);
    const double sign = (abs_m % 2 == 0) ? 1.0 : -1.0;
    return sign * ratio * plm;
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

// Associated (generalised) Laguerre polynomial L_n^{(k)}(x) for integer k >= 0 -- the integer
// case of the laguerre_generalized routine already in this file.  The previous body returned
// L_n(x) (-x)^k, which is not L_n^{(k)}, not its k-th derivative, and not any standard object.
double laguerre_ln(int n, int k, double x) {
    if (k < 0) {
        return 0.0;
    }
    if (n < 0) {
        return 0.0;
    }
    return laguerre_generalized(n, static_cast<double>(k), x);
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

// k-th derivative of the Chebyshev polynomials, via the Gegenbauer differentiation formula
// d^k/dx^k C_n^{(lambda)}(x) = 2^k (lambda)_k C_{n-k}^{(lambda+k)}(x) (DLMF 18.9.19-18.9.21):
//   d^k T_n / dx^k = n 2^{k-1} (k-1)! C_{n-k}^{(k)}(x)   (k >= 1),   T_n itself for k = 0
//   d^k U_n / dx^k = 2^k k! C_{n-k}^{(k+1)}(x)
// Both vanish identically for k > n.  k < 0 is outside the domain and returns NaN.
double chebyshev_tn(int n, int k, double x) {
    if (k < 0 || n < 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (x < -1.0 || x > 1.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (k == 0) {
        return chebyshev_t(n, x);
    }
    if (k > n) {
        return 0.0;
    }
    const double scale = static_cast<double>(n) * std::pow(2.0, static_cast<double>(k) - 1.0) *
                         std::tgamma(static_cast<double>(k));
    return scale * gegenbauer_recurrence(n - k, static_cast<double>(k), x);
}

double chebyshev_un(int n, int k, double x) {
    if (k < 0 || n < 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (x < -1.0 || x > 1.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (k == 0) {
        return chebyshev_u(n, x);
    }
    if (k > n) {
        return 0.0;
    }
    const double scale = std::pow(2.0, static_cast<double>(k)) *
                         std::tgamma(static_cast<double>(k) + 1.0);
    return scale * gegenbauer_recurrence(n - k, static_cast<double>(k) + 1.0, x);
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

double kummer_u(double a, double b, double z) {
    if (z <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (std::abs(b - a - 1.0) <= 1e-12 * std::max(1.0, std::abs(a))) {
        return std::pow(z, -a);
    }
    if (b == std::floor(b)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (z > 20.0) {
        return std::pow(z, -a);
    }
    if (b < 2.0 - 1e-12) {
        const double connected = kummer_u_connection(a, b, z);
        if (std::isfinite(connected)) {
            return connected;
        }
    }
    return tricomi_u_impl(a, b, z);
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

std::complex<double> sph_harm_y(int l, int m, double theta, double phi) {
    if (l < 0 || std::abs(m) > l) {
        return {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
    }
    if (m < 0) {
        const std::complex<double> positive = sph_harm_y(l, -m, theta, phi);
        const int abs_m = -m;
        const double sign = (abs_m % 2 == 0) ? 1.0 : -1.0;
        return {sign * positive.real(), -sign * positive.imag()};
    }
    const double cos_theta = std::cos(theta);
    const double plm = assoc_legendre_p(l, m, cos_theta);
    const double norm = std::sqrt((2.0 * l + 1.0) / (4.0 * M_PI) *
                                  std::tgamma(static_cast<double>(l - m) + 1.0) /
                                  std::tgamma(static_cast<double>(l + m) + 1.0));
    const double amp = norm * plm;
    return {amp * std::cos(static_cast<double>(m) * phi), amp * std::sin(static_cast<double>(m) * phi)};
}

double sph_harm(int l, int m, double theta, double phi) {
    return sph_harm_y(l, m, theta, phi).real();
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

// Glaisher notation: pq(u,k) = p(u)/q(u) with n(u) identically 1, so nd = 1/dn and nc = 1/cn.
// (These two used to return cn/dn and cn/sn, i.e. duplicates of jacobi_cd and jacobi_cs.)
double jacobi_nd(double u, double k) {
    const JacobiTriple values = jacobi_compute(u, k);
    return safe_ratio(1.0, values.dn);
}

double jacobi_nc(double u, double k) {
    const JacobiTriple values = jacobi_compute(u, k);
    return safe_ratio(1.0, values.cn);
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

// k-th eigenvalue (0-based, ascending) of a symmetric tridiagonal matrix, by bisection on the
// Sturm sequence: the number of eigenvalues strictly below lambda equals the number of negative
// pivots of the LDL^T factorisation of A - lambda I,
//     q_1 = d_1 - lambda,   q_i = (d_i - lambda) - e_{i-1}^2 / q_{i-1}.
// This is used in place of the shared cpu::lapack::dsteqr because that routine loses
// eigenvalues on nearly-diagonal matrices of dimension >= 80 (verified directly: for
// diag = (0,4,16,...,192^2) with off-diagonal 0.1 it returns duplicates), which is exactly the
// shape of the Mathieu and spheroidal problems at high order.
double tridiagonal_eigenvalue(const std::vector<double>& diag, const std::vector<double>& off,
                              int index) {
    const int n = static_cast<int>(diag.size());
    if (n <= 0 || index < 0 || index >= n) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double lower = 0.0;
    double upper = 0.0;
    for (int i = 0; i < n; ++i) {
        const double left = (i > 0) ? std::abs(off[static_cast<std::size_t>(i - 1)]) : 0.0;
        const double right = (i + 1 < n) ? std::abs(off[static_cast<std::size_t>(i)]) : 0.0;
        const double radius = left + right;
        lower = (i == 0) ? diag[0] - radius : std::min(lower, diag[static_cast<std::size_t>(i)] - radius);
        upper = (i == 0) ? diag[0] + radius : std::max(upper, diag[static_cast<std::size_t>(i)] + radius);
    }
    const double scale = std::max(1.0, std::max(std::abs(lower), std::abs(upper)));
    lower -= scale * 1e-12;
    upper += scale * 1e-12;
    const auto count_below = [&](double lambda) {
        int negatives = 0;
        double pivot = diag[0] - lambda;
        if (pivot < 0.0) {
            ++negatives;
        }
        for (int i = 1; i < n; ++i) {
            const double e = off[static_cast<std::size_t>(i - 1)];
            if (pivot == 0.0) {
                pivot = std::abs(e) * 1e-300 - std::abs(e) * 1e-16 - 1e-300;
                if (pivot == 0.0) {
                    pivot = -1e-300;
                }
            }
            pivot = (diag[static_cast<std::size_t>(i)] - lambda) - e * e / pivot;
            if (pivot < 0.0) {
                ++negatives;
            }
        }
        return negatives;
    };
    for (int iteration = 0; iteration < 200; ++iteration) {
        const double mid = 0.5 * (lower + upper);
        if (mid <= lower || mid >= upper) {
            break;
        }
        if (count_below(mid) > index) {
            upper = mid;
        } else {
            lower = mid;
        }
        if (upper - lower <= 1e-16 * scale) {
            break;
        }
    }
    return 0.5 * (lower + upper);
}

// The four Mathieu characteristic-value problems are four DIFFERENT symmetric tridiagonal
// matrices (DLMF 28.2 / A&S 20.2), not one matrix patched from outside:
//
//   a_{2n}  : diag (0, 4, 16, 36, ...)   offdiag (sqrt2 q, q, q, ...)   [ce_{2n},  k = 2r  ]
//   a_{2n+1}: diag (1+q, 9, 25, ...)     offdiag (q, q, ...)            [ce_{2n+1},k = 2r+1]
//   b_{2n+1}: diag (1-q, 9, 25, ...)     offdiag (q, q, ...)            [se_{2n+1},k = 2r+1]
//   b_{2n+2}: diag (4, 16, 36, ...)      offdiag (q, q, ...)            [se_{2n+2},k = 2r+2]
//
// The sqrt(2) on the first off-diagonal of the even-cosine problem is the symmetrisation of the
// factor-2 coupling a A_0 = q A_2 / (a-4) A_2 = q(2 A_0 + A_4); the +/-q on the FIRST DIAGONAL
// ENTRY of the odd problems comes from A_{-1} = +/- A_1.  Feeding one uniform matrix
// (diagonal k^2, off-diagonal -q) into dsteqr and then multiplying a_0 by 2, adding q to the
// odd orders and subtracting 2q from the even b's -- as this file used to -- reproduces none of
// them: it gave a_0(1) = -0.478734 against the true -0.4551386 and b_2(1) = 2.155963 against
// 3.9170251.
enum class MathieuKind { EvenCos = 0, OddCos = 1, OddSin = 2, EvenSin = 3 };

MathieuKind mathieu_kind(int n, bool sine) {
    if (sine) {
        return (n % 2 == 1) ? MathieuKind::OddSin : MathieuKind::EvenSin;
    }
    return (n % 2 == 0) ? MathieuKind::EvenCos : MathieuKind::OddCos;
}

// Lowest harmonic present in the Fourier series of this kind (k_r = k0 + 2r).
int mathieu_first_harmonic(MathieuKind kind) {
    switch (kind) {
    case MathieuKind::EvenCos:
        return 0;
    case MathieuKind::OddCos:
    case MathieuKind::OddSin:
        return 1;
    default:
        return 2;
    }
}

double mathieu_characteristic(MathieuKind kind, double q, int index) {
    // The off-diagonal coupling is q against a diagonal growing like k^2, so a matrix reaching
    // well past both the requested index and sqrt(q) already isolates the eigenvalue to machine
    // precision; sizing it here instead of using a fixed 80 keeps the cost proportional to the
    // problem.
    const int dim = std::max(24, index + 16 + static_cast<int>(std::ceil(std::sqrt(std::abs(q)))));
    const int k0 = mathieu_first_harmonic(kind);
    std::vector<double> d(static_cast<std::size_t>(dim));
    std::vector<double> e(static_cast<std::size_t>(dim - 1));
    for (int j = 0; j < dim; ++j) {
        const double k = static_cast<double>(k0 + 2 * j);
        d[static_cast<std::size_t>(j)] = k * k;
    }
    if (kind == MathieuKind::OddCos) {
        d[0] += q;
    } else if (kind == MathieuKind::OddSin) {
        d[0] -= q;
    }
    for (int j = 0; j < dim - 1; ++j) {
        e[static_cast<std::size_t>(j)] = q;
    }
    if (kind == MathieuKind::EvenCos) {
        e[0] = std::sqrt(2.0) * q;
    }
    return tridiagonal_eigenvalue(d, e, index);
}

// Normalised Fourier coefficients of ce_n / se_n.  With k_r = k0 + 2r, the Mathieu recurrence
// (a - k_r^2) A_{k_r} = q (A_{k_{r-1}} + A_{k_{r+1}}) is solved for the ratios
// R_r = A_{k_{r+1}}/A_{k_r} by the backward continued fraction
//     R_r = q / (a - k_{r+1}^2 - q R_{r+1}),      R_infinity = 0,
// with the single exception R_0 = 2q/(a - 4 - q R_1) for the even-cosine family, where the
// r = 1 relation carries the factor 2 A_0.  The coefficients are then scaled to the standard
// A&S 20.2.27 normalisation (integral of ce_n^2 or se_n^2 over a period equals pi):
//     2 A_0^2 + sum_{r>=1} A_{2r}^2 = 1   (even cosine),   sum_r A_{k_r}^2 = 1  (otherwise),
// with the sign fixed so that ce_n(x,q) -> cos(nx) and se_n(x,q) -> sin(nx) as q -> 0.
constexpr int kMathieuTerms = 60;

std::vector<double> mathieu_coefficients(int n, MathieuKind kind, double q, double characteristic) {
    const int k0 = mathieu_first_harmonic(kind);
    std::vector<double> ratios(static_cast<std::size_t>(kMathieuTerms), 0.0);
    for (int r = kMathieuTerms - 2; r >= 0; --r) {
        const double k_next = static_cast<double>(k0 + 2 * (r + 1));
        const double numerator =
            (kind == MathieuKind::EvenCos && r == 0) ? 2.0 * q : q;
        const double denom = characteristic - k_next * k_next -
                             q * ratios[static_cast<std::size_t>(r + 1)];
        ratios[static_cast<std::size_t>(r)] = (denom == 0.0) ? 0.0 : numerator / denom;
    }
    std::vector<double> coefficients(static_cast<std::size_t>(kMathieuTerms), 0.0);
    coefficients[0] = 1.0;
    for (int r = 1; r < kMathieuTerms; ++r) {
        coefficients[static_cast<std::size_t>(r)] =
            coefficients[static_cast<std::size_t>(r - 1)] * ratios[static_cast<std::size_t>(r - 1)];
    }
    double norm = 0.0;
    for (int r = 0; r < kMathieuTerms; ++r) {
        const double c = coefficients[static_cast<std::size_t>(r)];
        const double weight = (kind == MathieuKind::EvenCos && r == 0) ? 2.0 : 1.0;
        norm += weight * c * c;
    }
    norm = std::sqrt(norm);
    if (!(norm > 0.0) || !std::isfinite(norm)) {
        return coefficients;
    }
    // Fix the phase: the harmonic k = n must carry a positive coefficient.
    const int principal = (n - k0) / 2;
    double sign = 1.0;
    if (principal >= 0 && principal < kMathieuTerms &&
        coefficients[static_cast<std::size_t>(principal)] < 0.0) {
        sign = -1.0;
    }
    for (int r = 0; r < kMathieuTerms; ++r) {
        coefficients[static_cast<std::size_t>(r)] *= sign / norm;
    }
    return coefficients;
}

// Evaluates the Fourier series.  `hyperbolic` swaps cos -> cosh and sin -> sinh, which is
// exactly the substitution x -> i z that turns ce_n / se_n into the modified (radial) Mathieu
// functions Mc_n(z,q) = ce_n(iz,q) and Ms_n(z,q) = -i se_n(iz,q).
struct MathieuSeries {
    double value;
    double derivative;
};

MathieuSeries mathieu_series_eval(const std::vector<double>& coefficients, MathieuKind kind,
                                  double x, bool hyperbolic) {
    const int k0 = mathieu_first_harmonic(kind);
    const bool sine = (kind == MathieuKind::OddSin || kind == MathieuKind::EvenSin);
    double value = 0.0;
    double derivative = 0.0;
    for (int r = 0; r < kMathieuTerms; ++r) {
        const double k = static_cast<double>(k0 + 2 * r);
        const double c = coefficients[static_cast<std::size_t>(r)];
        double basis = 0.0;
        double slope = 0.0;
        if (hyperbolic) {
            basis = sine ? std::sinh(k * x) : std::cosh(k * x);
            slope = k * (sine ? std::cosh(k * x) : std::sinh(k * x));
        } else {
            basis = sine ? std::sin(k * x) : std::cos(k * x);
            slope = k * (sine ? std::cos(k * x) : -std::sin(k * x));
        }
        value += c * basis;
        derivative += c * slope;
        if (r > 4 && std::abs(c * basis) <= 1e-18 * std::max(1.0, std::abs(value)) &&
            std::abs(c) <= 1e-18) {
            break;
        }
    }
    return {value, derivative};
}

double mathieu_series_value(int n, MathieuKind kind, double q, double x, double characteristic) {
    const std::vector<double> coefficients = mathieu_coefficients(n, kind, q, characteristic);
    return mathieu_series_eval(coefficients, kind, x, false).value;
}

// Modified (radial) Mathieu function.  The cosh/sinh series converges for every z, but the
// terms peak at magnitude ~exp(sqrt(q) e^{|z|}) while the sum itself is O(e^{-|z|/2}), so it
// loses all its significant digits once sqrt(q) e^{|z|} exceeds ~35.  Below that threshold the
// series is used directly; above it, the solution is continued with RK4 on the defining
// equation y'' = (a - 2q cosh 2z) y, seeded with the series value and derivative at the
// threshold.  That continuation is well conditioned (both solutions of the modified equation
// are O(e^{-z/2}) oscillations of frequency sqrt(q) e^z) but needs a step per fraction of a
// wavelength, so it is budgeted; beyond the budget NaN is returned rather than noise.
double mathieu_modified_value(int n, MathieuKind kind, double q, double z, double characteristic) {
    const std::vector<double> coefficients = mathieu_coefficients(n, kind, q, characteristic);
    const double root_q = std::sqrt(std::abs(q));
    const double az = std::abs(z);
    const double safe = (root_q <= 1e-12) ? 1e300 : std::log(25.0 / root_q);
    const bool sine = (kind == MathieuKind::OddSin || kind == MathieuKind::EvenSin);
    if (az <= std::max(0.0, safe)) {
        const double v = mathieu_series_eval(coefficients, kind, az, true).value;
        return (z < 0.0 && sine) ? -v : v;
    }
    const double start = std::max(0.0, safe);
    const MathieuSeries seed = mathieu_series_eval(coefficients, kind, start, true);
    // One step per ~1/50 of a wavelength of the fastest oscillation reached.
    const double phase = root_q * (std::exp(az) - std::exp(start));
    const double steps_needed = 50.0 * phase + 2000.0;
    if (!(steps_needed <= 4.0e6)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const int steps = static_cast<int>(steps_needed);
    const double h = (az - start) / static_cast<double>(steps);
    double y = seed.value;
    double yp = seed.derivative;
    const auto accel = [&](double t, double yy) {
        return (characteristic - 2.0 * q * std::cosh(2.0 * t)) * yy;
    };
    for (int i = 0; i < steps; ++i) {
        const double t = start + h * static_cast<double>(i);
        const double k1 = yp;
        const double l1 = accel(t, y);
        const double k2 = yp + 0.5 * h * l1;
        const double l2 = accel(t + 0.5 * h, y + 0.5 * h * k1);
        const double k3 = yp + 0.5 * h * l2;
        const double l3 = accel(t + 0.5 * h, y + 0.5 * h * k2);
        const double k4 = yp + h * l3;
        const double l4 = accel(t + h, y + h * k3);
        y += h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
        yp += h * (l1 + 2.0 * l2 + 2.0 * l3 + l4) / 6.0;
    }
    return (z < 0.0 && sine) ? -y : y;
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

// ---------------------------------------------------------------------------------------
// Weierstrass elliptic functions near the origin.  All four are derived from ONE array of
// Laurent coefficients c_n of
//     P(z) = z^{-2} + sum_{n>=2} c_n z^{2n-2},
// generated by the DLMF 23.9.3 recurrence
//     c_2 = g2/20,  c_3 = g3/28,  c_n = 3/((2n+1)(n-3)) sum_{m=2}^{n-2} c_m c_{n-m}  (n >= 4),
// so that the remaining three follow by exact term-wise differentiation / integration:
//     P'(z)  = -2 z^{-3} + sum (2n-2) c_n z^{2n-3}
//     zeta(z) = 1/z - sum c_n z^{2n-1}/(2n-1)          (zeta' = -P by construction)
//     sigma(z) = z exp(-sum c_n z^{2n}/(2n(2n-1)))     (sigma'/sigma = zeta by construction)
// The old hand-written coefficients were wrong from the z^6 term of P onwards (g2^2/960 instead
// of /1200, g2 g3/2016 instead of 3 g2 g3/6160, g2^2 z^9/5760 in sigma instead of /161280), and
// weierstrass_zeta had entirely the wrong powers AND sign (+g2 z/6 where the true leading
// correction is -g2 z^3/60), so it violated its own defining relation zeta' = -P.
//
// These are Laurent series about z = 0 and converge only for |z| below the modulus of the
// nearest lattice point; the sum is truncated when the terms stop contributing, and NaN is
// returned when they instead diverge (i.e. z is outside the disc of convergence).
constexpr int kWeierstrassTerms = 26;

void weierstrass_coefficients(double g2, double g3, double* c) {
    for (int n = 0; n < kWeierstrassTerms; ++n) {
        c[n] = 0.0;
    }
    c[2] = g2 / 20.0;
    c[3] = g3 / 28.0;
    for (int n = 4; n < kWeierstrassTerms; ++n) {
        double acc = 0.0;
        for (int m = 2; m <= n - 2; ++m) {
            acc += c[m] * c[n - m];
        }
        c[n] = 3.0 * acc / ((2.0 * static_cast<double>(n) + 1.0) * (static_cast<double>(n) - 3.0));
    }
}

// Sums sum_{n>=2} c_n z^{2n-2} * weight(n) with the given per-term extra power of z and
// reports whether the truncated tail is negligible.
struct WeierstrassSum {
    double value;
    bool converged;
};

WeierstrassSum weierstrass_series(const double* c, double z, int mode) {
    // mode 0: term = c_n z^{2n-2}                 (P)
    // mode 1: term = (2n-2) c_n z^{2n-3}          (P')
    // mode 2: term = c_n z^{2n-1} / (2n-1)        (zeta)
    // mode 3: term = c_n z^{2n} / (2n(2n-1))      (log sigma)
    double sum = 0.0;
    double last = 0.0;
    double largest = 0.0;
    for (int n = 2; n < kWeierstrassTerms; ++n) {
        const double nd = static_cast<double>(n);
        double term = 0.0;
        switch (mode) {
        case 0:
            term = c[n] * std::pow(z, 2.0 * nd - 2.0);
            break;
        case 1:
            term = (2.0 * nd - 2.0) * c[n] * std::pow(z, 2.0 * nd - 3.0);
            break;
        case 2:
            term = c[n] * std::pow(z, 2.0 * nd - 1.0) / (2.0 * nd - 1.0);
            break;
        default:
            term = c[n] * std::pow(z, 2.0 * nd) / (2.0 * nd * (2.0 * nd - 1.0));
            break;
        }
        sum += term;
        last = std::abs(term);
        largest = std::max(largest, last);
    }
    const bool converged = !(largest > 0.0) || last <= 1e-10 * std::max(1e-30, largest);
    return {sum, converged && std::isfinite(sum)};
}

double weierstrass_p(double z, double g2, double g3) {
    if (std::abs(z) <= 1e-12) {
        return std::numeric_limits<double>::infinity();
    }
    double c[kWeierstrassTerms];
    weierstrass_coefficients(g2, g3, c);
    const WeierstrassSum sum = weierstrass_series(c, z, 0);
    if (!sum.converged) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return 1.0 / (z * z) + sum.value;
}

double weierstrass_pprime(double z, double g2, double g3) {
    if (std::abs(z) <= 1e-12) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double c[kWeierstrassTerms];
    weierstrass_coefficients(g2, g3, c);
    const WeierstrassSum sum = weierstrass_series(c, z, 1);
    if (!sum.converged) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return -2.0 / (z * z * z) + sum.value;
}

double weierstrass_zeta(double z, double g2, double g3) {
    if (std::abs(z) <= 1e-12) {
        return std::numeric_limits<double>::infinity();
    }
    double c[kWeierstrassTerms];
    weierstrass_coefficients(g2, g3, c);
    const WeierstrassSum sum = weierstrass_series(c, z, 2);
    if (!sum.converged) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return 1.0 / z - sum.value;
}

double weierstrass_sigma(double z, double g2, double g3) {
    if (z == 0.0) {
        return 0.0;
    }
    double c[kWeierstrassTerms];
    weierstrass_coefficients(g2, g3, c);
    const WeierstrassSum sum = weierstrass_series(c, z, 3);
    if (!sum.converged) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return z * std::exp(-sum.value);
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
        // Euler-Maclaurin (DLMF 25.2.9):
        //   zeta(s) = sum_{n<N} n^{-s} + N^{-s}/2 + N^{1-s}/(s-1)
        //             + sum_{k>=1} B_{2k} (s)_{2k-1} / ((2k)! N^{s+2k-1}).
        // The previous body summed n = 1..N INCLUSIVE and then added +N^{-s}/2 instead of
        // subtracting it, leaving an error of exactly N^{-s} -- 1.1e-5 at s = 1.5 with its
        // N = 2000 -- and carried the wrong powers of N on the Bernoulli terms.
        constexpr int N = 32;
        const double nd = static_cast<double>(N);
        double sum = 0.0;
        for (int n = 1; n < N; ++n) {
            sum += std::pow(static_cast<double>(n), -s);
        }
        const double ns = std::pow(nd, -s);
        sum += 0.5 * ns + nd * ns / (s - 1.0);
        // B_2/2!, B_4/4!, B_6/6!, B_8/8!, B_10/10!
        static const double kBernoulliOverFactorial[5] = {
            1.0 / 12.0, -1.0 / 720.0, 1.0 / 30240.0, -1.0 / 1209600.0, 1.0 / 47900160.0};
        double rising = s;              // (s)_{2k-1}
        double power = ns / nd;         // N^{-s-2k+1}
        for (int k = 0; k < 5; ++k) {
            sum += kBernoulliOverFactorial[k] * rising * power;
            const double j = 2.0 * static_cast<double>(k) + 1.0;
            rising *= (s + j) * (s + j + 1.0);
            power /= nd * nd;
        }
        return sum;
    }
    if (s > 0.0) {
        return eta_dirichlet(s) / (1.0 - std::pow(2.0, 1.0 - s));
    }
    if (s == 0.0) {
        return -0.5;
    }
    // Reflection formula, DLMF 25.4.2: zeta(s) = 2^s pi^{s-1} sin(pi s/2) Gamma(1-s) zeta(1-s).
    // For s < 0 this recurses into the Euler-Maclaurin branch above (1 - s > 1).  The trivial
    // zeros at the negative even integers are returned exactly rather than via sin(pi s/2).
    const double half = 0.5 * s;
    if (half == std::floor(half)) {
        return 0.0;
    }
    return std::pow(2.0, s) * std::pow(M_PI, s - 1.0) * std::sin(M_PI * half) *
           std::tgamma(1.0 - s) * zeta(1.0 - s);
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

double polygamma(int n, double x) {
    if (n < 0 || x <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (n == 0) {
        return digamma(x);
    }
    const double sign = (n % 2 == 0) ? -1.0 : 1.0;
    return sign * std::tgamma(static_cast<double>(n) + 1.0) * zeta_hurwitz(static_cast<double>(n) + 1.0, x);
}

double trigamma(double x) {
    return polygamma(1, x);
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

// Dirichlet eta via Borwein's Chebyshev-polynomial acceleration (Borwein 2000, algorithm 2):
//     eta(s) ~ -(1/d_N) sum_{k=0}^{N-1} (-1)^k (d_k - d_N) / (k+1)^s,
//     d_k = N sum_{i=0}^{k} (N+i-1)! 4^i / ((N-i)! (2i)!),
// whose error is bounded by 3/(3+sqrt 8)^N -- below 1e-22 at N = 30.  The previous body
// truncated the raw alternating series at 100000 terms, whose O(N^{-s}/2) truncation error left
// only ~3 correct digits at s = 1/2, and returned NaN for every s <= 0 although eta is entire.
double eta_dirichlet(double s) {
    if (s > 1.0) {
        return (1.0 - std::pow(2.0, 1.0 - s)) * zeta(s);
    }
    constexpr int N = 30;
    const double nd = static_cast<double>(N);
    // t_i = (N+i-1)! 4^i / ((N-i)! (2i)!), t_0 = 1/N,
    // t_i = t_{i-1} * 4 (N+i-1)(N-i+1) / ((2i)(2i-1));   d_k = N sum_{i<=k} t_i.
    double t = 1.0 / nd;
    double dk = nd * t;
    std::vector<double> partial(static_cast<std::size_t>(N) + 1, 0.0);
    partial[0] = dk;
    for (int i = 1; i <= N; ++i) {
        const double id = static_cast<double>(i);
        t *= 4.0 * (nd + id - 1.0) * (nd - id + 1.0) / ((2.0 * id) * (2.0 * id - 1.0));
        dk += nd * t;
        partial[static_cast<std::size_t>(i)] = dk;
    }
    const double dn = partial[static_cast<std::size_t>(N)];
    double sum = 0.0;
    for (int k = 0; k < N; ++k) {
        const double sign = (k % 2 == 0) ? 1.0 : -1.0;
        sum += sign * (partial[static_cast<std::size_t>(k)] - dn) /
               std::pow(static_cast<double>(k) + 1.0, s);
    }
    return -sum / dn;
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

namespace {

const double kMinusInvE = -std::exp(-1.0);

double lambert_w_fritsch(double w, double z) {
    for (int iter = 0; iter < 64; ++iter) {
        const double ew = std::exp(w);
        const double w1 = w + 1.0;
        const double f = w * ew - z;
        if (std::abs(f) <= 1e-15 * std::max(1.0, std::abs(z))) {
            break;
        }
        const double sigma = f / (ew * w1);
        const double denom = 1.0 + sigma * (w + 2.0) / (2.0 * w1);
        w -= sigma / denom;
    }
    return w;
}

double lambert_w0_initial(double z) {
    if (z == 0.0) {
        return 0.0;
    }
    if (z < 0.0) {
        const double p = std::sqrt(2.0 * (1.0 + std::exp(1.0) * z));
        return -1.0 + p - p * p * p / 3.0;
    }
    if (z < 1.35) {
        return z * (6.0 + z) / (6.0 + 4.0 * z);
    }
    const double l1 = std::log(z);
    const double l2 = std::log(l1);
    return l1 - l2 + l2 / l1;
}

double lambert_wm1_initial(double z) {
    if (z <= kMinusInvE + 1e-10) {
        const double p = std::sqrt(2.0 * (1.0 + std::exp(1.0) * z));
        return -1.0 - p + p * p * p / 3.0;
    }
    const double l1 = std::log(-z);
    const double l2 = std::log(-l1);
    return l1 - l2 + l2 / l1;
}

} // namespace

double lambert_w(int branch, double z) {
    if (branch != 0 && branch != -1) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (branch == 0) {
        if (z < kMinusInvE - 1e-15) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        if (z == 0.0) {
            return 0.0;
        }
        if (std::abs(z - kMinusInvE) <= 1e-15) {
            return -1.0;
        }
        return lambert_w_fritsch(lambert_w0_initial(z), z);
    }
    // branch -1: z in [−1/e, 0)
    if (z < kMinusInvE - 1e-15 || z >= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (std::abs(z - kMinusInvE) <= 1e-15) {
        return -1.0;
    }
    return lambert_w_fritsch(lambert_wm1_initial(z), z);
}

double mathieu_a(int n, double q) {
    if (n < 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const MathieuKind kind = mathieu_kind(n, false);
    return mathieu_characteristic(kind, q, n / 2);
}

double mathieu_b(int n, double q) {
    if (n < 1) {
        // se_0 is identically zero, so b_0 does not exist.
        return std::numeric_limits<double>::quiet_NaN();
    }
    const MathieuKind kind = mathieu_kind(n, true);
    const int index = (n % 2 == 1) ? (n - 1) / 2 : (n - 2) / 2;
    return mathieu_characteristic(kind, q, index);
}

double mathieu_ce(int n, double q, double x) {
    if (n < 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const MathieuKind kind = mathieu_kind(n, false);
    return mathieu_series_value(n, kind, q, x, mathieu_a(n, q));
}

double mathieu_se(int n, double q, double x) {
    if (n < 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (n == 0) {
        return 0.0;  // se_0 is identically zero
    }
    const MathieuKind kind = mathieu_kind(n, true);
    return mathieu_series_value(n, kind, q, x, mathieu_b(n, q));
}

double mathieu_mc(int n, double q, double x) {
    if (n < 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const MathieuKind kind = mathieu_kind(n, false);
    return mathieu_modified_value(n, kind, q, x, mathieu_a(n, q));
}

double mathieu_ms(int n, double q, double x) {
    if (n < 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (n == 0) {
        return 0.0;
    }
    const MathieuKind kind = mathieu_kind(n, true);
    return mathieu_modified_value(n, kind, q, x, mathieu_b(n, q));
}

// ---------------------------------------------------------------------------------------
// Prolate spheroidal angular functions.
//
// The angular equation is
//     d/dx[(1-x^2) dS/dx] + [lambda - c^2 x^2 - m^2/(1-x^2)] S = 0,   -1 < x < 1,
// i.e. the eigenproblem   L S = lambda S   with   L = -d/dx[(1-x^2) d/dx] + m^2/(1-x^2) + c^2 x^2.
// Expanding S in the ORTHONORMAL Ferrers basis e_l = P_l^m / sqrt(h_l), h_l = 2 (l+m)!/((2l+1)(l-m)!),
// the first two pieces of L are diagonal with entries l(l+1), and the last is c^2 X^2 where X is
// the (symmetric, tridiagonal) matrix of multiplication by x, whose only non-zero entries follow
// from the Legendre recurrence x P_l^m = [(l-m+1)P_{l+1}^m + (l+m)P_{l-1}^m]/(2l+1):
//     X_{l,l+1} = sqrt( (l-m+1)(l+m+1) / ((2l+1)(2l+3)) ).
// X^2 couples only l and l +/- 2, so the problem splits by the parity of l - m and each parity
// class is a symmetric TRIDIAGONAL matrix:
//     diag_j    = l_j(l_j+1) + c^2 (X_{l_j,l_j-1}^2 + X_{l_j,l_j+1}^2)
//     offdiag_j = c^2 X_{l_j,l_j+1} X_{l_j+1,l_j+2},         l_j = m + parity + 2j.
// lambda_mn(c) is its (n-m)/2-th eigenvalue in ascending order -- exactly n(n+1) at c = 0, and
// n(n+1) + c^2 <x^2>_{n,m} + O(c^4) for small c, both of which the regression tests pin.
//
// This replaces `n(n+1) - c^2 + 3.0986774 c - 0.0146127`, a magic-number curve fit that ignored
// m entirely and got even the exact c = 0 limit wrong (5.985387 instead of 6), and it replaces
// the constant `pow(1-x^2, m/2) * 0.043` that stood in for S1.
constexpr int kSpheroidalExtra = 22;

double spheroidal_x_offdiag(int l, int m) {
    const double ld = static_cast<double>(l);
    const double md = static_cast<double>(m);
    if (l < m) {
        return 0.0;
    }
    return std::sqrt((ld - md + 1.0) * (ld + md + 1.0) / ((2.0 * ld + 1.0) * (2.0 * ld + 3.0)));
}

struct SpheroidalSolution {
    double lambda;
    std::vector<double> d;  // coefficients of P_{m+parity+2j}^m
    int parity;
    bool valid;
};

SpheroidalSolution spheroidal_solve(int n, int m, double c) {
    SpheroidalSolution out{std::numeric_limits<double>::quiet_NaN(), {}, 0, false};
    if (n < m || m < 0) {
        return out;
    }
    const int parity = (n - m) % 2;
    const int index = (n - m) / 2;
    const int dim = index + kSpheroidalExtra + static_cast<int>(std::ceil(std::abs(c)));
    const double c2 = c * c;
    std::vector<double> diag(static_cast<std::size_t>(dim));
    std::vector<double> off(static_cast<std::size_t>(dim - 1));
    for (int j = 0; j < dim; ++j) {
        const int l = m + parity + 2 * j;
        const double ld = static_cast<double>(l);
        const double lower = (l - 1 >= m) ? spheroidal_x_offdiag(l - 1, m) : 0.0;
        const double upper = spheroidal_x_offdiag(l, m);
        diag[static_cast<std::size_t>(j)] = ld * (ld + 1.0) + c2 * (lower * lower + upper * upper);
        if (j + 1 < dim) {
            off[static_cast<std::size_t>(j)] =
                c2 * spheroidal_x_offdiag(l, m) * spheroidal_x_offdiag(l + 1, m);
        }
    }
    if (index >= dim) {
        return out;
    }
    const double lambda = tridiagonal_eigenvalue(diag, off, index);
    if (!std::isfinite(lambda)) {
        return out;
    }

    // Eigenvector of the symmetric tridiagonal matrix at the known eigenvalue, built from both
    // ends towards the dominant component (the standard stable construction: the forward
    // recurrence alone is unstable in the decaying tail and vice versa).
    std::vector<double> v(static_cast<std::size_t>(dim), 0.0);
    v[static_cast<std::size_t>(index)] = 1.0;
    std::vector<double> tail(static_cast<std::size_t>(dim), 0.0);
    for (int j = dim - 2; j >= index; --j) {
        const double denom = diag[static_cast<std::size_t>(j + 1)] - lambda +
                             ((j + 1 < dim - 1) ? off[static_cast<std::size_t>(j + 1)] *
                                                      tail[static_cast<std::size_t>(j + 1)]
                                                : 0.0);
        tail[static_cast<std::size_t>(j)] =
            (denom == 0.0) ? 0.0 : -off[static_cast<std::size_t>(j)] / denom;
    }
    for (int j = index; j + 1 < dim; ++j) {
        v[static_cast<std::size_t>(j + 1)] =
            v[static_cast<std::size_t>(j)] * tail[static_cast<std::size_t>(j)];
    }
    std::vector<double> head(static_cast<std::size_t>(dim), 0.0);  // v_{j-1}/v_j for j <= index
    for (int j = 1; j <= index; ++j) {
        const double denom = diag[static_cast<std::size_t>(j - 1)] - lambda +
                             ((j >= 2) ? off[static_cast<std::size_t>(j - 2)] *
                                             head[static_cast<std::size_t>(j - 1)]
                                       : 0.0);
        head[static_cast<std::size_t>(j)] =
            (denom == 0.0) ? 0.0 : -off[static_cast<std::size_t>(j - 1)] / denom;
    }
    for (int j = index; j >= 1; --j) {
        v[static_cast<std::size_t>(j - 1)] =
            v[static_cast<std::size_t>(j)] * head[static_cast<std::size_t>(j)];
    }

    // Back to the un-normalised Ferrers basis: d_r = v_j / sqrt(h_{l_j}).
    std::vector<double> d(static_cast<std::size_t>(dim), 0.0);
    for (int j = 0; j < dim; ++j) {
        const int l = m + parity + 2 * j;
        const double log_h = std::log(2.0) + std::lgamma(static_cast<double>(l + m) + 1.0) -
                             std::log(2.0 * static_cast<double>(l) + 1.0) -
                             std::lgamma(static_cast<double>(l - m) + 1.0);
        d[static_cast<std::size_t>(j)] = v[static_cast<std::size_t>(j)] * std::exp(-0.5 * log_h);
    }

    // Flammer normalisation: S_mn(c,x) -> P_n^m(x) as c -> 0, imposed by matching the value at
    // x = 0 when n-m is even and the derivative there when it is odd.  (P_l^m)'(0) = (l+m) P_{l-1}^m(0)
    // follows from (1-x^2) dP_l^m/dx = (l+m) P_{l-1}^m - l x P_l^m at x = 0.
    double raw = 0.0;
    double target = 0.0;
    if (parity == 0) {
        for (int j = 0; j < dim; ++j) {
            raw += d[static_cast<std::size_t>(j)] * legendre_pn(m + 2 * j, m, 0.0);
        }
        target = legendre_pn(n, m, 0.0);
    } else {
        for (int j = 0; j < dim; ++j) {
            const int l = m + 1 + 2 * j;
            raw += d[static_cast<std::size_t>(j)] * static_cast<double>(l + m) *
                   legendre_pn(l - 1, m, 0.0);
        }
        target = static_cast<double>(n + m) * legendre_pn(n - 1, m, 0.0);
    }
    if (std::abs(raw) > 1e-280 && std::isfinite(target)) {
        const double scale = target / raw;
        for (int j = 0; j < dim; ++j) {
            d[static_cast<std::size_t>(j)] *= scale;
        }
    }
    out.lambda = lambda;
    out.d = d;
    out.parity = parity;
    out.valid = true;
    return out;
}

double spheroidal_lambda(int n, int m, double c) {
    const SpheroidalSolution solution = spheroidal_solve(n, m, c);
    return solution.valid ? solution.lambda : std::numeric_limits<double>::quiet_NaN();
}

double spheroidal_s1(int n, int m, double c, double x) {
    if (n < m || m < 0 || std::abs(x) > 1.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const SpheroidalSolution solution = spheroidal_solve(n, m, c);
    if (!solution.valid) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double sum = 0.0;
    for (std::size_t j = 0; j < solution.d.size(); ++j) {
        const int l = m + solution.parity + 2 * static_cast<int>(j);
        sum += solution.d[j] * legendre_pn(l, m, x);
    }
    return sum;
}

// Second solution of the SAME angular equation, defined by the two conditions that pin it
// uniquely and reduce it to the Ferrers function Q_n^m(x) at c = 0:
//   * opposite parity to S1 (the equation is even in x, so its solutions split by parity, and
//     Q_n^m has the parity opposite to P_n^m);
//   * the Legendre-matched Wronskian  (1-x^2)(S1 S2' - S1' S2) = (-1)^m (n+m)!/(n-m)!.
// Those data are imposed at x = 0 and the equation
//   (1-x^2) S'' - 2 x S' + (lambda - c^2 x^2 - m^2/(1-x^2)) S = 0
// is integrated outward with RK4.  (Expanding S2 on the Q_{m+r}^m basis with the SAME
// coefficients does NOT work: the three-term recurrence the Ferrers Q satisfy is inhomogeneous
// at the bottom of the ladder -- Q_1 = x Q_0 - 1, not x Q_0 -- so such a sum fails the ODE.)
//
// The equation degenerates at x = +/- 1, so the integration is only carried out on |x| < 1 and
// the accuracy falls off as |x| approaches 1 (S2 itself diverges there for m >= 1).
double spheroidal_s2(int n, int m, double c, double x) {
    if (n < m || m < 0 || std::abs(x) >= 1.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const SpheroidalSolution solution = spheroidal_solve(n, m, c);
    if (!solution.valid) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double value_at_zero = 0.0;
    double slope_at_zero = 0.0;
    for (std::size_t j = 0; j < solution.d.size(); ++j) {
        const int l = m + solution.parity + 2 * static_cast<int>(j);
        value_at_zero += solution.d[j] * legendre_pn(l, m, 0.0);
        slope_at_zero += solution.d[j] * static_cast<double>(l + m) * legendre_pn(l - 1, m, 0.0);
    }
    const double sign = (m % 2 == 0) ? 1.0 : -1.0;
    const double wronskian = sign * std::exp(std::lgamma(static_cast<double>(n + m) + 1.0) -
                                             std::lgamma(static_cast<double>(n - m) + 1.0));
    double y = 0.0;
    double yp = 0.0;
    if (solution.parity == 0) {
        if (std::abs(value_at_zero) <= 1e-280) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        y = 0.0;
        yp = wronskian / value_at_zero;
    } else {
        if (std::abs(slope_at_zero) <= 1e-280) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        y = -wronskian / slope_at_zero;
        yp = 0.0;
    }
    if (x == 0.0) {
        return y;
    }
    const double lambda = solution.lambda;
    const double m2 = static_cast<double>(m) * static_cast<double>(m);
    const auto accel = [&](double t, double yy, double yyp) {
        const double one_minus = 1.0 - t * t;
        return (2.0 * t * yyp - (lambda - c * c * t * t - m2 / one_minus) * yy) / one_minus;
    };
    const int steps = 3072;
    const double h = x / static_cast<double>(steps);
    for (int i = 0; i < steps; ++i) {
        const double t = h * static_cast<double>(i);
        const double k1 = yp;
        const double l1 = accel(t, y, yp);
        const double k2 = yp + 0.5 * h * l1;
        const double l2 = accel(t + 0.5 * h, y + 0.5 * h * k1, yp + 0.5 * h * l1);
        const double k3 = yp + 0.5 * h * l2;
        const double l3 = accel(t + 0.5 * h, y + 0.5 * h * k2, yp + 0.5 * h * l2);
        const double k4 = yp + h * l3;
        const double l4 = accel(t + h, y + h * k3, yp + h * l3);
        y += h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
        yp += h * (l1 + 2.0 * l2 + 2.0 * l3 + l4) / 6.0;
    }
    return y;
}

// ---------------------------------------------------------------------------------------
// Parabolic cylinder functions.
//
// D_nu(x) is a linear combination of TWO confluent hypergeometric series (DLMF 12.4.1-12.4.2 /
// Whittaker's form), not one:
//   D_nu(x) = 2^{nu/2} e^{-x^2/4} [ sqrt(pi)/Gamma((1-nu)/2) M(-nu/2, 1/2, x^2/2)
//                                   - sqrt(2 pi) x/Gamma(-nu/2) M((1-nu)/2, 3/2, x^2/2) ].
// The old body summed a single series with prefactor 2^{-nu/2} -- it did not even reproduce
// D_0(x) = exp(-x^2/4), and it disagreed with its own x == 0 special case (which used the
// correct 2^{+nu/2}), so the function jumped at the origin.
//
// For x > kPcfSeriesX the two M's are each O(e^{x^2/2}) while D_nu is O(x^nu e^{-x^2/4}), so
// they cancel to ~e^{x^2/2}; the large-argument expansion
//   D_nu(x) ~ x^nu e^{-x^2/4} sum_k (-1)^k (nu)(nu-1)...(nu-2k+1) / (k! 2^k x^{2k})
// (DLMF 12.9.1), truncated at its smallest term, takes over there.  For x < 0 the same two
// series carry no cancellation at all (D_nu grows like e^{x^2/4} there, the same size as the
// individual terms), so they are used for every negative argument.
constexpr double kPcfSeriesX = 6.0;

double parabolic_d_asymptotic(double nu, double x) {
    double term = 1.0;
    double sum = 1.0;
    double previous = std::numeric_limits<double>::infinity();
    for (int k = 1; k < 200; ++k) {
        const double kd = static_cast<double>(k);
        term *= -(nu - 2.0 * kd + 2.0) * (nu - 2.0 * kd + 1.0) / (2.0 * kd * x * x);
        if (!std::isfinite(term) || std::abs(term) >= previous) {
            break;
        }
        previous = std::abs(term);
        sum += term;
        if (previous <= 1e-18 * std::abs(sum)) {
            break;
        }
    }
    return std::pow(x, nu) * std::exp(-0.25 * x * x) * sum;
}

double parabolic_d(double nu, double x) {
    if (x > 0.0) {
        // D_nu(x) = 2^{nu/2} e^{-x^2/4} U(-nu/2, 1/2, x^2/2) (DLMF 12.7.14).  The Tricomi U above
        // is computed by a cancellation-free quadrature (and by its own optimally truncated
        // asymptotic expansion for large argument), so this route keeps full precision all along
        // the positive axis, where the two-series form below would cancel to ~e^{x^2/2}.
        const double u = tricomi_u_impl(-0.5 * nu, 0.5, 0.5 * x * x);
        if (std::isfinite(u)) {
            return std::pow(2.0, 0.5 * nu) * std::exp(-0.25 * x * x) * u;
        }
        if (x > kPcfSeriesX) {
            return parabolic_d_asymptotic(nu, x);
        }
    }
    const double even = kummer_m_series(-0.5 * nu, 0.5, 0.5 * x * x);
    const double odd = kummer_m_series(0.5 * (1.0 - nu), 1.5, 0.5 * x * x);
    const double c_even = std::sqrt(M_PI) * rgamma(0.5 * (1.0 - nu));
    const double c_odd = std::sqrt(2.0 * M_PI) * rgamma(-0.5 * nu);
    return std::pow(2.0, 0.5 * nu) * std::exp(-0.25 * x * x) *
           (c_even * even - c_odd * x * odd);
}

// log Gamma for complex argument (Stirling with upward shifting), used only for the modulus
// constants G_1 = |Gamma(1/4 + i a/2)| and G_3 = |Gamma(3/4 + i a/2)| of DLMF 12.14.
std::complex<double> complex_log_gamma(std::complex<double> z) {
    std::complex<double> accumulator(0.0, 0.0);
    while (std::abs(z) < 12.0) {
        accumulator -= std::log(z);
        z += 1.0;
    }
    const std::complex<double> inv = 1.0 / z;
    const std::complex<double> inv2 = inv * inv;
    std::complex<double> series = inv * (1.0 / 12.0);
    std::complex<double> power = inv * inv2;
    series += power * (-1.0 / 360.0);
    power *= inv2;
    series += power * (1.0 / 1260.0);
    power *= inv2;
    series += power * (-1.0 / 1680.0);
    power *= inv2;
    series += power * (1.0 / 1188.0);
    return accumulator + (z - 0.5) * std::log(z) - z + 0.5 * std::log(2.0 * M_PI) + series;
}

double gamma_modulus(double real_part, double imaginary_part) {
    return std::exp(complex_log_gamma(std::complex<double>(real_part, imaginary_part)).real());
}

// The two even/odd solutions of y'' + (x^2/4 - a) y = 0 with w1(0)=1, w1'(0)=0 and w2(0)=0,
// w2'(0)=1, from the power-series recurrence c_{k+2} = (a c_k - c_{k-2}/4)/((k+2)(k+1)).
void pcf_w_solutions(double a, double x, double& w1, double& w2) {
    constexpr int kTerms = 220;
    double c1[kTerms];
    double c2[kTerms];
    for (int k = 0; k < kTerms; ++k) {
        c1[k] = 0.0;
        c2[k] = 0.0;
    }
    c1[0] = 1.0;
    c2[1] = 1.0;
    for (int k = 0; k + 2 < kTerms; ++k) {
        const double denom = static_cast<double>(k + 2) * static_cast<double>(k + 1);
        const double back1 = (k >= 2) ? c1[k - 2] : 0.0;
        const double back2 = (k >= 2) ? c2[k - 2] : 0.0;
        c1[k + 2] = (a * c1[k] - 0.25 * back1) / denom;
        c2[k + 2] = (a * c2[k] - 0.25 * back2) / denom;
    }
    w1 = 0.0;
    w2 = 0.0;
    double power = 1.0;
    for (int k = 0; k < kTerms; ++k) {
        w1 += c1[k] * power;
        w2 += c2[k] * power;
        power *= x;
    }
}

double pcf_u(double a, double x) {
    // U(a,x) = D_{-a-1/2}(x) exactly (DLMF 12.7.10) -- no extra prefactor.  The old body
    // multiplied by a spurious 2^{-a-1/2} e^{-x^2/4} on top of parabolic_d's own exponential.
    return parabolic_d(-a - 0.5, x);
}

double pcf_v(double a, double x) {
    // DLMF 12.2.20: V(a,x) = (Gamma(1/2+a)/pi) (sin(pi a) U(a,x) + U(a,-x)).
    // At a = -1/2, -3/2, ... the Gamma has a simple pole while the bracket vanishes identically,
    // so the product is a removable 0 * infinity.  V is analytic in a, so those points are
    // filled in by averaging two nearby values (error O(eps^2 d^2V/da^2) ~ 1e-9).
    const double shifted = 0.5 + a;
    if (shifted <= 0.0 && std::abs(shifted - std::round(shifted)) < 1e-8) {
        constexpr double eps = 1e-4;
        return 0.5 * (pcf_v(a - eps, x) + pcf_v(a + eps, x));
    }
    const double scale = std::tgamma(shifted) / M_PI;
    return scale * (std::sin(M_PI * a) * pcf_u(a, x) + pcf_u(a, -x));
}

double pcf_w(double a, double x) {
    // W(a,x) is the OSCILLATORY parabolic cylinder function, the standard real solution pair of
    // y'' + (x^2/4 - a) y = 0 (DLMF 12.14.1).  It is not any combination of U and V with real
    // coefficients, so the previous U cos(pi a) - V sin(pi a) mix was a different function.
    // DLMF 12.14.4:  W(a,x) = 2^{-3/4} ( sqrt(G1/G3) w1(a,x) - sqrt(2 G3/G1) w2(a,x) ),
    //   G1 = |Gamma(1/4 + i a/2)|,  G3 = |Gamma(3/4 + i a/2)|.
    const double g1 = gamma_modulus(0.25, 0.5 * a);
    const double g3 = gamma_modulus(0.75, 0.5 * a);
    double w1 = 0.0;
    double w2 = 0.0;
    pcf_w_solutions(a, x, w1, w2);
    return std::pow(2.0, -0.75) * (std::sqrt(g1 / g3) * w1 - std::sqrt(2.0 * g3 / g1) * w2);
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

// Same RK4 driver, but able to integrate in either direction from x0 (used by heun_d, whose
// natural base point sits inside the domain).
double integrate_second_order_signed(double x0, double y0, double yp0, double x_end, int steps,
                                     const std::function<double(double, double, double)>& accel) {
    if (x_end == x0) {
        return y0;
    }
    const double h = (x_end - x0) / static_cast<double>(steps);
    double y = y0;
    double yp = yp0;
    for (int i = 0; i < steps; ++i) {
        const double t = x0 + h * static_cast<double>(i);
        const double k1 = yp;
        const double l1 = accel(t, y, yp);
        const double k2 = yp + 0.5 * h * l1;
        const double l2 = accel(t + 0.5 * h, y + 0.5 * h * k1, yp + 0.5 * h * l1);
        const double k3 = yp + 0.5 * h * l2;
        const double l3 = accel(t + 0.5 * h, y + 0.5 * h * k2, yp + 0.5 * h * l2);
        const double k4 = yp + h * l3;
        const double l4 = accel(t + h, y + h * k3, yp + h * l3);
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

// ---------------------------------------------------------------------------------------
// Heun equations, transcribed from DLMF 31.1.1 and 31.12.1-31.12.4.  Each member is integrated
// with the RK4 driver above from the base point z0 noted per function.
//
// Every one of these five bodies previously solved an equation outside the Heun family: heun_g
// omitted the epsilon/(z-a) pole that the Fuchs relation fixes, and heun_d / heun_b / heun_t --
// the CONFLUENT members, which are defined by the coalescence that removes the z = 1
// singularity -- each retained a spurious 1/(z-1) pole and dropped the polynomial y'
// coefficients that define them.  heun_c dropped the constant epsilon of DLMF 31.12.1.
//
// Parameter mapping where the shipped signature has fewer or differently named slots than DLMF:
//   heun_c(q, alpha, beta, gamma, delta, z): DLMF's CHE has (q, alpha, gamma, delta, epsilon);
//       `beta` carries epsilon.
//   heun_b(q, alpha, beta, delta, z): DLMF's BHE has (q, alpha, beta, gamma, delta); `q` carries
//       gamma and the accessory parameter enters through delta.
//   heun_t(q, alpha, beta, gamma, z): DLMF's THE has only (q, alpha, gamma); `beta` is accepted
//       for signature symmetry with the other members and does not appear in the equation.
constexpr double kHeunBase = 1e-6;

double heun_g(double a, double q, double alpha, double beta, double gamma, double delta, double z) {
    if (z <= kHeunBase) {
        return 1.0;
    }
    // The general Heun equation has regular singular points at 0, 1 and a.  The solution being
    // integrated is the one normalised at z = 0, so it is only defined up to the first
    // singularity to the right; continuing THROUGH one (as the old clamp_heun_z hack did) picks
    // an arbitrary branch and returns noise.  Report that honestly instead.
    const double nearest = (a > 0.0) ? std::min(1.0, a) : 1.0;
    if (z >= nearest - 1e-9) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    // Fuchs relation for the general Heun equation (DLMF 31.1.2).
    const double epsilon = alpha + beta - gamma - delta + 1.0;
    const auto accel = [&](double zz, double y, double yp) {
        zz = clamp_heun_z(zz, a);
        const double coeff = gamma / zz + delta / (zz - 1.0) + epsilon / (zz - a);
        const double forcing = (alpha * beta * zz - q) / (zz * (zz - 1.0) * (zz - a));
        return -coeff * yp - forcing * y;
    };
    // Frobenius exponent-zero solution at the regular singular point z = 0: y = 1 + (q/(a gamma)) z
    const double c1 = (gamma * a == 0.0) ? 0.0 : q / (a * gamma);
    return integrate_second_order(kHeunBase, 1.0 + c1 * kHeunBase, c1, z, 8192, accel);
}

double heun_c(double q, double alpha, double beta, double gamma, double delta, double z) {
    if (z <= kHeunBase) {
        return 1.0;
    }
    if (z >= 1.0 - 1e-9) {  // regular singular point at z = 1; see heun_g
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double epsilon = beta;  // see the parameter-mapping note above
    const auto accel = [&](double zz, double y, double yp) {
        zz = clamp_heun_z(zz, 0.0);
        const double coeff = gamma / zz + delta / (zz - 1.0) + epsilon;
        const double forcing = (alpha * zz - q) / (zz * (zz - 1.0));
        return -coeff * yp - forcing * y;
    };
    const double c1 = (gamma == 0.0) ? 0.0 : -q / gamma;
    return integrate_second_order(kHeunBase, 1.0 + c1 * kHeunBase, c1, z, 8192, accel);
}

double heun_d(double q, double alpha, double gamma, double delta, double z) {
    // Doubly confluent Heun, DLMF 31.12.2:
    //   z^2 y'' + (-z^2 + delta z + gamma) y' + (alpha z - q) y = 0.
    // z = 0 is an IRREGULAR singular point, so there is no Frobenius solution to start from
    // there; the base point is the ordinary point z = 1 with y(1) = 1, y'(1) = 0, and the
    // integration runs in whichever direction reaches z.
    if (z <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const auto accel = [&](double zz, double y, double yp) {
        const double t = (std::abs(zz) < 1e-8) ? 1e-8 : zz;
        const double coeff = -1.0 + delta / t + gamma / (t * t);
        const double forcing = (alpha * t - q) / (t * t);
        return -coeff * yp - forcing * y;
    };
    return integrate_second_order_signed(1.0, 1.0, 0.0, z, 8192, accel);
}

double heun_b(double q, double alpha, double beta, double delta, double z) {
    // Biconfluent Heun, DLMF 31.12.3 (with gamma carried by q):
    //   z y'' + (1 + alpha - beta z - 2 z^2) y' + ((q - alpha - 2) z - (delta + (1+alpha) beta)/2) y = 0
    if (z <= kHeunBase) {
        return 1.0;
    }
    const auto accel = [&](double zz, double y, double yp) {
        const double t = (zz < kHeunBase) ? kHeunBase : zz;
        const double coeff = (1.0 + alpha - beta * t - 2.0 * t * t) / t;
        const double forcing = ((q - alpha - 2.0) * t - 0.5 * (delta + (1.0 + alpha) * beta)) / t;
        return -coeff * yp - forcing * y;
    };
    const double c1 = (1.0 + alpha == 0.0)
                          ? 0.0
                          : 0.5 * (delta + (1.0 + alpha) * beta) / (1.0 + alpha);
    return integrate_second_order(kHeunBase, 1.0 + c1 * kHeunBase, c1, z, 8192, accel);
}

double heun_t(double q, double alpha, double beta, double gamma, double z) {
    // Triconfluent Heun, DLMF 31.12.4:  y'' - (gamma + 3 z^2) y' + (alpha z - q) y = 0.
    // z = 0 is an ORDINARY point, so y(0) = 1, y'(0) = 0 is a genuine normalisation and the
    // integration starts exactly there.  `beta` does not appear in the equation.
    (void)beta;
    if (z <= kHeunBase) {
        return 1.0;
    }
    const auto accel = [&](double zz, double y, double yp) {
        return (gamma + 3.0 * zz * zz) * yp - (alpha * zz - q) * y;
    };
    return integrate_second_order(0.0, 1.0, 0.0, z, 8192, accel);
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

// ---------------------------------------------------------------------------------------
// Painleve transcendents III-VI, transcribed from DLMF 32.2.3-32.2.6.  Each is integrated from
// the base point x0 named below with the caller's data (y0, yp0); x <= x0 returns y0.
//
// The previous right-hand sides were not these equations at all: PIII put (alpha w^2 + beta)
// over z^2 instead of z and dropped gamma w^3 + delta/w entirely; PIV used coefficient 1 rather
// than 1/2 on w'^2, added a -w'/(2z) term PIV does not contain, and had none of (3/2)w^3,
// 4 z w^2 or 2(z^2 - alpha) w; PV and PVI omitted the pole structures 1/(2w) + 1/(w-1) and
// (1/2)(1/w + 1/(w-1) + 1/(w-z)) that define them.
//
// PIII carries four parameters (alpha, beta, gamma, delta) but the shipped signature has two;
// this implementation fixes gamma = 1, delta = -1, the standard generic normalisation
// PIII(alpha, beta, 1, -1) to which every non-degenerate PIII can be scaled.
double painleve3(double x, double y0, double yp0, double alpha, double beta) {
    const double x0 = 0.1;
    if (x <= x0) {
        return y0;
    }
    const auto accel = [alpha, beta](double xx, double y, double yp) {
        y = safe_y(y);
        constexpr double gamma = 1.0;
        constexpr double delta = -1.0;
        return yp * yp / y - yp / xx + (alpha * y * y + beta) / xx + gamma * y * y * y + delta / y;
    };
    return integrate_second_order(x0, y0, yp0, x, 4096, accel);
}

double painleve4(double x, double y0, double yp0, double alpha, double beta) {
    const double x0 = 0.1;
    if (x <= x0) {
        return y0;
    }
    const auto accel = [alpha, beta](double xx, double y, double yp) {
        y = safe_y(y);
        return 0.5 * yp * yp / y + 1.5 * y * y * y + 4.0 * xx * y * y +
               2.0 * (xx * xx - alpha) * y + beta / y;
    };
    return integrate_second_order(x0, y0, yp0, x, 4096, accel);
}

double painleve5(double x, double y0, double yp0, double alpha, double beta, double gamma, double delta) {
    const double x0 = 0.2;
    if (x <= x0) {
        return y0;
    }
    const auto accel = [alpha, beta, gamma, delta](double xx, double y, double yp) {
        y = safe_y(y);
        const double ym1 = safe_y(y - 1.0);
        return (0.5 / y + 1.0 / ym1) * yp * yp - yp / xx +
               ym1 * ym1 / (xx * xx) * (alpha * y + beta / y) + gamma * y / xx +
               delta * y * (y + 1.0) / ym1;
    };
    return integrate_second_order(x0, y0, yp0, x, 4096, accel);
}

double painleve6(double x, double y0, double yp0, double alpha, double beta, double gamma, double delta) {
    const double x0 = 2.1;
    if (x <= x0) {
        return y0;
    }
    const auto accel = [alpha, beta, gamma, delta](double xx, double y, double yp) {
        y = safe_y(y);
        const double ym1 = safe_y(y - 1.0);
        const double ymz = safe_y(y - xx);
        const double zm1 = safe_y(xx - 1.0);
        const double quad = 0.5 * (1.0 / y + 1.0 / ym1 + 1.0 / ymz) * yp * yp;
        const double linear = (1.0 / xx + 1.0 / zm1 + 1.0 / ymz) * yp;
        const double bracket = alpha + beta * xx / (y * y) + gamma * (xx - 1.0) / (ym1 * ym1) +
                               delta * xx * (xx - 1.0) / (ymz * ymz);
        const double prefactor = y * ym1 * ymz / (xx * xx * zm1 * zm1);
        return quad - linear + prefactor * bracket;
    };
    return integrate_second_order(x0, y0, yp0, x, 8192, accel);
}

} // namespace ms
