// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
// Regression tests for the audited defects in the `special` module.
//
// Every test here FAILS against the pre-fix implementation.  Where a defect had a closed form,
// a defining identity or an ODE it must satisfy, the identity is asserted rather than (or as
// well as) a pinned number, so the test constrains the contract and not just one sample.
//
// The pinned reference values were each derived independently of this library and the
// derivation is named in the comment above the assertion.

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <vector>

#include "ms/special/special.hpp"

using namespace ms;

namespace {

constexpr double kPi = 3.14159265358979323846;

// Central second derivative, used for ODE-residual checks.
template <typename F>
double d2(F f, double x, double h) {
    return (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h);
}
template <typename F>
double d1(F f, double x, double h) {
    return (f(x + h) - f(x - h)) / (2.0 * h);
}

}  // namespace

// ---------------------------------------------------------------------------------------
// [CRITICAL] bessel_k: the upward recurrence had the wrong sign AND half the coefficient, so
// every order >= 2 was wrong and often negative.  DLMF 10.29.1: K_{n+1} = K_{n-1} + (2n/x) K_n.
// ---------------------------------------------------------------------------------------
TEST(SpecialAuditFixes, BesselK_HighOrderIsPositiveAndCorrect) {
    // Reference values: the A&S 9.6.11 ascending log-series summed in long double.
    EXPECT_NEAR(bessel_k(2, 1.0), 1.6248388986351774, 1e-13);
    EXPECT_NEAR(bessel_k(3, 1.0), 7.1012628247379448, 1e-12);
    EXPECT_NEAR(bessel_k(5, 2.0), 9.4310491005964678, 1e-12);
    EXPECT_NEAR(bessel_k(4, 0.5), 752.24509791040396, 1e-9);

    // K_nu(x) is strictly positive for every x > 0 -- the old code returned -0.180883 at (2, 1).
    for (int nu = 0; nu <= 25; ++nu) {
        for (double x : {0.25, 1.0, 4.0, 20.0}) {
            EXPECT_GT(bessel_k(nu, x), 0.0) << "K_" << nu << "(" << x << ")";
        }
    }
    // The defining recurrence itself.
    for (double x : {0.3, 1.0, 5.0, 15.0}) {
        for (int n = 1; n < 20; ++n) {
            const double lhs = bessel_k(n + 1, x);
            const double rhs = bessel_k(n - 1, x) + (2.0 * n / x) * bessel_k(n, x);
            EXPECT_NEAR(lhs, rhs, 1e-11 * std::abs(rhs)) << "n=" << n << " x=" << x;
        }
    }
}

// ---------------------------------------------------------------------------------------
// [CRITICAL] bessel_i: the recurrence added where DLMF 10.29.1 subtracts, and the corrected
// upward recurrence would still be the unstable direction for I (the minimal solution).
// ---------------------------------------------------------------------------------------
TEST(SpecialAuditFixes, BesselI_HighOrderIsCorrect) {
    // Reference values: the ascending series sum_{k} (x/2)^{2k+nu}/(k!(nu+k)!) in long double.
    EXPECT_NEAR(bessel_i(2, 1.0), 0.13574766976703828, 1e-15);
    EXPECT_NEAR(bessel_i(5, 1.0), 2.7146315595697189e-04, 1e-18);
    EXPECT_NEAR(bessel_i(3, 5.0), 10.331150169151138, 1e-13);

    // I_n(x) decreases with n and stays positive -- the old code gave 516.555 for I_5(1).
    for (double x : {0.5, 2.0, 8.0}) {
        for (int n = 0; n < 20; ++n) {
            EXPECT_GT(bessel_i(n, x), 0.0);
            EXPECT_LT(bessel_i(n + 1, x), bessel_i(n, x));
        }
    }
    // Wronskian I_n(x)K_{n+1}(x) + I_{n+1}(x)K_n(x) = 1/x (DLMF 10.28.2).
    for (double x : {0.5, 1.0, 3.0, 8.0, 20.0}) {
        for (int n : {0, 1, 4, 10, 20}) {
            const double w = bessel_i(n, x) * bessel_k(n + 1, x) + bessel_i(n + 1, x) * bessel_k(n, x);
            EXPECT_NEAR(w, 1.0 / x, 1e-12 / x) << "n=" << n << " x=" << x;
        }
    }
}

// ---------------------------------------------------------------------------------------
// [HIGH] bessel_j applied the upward recurrence unconditionally, which is exponentially
// unstable for n > x: J_20(1) came out as 3.17e5, violating |J_n| <= 1.
// ---------------------------------------------------------------------------------------
TEST(SpecialAuditFixes, BesselJ_AboveTurningPointAndLargeArgument) {
    // Reference: ascending series in long double (no cancellation at all when x < nu).
    EXPECT_NEAR(bessel_j(10, 1.0), 2.6306151236874534e-10, 1e-24);
    EXPECT_NEAR(bessel_j(20, 1.0), 3.8735030085246576e-25, 1e-38);
    EXPECT_NEAR(bessel_j(30, 3.0), 6.7223399381463311e-28, 1e-41);
    // Reference: Bessel's integral J_n(x) = (1/pi) int_0^pi cos(n t - x sin t) dt, trapezoid.
    EXPECT_NEAR(bessel_j(0, 50.0), 0.055812327669251816, 1e-13);

    // |J_n(x)| <= 1 for every real x and integer n.
    for (int n = 0; n <= 40; ++n) {
        for (double x : {0.5, 1.0, 5.0, 12.0, 25.0, 50.0}) {
            EXPECT_LE(std::abs(bessel_j(n, x)), 1.0) << "J_" << n << "(" << x << ")";
        }
    }
    // Wronskian J_{n+1}(x)Y_n(x) - J_n(x)Y_{n+1}(x) = 2/(pi x) (DLMF 10.5.2).
    for (double x : {0.5, 1.0, 3.0, 7.0, 20.0, 40.0}) {
        for (int n : {0, 1, 3, 6}) {
            const double w = bessel_j(n + 1, x) * bessel_y(n, x) - bessel_j(n, x) * bessel_y(n + 1, x);
            EXPECT_NEAR(w, 2.0 / (kPi * x), 1e-11 / x) << "n=" << n << " x=" << x;
        }
    }
}

// ---------------------------------------------------------------------------------------
// [HIGH] Airy: the x <= 4 branch was a least-squares polynomial fit covering all of x < 0
// (where Ai and Bi oscillate), and the x > 4 branch used the wrong prefactors and scaled the
// derivatives by 0.1x instead of sqrt(x).
// ---------------------------------------------------------------------------------------
TEST(SpecialAuditFixes, AiryReference) {
    // Exact constants: Ai(0) = 3^{-2/3}/Gamma(2/3), Ai'(0) = -3^{-1/3}/Gamma(1/3),
    // Bi(0) = sqrt(3) Ai(0), Bi'(0) = sqrt(3) |Ai'(0)|.
    EXPECT_NEAR(airy_ai(0.0), 0.3550280538878172, 1e-15);
    EXPECT_NEAR(airy_aip(0.0), -0.2588194037928068, 1e-15);
    EXPECT_NEAR(airy_bi(0.0), 0.6149266274460007, 1e-15);
    EXPECT_NEAR(airy_bip(0.0), 0.4482883573538264, 1e-15);
    // Reference: long-double Taylor march of y'' = x y from the exact origin data (Bi is the
    // dominant solution so the march is stable for it), and the DLMF 9.7 asymptotic expansion
    // in long double for Ai at large positive x.
    EXPECT_NEAR(airy_ai(1.0), 0.13529241631288140, 1e-15);
    EXPECT_NEAR(airy_ai(4.0), 9.5156385120480195e-04, 1e-17);
    EXPECT_NEAR(airy_ai(-5.0), 0.35076100902411150, 1e-13);
    EXPECT_NEAR(airy_ai(-10.0), 0.04024123848644319, 1e-13);
    EXPECT_NEAR(airy_bi(2.0), 3.2980949999782137, 1e-13);
    EXPECT_NEAR(airy_bi(10.0), 455641153.54822463, 1e-2);

    // The Airy Wronskian Ai(x)Bi'(x) - Ai'(x)Bi(x) = 1/pi holds for EVERY x.
    for (double x = -30.0; x <= 20.0; x += 0.25) {
        const double w = airy_ai(x) * airy_bip(x) - airy_aip(x) * airy_bi(x);
        EXPECT_NEAR(w, 1.0 / kPi, 1e-11) << "x=" << x;
    }
    // Ai/Bi oscillate and stay bounded for x < 0 -- the old fit gave Ai(-5) = -23.6.
    for (double x = -25.0; x < 0.0; x += 0.5) {
        EXPECT_LT(std::abs(airy_ai(x)), 1.0) << "x=" << x;
        EXPECT_LT(std::abs(airy_bi(x)), 1.0) << "x=" << x;
    }
    // ODE residual y'' = x y.
    for (double x : {-9.0, -3.0, -0.5, 0.5, 3.0, 9.0}) {
        const double h = 1e-3;
        EXPECT_NEAR(d2(airy_ai, x, h), x * airy_ai(x), 1e-6 * std::max(1.0, std::abs(x * airy_ai(x))));
        EXPECT_NEAR(d2(airy_bi, x, h), x * airy_bi(x), 1e-6 * std::max(1.0, std::abs(x * airy_bi(x))));
    }
}

// ---------------------------------------------------------------------------------------
// [HIGH] erfi dropped the (2n-1) factor from the series ratio and so summed
// x^{2n+1}/(n!(2n+1)!!) instead of x^{2n+1}/(n!(2n+1)).
// ---------------------------------------------------------------------------------------
TEST(SpecialAuditFixes, ErfiSeriesRatio) {
    // Reference: the documented series summed independently in long double.
    EXPECT_NEAR(erfi(0.5), 0.6149520946965109, 1e-15);
    EXPECT_NEAR(erfi(1.0), 1.6504257587975428, 1e-14);   // old: 1.543960 (6.5% low)
    EXPECT_NEAR(erfi(2.0), 18.564802414575552, 1e-13);   // old: ~64% low
    EXPECT_NEAR(erfi(3.0), 1629.9946226015664, 1e-10);
    // erfi'(x) = (2/sqrt(pi)) exp(x^2) -- the defining relation.
    for (double x : {0.25, 0.5, 1.0, 2.0, 3.0}) {
        const double h = 1e-5;
        const double numeric = d1([](double t) { return erfi(t); }, x, h);
        const double exact = 2.0 / std::sqrt(kPi) * std::exp(x * x);
        EXPECT_NEAR(numeric, exact, 1e-7 * exact) << "x=" << x;
    }
}

// ---------------------------------------------------------------------------------------
// [HIGH] legendre_q kept only the log term, dropping the subtracted polynomial W_{n-1}.
// ---------------------------------------------------------------------------------------
TEST(SpecialAuditFixes, LegendreQSecondKind) {
    // Reference: Q_n(x) = (1/2)P_n ln((1+x)/(1-x)) - sum_{k=1}^{n} P_{k-1}P_{n-k}/k, evaluated
    // with the module's (independently correct) legendre_p.
    for (int n = 0; n <= 8; ++n) {
        for (double x : {-0.9, -0.3, 0.2, 0.5, 0.85}) {
            double w = 0.0;
            for (int k = 1; k <= n; ++k) {
                w += legendre_p(k - 1, x) * legendre_p(n - k, x) / k;
            }
            const double expected = 0.5 * std::log((1.0 + x) / (1.0 - x)) * legendre_p(n, x) - w;
            EXPECT_NEAR(legendre_q(n, x), expected, 1e-12) << "n=" << n << " x=" << x;
        }
    }
    EXPECT_NEAR(legendre_q(1, 0.5), -0.7253469278329725, 1e-14);  // old: +0.274653 (wrong sign)
    EXPECT_NEAR(legendre_q(2, 0.5), -0.8186632680417568, 1e-14);  // old: -0.068663
    EXPECT_NEAR(legendre_q(1, 0.0), -1.0, 1e-14);                 // old: 0 for every n at x=0
    EXPECT_NEAR(legendre_q(3, 0.0), 2.0 / 3.0, 1e-14);
    // Legendre ODE (1-x^2)Q'' - 2xQ' + n(n+1)Q = 0.
    for (int n = 0; n <= 5; ++n) {
        const double x = 0.4;
        const double h = 1e-4;
        const auto q = [n](double t) { return legendre_q(n, t); };
        EXPECT_NEAR((1 - x * x) * d2(q, x, h) - 2 * x * d1(q, x, h) + n * (n + 1.0) * q(x), 0.0, 1e-6);
    }
}

// ---------------------------------------------------------------------------------------
// [HIGH] struve_l was a bare alias for struve_h, and struve_k used bessel_j where the
// definition K_nu = H_nu - Y_nu calls for bessel_y.
// ---------------------------------------------------------------------------------------
TEST(SpecialAuditFixes, StruveModifiedAndSecondKind) {
    // Reference: the non-alternating series sum_m (x/2)^{2m+nu+1}/(Gamma(m+3/2)Gamma(m+nu+3/2))
    // summed independently in long double.
    EXPECT_NEAR(struve_l(0, 1.0), 0.7102431859378909, 1e-14);
    EXPECT_NEAR(struve_l(1, 1.0), 0.2267643810558087, 1e-14);
    EXPECT_NEAR(struve_l(0, 3.0), 4.6486857317537100, 1e-13);
    // L is strictly larger than H for x > 0 (same terms, no sign alternation) -- the old code
    // returned exactly struve_h.
    for (double x : {0.5, 1.0, 3.0, 8.0}) {
        for (int nu = 0; nu <= 3; ++nu) {
            EXPECT_GT(struve_l(nu, x), struve_h(nu, x)) << "nu=" << nu << " x=" << x;
        }
    }
    // K_nu = H_nu - Y_nu by definition.
    for (double x : {0.4, 1.0, 5.0, 30.0}) {
        for (int nu = 0; nu <= 3; ++nu) {
            EXPECT_NEAR(struve_k(nu, x), struve_h(nu, x) - bessel_y(nu, x), 1e-9)
                << "nu=" << nu << " x=" << x;
        }
    }
    EXPECT_NEAR(struve_k(0, 1.0), 0.4803996628326109, 1e-12);  // old: -0.196541
    EXPECT_NEAR(struve_yn(0, 1.0), struve_k(0, 1.0), 0.0);
    // Struve's inhomogeneous ODE: x^2 H'' + x H' + (x^2 - nu^2) H = 4 (x/2)^{nu+1}/(sqrt(pi)Gamma(nu+1/2))
    // and its modified counterpart for L (with -(x^2 + nu^2)).
    for (int nu = 0; nu <= 2; ++nu) {
        for (double x : {0.8, 2.0, 6.0}) {
            const double h = 1e-3;
            const double rhs = 4.0 * std::pow(0.5 * x, nu + 1.0) / (std::sqrt(kPi) * std::tgamma(nu + 0.5));
            const auto hf = [nu](double t) { return struve_h(nu, t); };
            const auto lf = [nu](double t) { return struve_l(nu, t); };
            EXPECT_NEAR(x * x * d2(hf, x, h) + x * d1(hf, x, h) + (x * x - nu * nu) * hf(x), rhs,
                        1e-4 * std::abs(rhs) + 1e-8);
            EXPECT_NEAR(x * x * d2(lf, x, h) + x * d1(lf, x, h) - (x * x + nu * nu) * lf(x), rhs,
                        1e-4 * std::abs(rhs) + 1e-8);
        }
    }
}

// ---------------------------------------------------------------------------------------
// [HIGH] bessel_zero_jnu searched a fixed nu-independent bracket, returning negative values,
// the trivial zero at the origin, or the wrong index.
// ---------------------------------------------------------------------------------------
TEST(SpecialAuditFixes, BesselZerosAreTheNthPositiveZero) {
    // Reference values: A&S Table 9.5 / DLMF Table 10.6.1.
    struct Ref { int nu; int n; double value; };
    const Ref j_ref[] = {{0, 1, 2.404825557695773}, {0, 2, 5.520078110286311},
                         {1, 1, 3.831705970207512}, {1, 2, 7.015586669815619},
                         {2, 1, 5.135622301840683}, {2, 2, 8.417244140399864},
                         {3, 2, 9.761023129981670}, {5, 1, 8.771483815959954},
                         {10, 1, 14.475500686554541}};
    for (const Ref& r : j_ref) {
        EXPECT_NEAR(bessel_zero_jnu(r.nu, r.n), r.value, 1e-9)
            << "j_{" << r.nu << "," << r.n << "}";
    }
    const Ref y_ref[] = {{0, 1, 0.8935769662791675}, {0, 2, 3.9576784193148579},
                         {1, 1, 2.1971413260310170}, {1, 2, 5.4296810407941351},
                         {2, 1, 3.3842417671495935}};
    for (const Ref& r : y_ref) {
        EXPECT_NEAR(bessel_zero_ynu(r.nu, r.n), r.value, 1e-9)
            << "y_{" << r.nu << "," << r.n << "}";
    }
    // Structural contract: positive, increasing in n, and an actual root.
    for (int nu = 0; nu <= 5; ++nu) {
        double previous = 0.0;
        for (int n = 1; n <= 5; ++n) {
            const double zj = bessel_zero_jnu(nu, n);
            EXPECT_GT(zj, previous);
            EXPECT_NEAR(bessel_j(nu, zj), 0.0, 1e-10);
            previous = zj;
            const double zy = bessel_zero_ynu(nu, n);
            EXPECT_GT(zy, 0.0);
            EXPECT_NEAR(bessel_y(nu, zy), 0.0, 1e-9);
        }
    }
}

// ---------------------------------------------------------------------------------------
// [HIGH] Kelvin functions: bei and kei returned the constant 0 for every nu >= 1, and ber/ker
// returned the ORDINARY Bessel functions of a real argument.
// ---------------------------------------------------------------------------------------
TEST(SpecialAuditFixes, KelvinGeneralOrder) {
    // Reference: ber_nu + i bei_nu = J_nu(x e^{3 i pi/4}) evaluated from the complex ascending
    // series computed here in the test.
    for (int nu = 0; nu <= 3; ++nu) {
        for (double x : {0.5, 1.0, 3.0, 6.0}) {
            const std::complex<double> z = x * std::polar(1.0, 3.0 * kPi / 4.0);
            const std::complex<double> half = z / 2.0;
            std::complex<double> term = std::pow(half, static_cast<double>(nu)) /
                                        std::tgamma(static_cast<double>(nu) + 1.0);
            std::complex<double> sum = term;
            for (int k = 1; k < 200; ++k) {
                term *= -(half * half) / (static_cast<double>(k) * static_cast<double>(nu + k));
                sum += term;
                if (std::abs(term) <= 1e-18 * std::abs(sum)) {
                    break;
                }
            }
            EXPECT_NEAR(kelvin_ber(nu, x), sum.real(), 1e-11) << "nu=" << nu << " x=" << x;
            EXPECT_NEAR(kelvin_bei(nu, x), sum.imag(), 1e-11) << "nu=" << nu << " x=" << x;
        }
    }
    // bei_nu and kei_nu are NOT identically zero for nu >= 1 (the old code returned 0.0).
    EXPECT_NEAR(kelvin_bei(1, 1.0), 0.30755663137553668, 1e-12);
    EXPECT_NEAR(kelvin_ber(1, 1.0), -0.39586826101971140, 1e-12);
    EXPECT_NEAR(kelvin_kei(1, 1.0), -0.24199596642973845, 1e-11);
    EXPECT_NEAR(kelvin_ker(1, 1.0), -0.74032227684198271, 1e-11);
    EXPECT_NE(kelvin_bei(2, 0.5), 0.0);
    EXPECT_NE(kelvin_kei(2, 0.5), 0.0);
    // ker_nu(1) is not bessel_k(nu, 1) -- the old code returned exactly that.
    EXPECT_GT(std::abs(kelvin_ker(1, 1.0) - bessel_k(1, 1.0)), 0.1);
    // The Kelvin ODE x^2 y'' + x y' - (i x^2 + nu^2) y = 0 for y = ber + i bei and ker + i kei.
    for (int nu = 0; nu <= 2; ++nu) {
        for (double x : {0.7, 2.0, 5.0}) {
            const double h = 1e-3;
            const auto f = [nu](double t) {
                return std::complex<double>(kelvin_ber(nu, t), kelvin_bei(nu, t));
            };
            const auto g = [nu](double t) {
                return std::complex<double>(kelvin_ker(nu, t), kelvin_kei(nu, t));
            };
            const std::complex<double> rf = x * x * (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h) +
                                            x * (f(x + h) - f(x - h)) / (2 * h) -
                                            (std::complex<double>(0, 1) * x * x +
                                             static_cast<double>(nu * nu)) * f(x);
            const std::complex<double> rg = x * x * (g(x + h) - 2.0 * g(x) + g(x - h)) / (h * h) +
                                            x * (g(x + h) - g(x - h)) / (2 * h) -
                                            (std::complex<double>(0, 1) * x * x +
                                             static_cast<double>(nu * nu)) * g(x);
            EXPECT_LT(std::abs(rf), 1e-4 * std::max(1.0, std::abs(f(x))));
            EXPECT_LT(std::abs(rg), 1e-4 * std::max(1.0, std::abs(g(x))));
        }
    }
}

// ---------------------------------------------------------------------------------------
// [HIGH] jacobi_nd and jacobi_nc were byte-identical duplicates of jacobi_cd and jacobi_cs.
// ---------------------------------------------------------------------------------------
TEST(SpecialAuditFixes, JacobiGlaisherReciprocalPairs) {
    for (double u : {0.3, 0.5, 1.2, 2.0}) {
        for (double k : {0.1, 0.5, 0.9}) {
            EXPECT_NEAR(jacobi_nd(u, k) * jacobi_dn(u, k), 1.0, 1e-12);
            EXPECT_NEAR(jacobi_nc(u, k) * jacobi_cn(u, k), 1.0, 1e-12);
            EXPECT_NEAR(jacobi_ns(u, k) * jacobi_sn(u, k), 1.0, 1e-12);
            EXPECT_NEAR(jacobi_cd(u, k) * jacobi_dc(u, k), 1.0, 1e-12);
            EXPECT_NEAR(jacobi_sc(u, k) * jacobi_cs(u, k), 1.0, 1e-12);
            EXPECT_NEAR(jacobi_sd(u, k) * jacobi_ds(u, k), 1.0, 1e-12);
        }
    }
    EXPECT_NEAR(jacobi_nd(0.5, 0.5), 1.0294659945723399, 1e-9);
    EXPECT_NEAR(jacobi_nc(0.5, 0.5), 1.1364397998311904, 1e-9);
    // ... and they are no longer duplicates of cd / cs.
    EXPECT_GT(std::abs(jacobi_nd(0.5, 0.5) - jacobi_cd(0.5, 0.5)), 0.1);
    EXPECT_GT(std::abs(jacobi_nc(0.5, 0.5) - jacobi_cs(0.5, 0.5)), 0.5);
}

// ---------------------------------------------------------------------------------------
// [MEDIUM] laguerre_ln returned L_n(x)(-x)^k; [MEDIUM] chebyshev_tn/un discarded k entirely.
// ---------------------------------------------------------------------------------------
TEST(SpecialAuditFixes, AssociatedLaguerreAndChebyshevDerivatives) {
    // Reference: L_n^{(k)}(x) = sum_i (-1)^i C(n+k, n-i) x^i / i!, evaluated here.
    for (int n = 0; n <= 5; ++n) {
        for (int k = 0; k <= 4; ++k) {
            for (double x : {0.3, 0.7, 2.0}) {
                double expected = 0.0;
                for (int i = 0; i <= n; ++i) {
                    const double c = std::tgamma(n + k + 1.0) /
                                     (std::tgamma(n - i + 1.0) * std::tgamma(k + i + 1.0));
                    expected += ((i % 2) ? -1.0 : 1.0) * c * std::pow(x, i) / std::tgamma(i + 1.0);
                }
                EXPECT_NEAR(laguerre_ln(n, k, x), expected, 1e-11 * std::max(1.0, std::abs(expected)))
                    << "n=" << n << " k=" << k;
            }
        }
    }
    EXPECT_NEAR(laguerre_ln(2, 1, 0.5), 1.625, 1e-13);       // old: -0.0625
    EXPECT_NEAR(laguerre_ln(3, 2, 1.0), 7.0 / 3.0, 1e-13);
    // laguerre_ln must now agree with the module's generalised Laguerre at integer alpha.
    for (int n = 0; n <= 5; ++n) {
        for (int k = 0; k <= 3; ++k) {
            EXPECT_NEAR(laguerre_ln(n, k, 0.6), laguerre_la(n, static_cast<double>(k), 0.6), 1e-13);
        }
    }

    // k is the derivative order; k = 0 reproduces the two-argument functions and k >= 1 does not.
    for (int n = 0; n <= 6; ++n) {
        EXPECT_NEAR(chebyshev_tn(n, 0, 0.37), chebyshev_t(n, 0.37), 1e-13);
        EXPECT_NEAR(chebyshev_un(n, 0, 0.37), chebyshev_u(n, 0.37), 1e-13);
    }
    EXPECT_NEAR(chebyshev_tn(3, 1, 0.25), -2.25, 1e-12);   // old: chebyshev_t(3,0.25) = -1
    EXPECT_NEAR(chebyshev_un(3, 1, 0.25), -2.5, 1e-12);
    for (int n = 1; n <= 6; ++n) {
        const double x = 0.37;
        const double h = 1e-3;
        EXPECT_NEAR(chebyshev_tn(n, 1, x), d1([n](double t) { return chebyshev_t(n, t); }, x, h), 1e-4);
        EXPECT_NEAR(chebyshev_tn(n, 2, x), d2([n](double t) { return chebyshev_t(n, t); }, x, h), 1e-3);
        EXPECT_NEAR(chebyshev_un(n, 1, x), d1([n](double t) { return chebyshev_u(n, t); }, x, h), 1e-4);
        EXPECT_NEAR(chebyshev_un(n, 2, x), d2([n](double t) { return chebyshev_u(n, t); }, x, h), 1e-3);
    }
    EXPECT_DOUBLE_EQ(chebyshev_tn(3, 9, 0.25), 0.0);  // k > n
}

// ---------------------------------------------------------------------------------------
// [HIGH] meijer_g / fox_h: the shared helper did not compute the G^{1,1}_{1,1} it documented.
// ---------------------------------------------------------------------------------------
TEST(SpecialAuditFixes, MeijerGClosedForm) {
    // G^{1,1}_{1,1}(z | a; b) = Gamma(1-a+b) z^b (1+z)^{a-b-1}, which is the residue sum
    // sum_k (-1)^k/k! Gamma(1-a+b+k) z^{b+k} (checked directly below for |z| < 1).
    for (double a : {0.5, 1.0, 2.0}) {
        for (double b : {1.5, 2.0}) {
            for (double z : {0.2, 0.4, 0.6}) {
                // sum_k (-1)^k/k! Gamma(1-a+b+k) z^{b+k}, carried by the term ratio so no
                // intermediate Gamma overflows; converges for |z| < 1.
                double piece = std::tgamma(1 - a + b) * std::pow(z, b);
                double acc = piece;
                for (int k = 1; k < 4000; ++k) {
                    piece *= -(1 - a + b + k - 1) * z / k;
                    acc += piece;
                    if (std::abs(piece) < 1e-18 * std::abs(acc)) {
                        break;
                    }
                }
                EXPECT_NEAR(meijer_g(a, b, z), acc, 1e-10 * std::abs(acc))
                    << "a=" << a << " b=" << b << " z=" << z;
            }
        }
    }
    EXPECT_NEAR(meijer_g(1.0, 2.0, 0.5), 1.0 / 9.0, 1e-14);  // old: 0.39346934028736663
    // fox_h with unit exponents IS the Meijer G above; this documents the alias rather than
    // pretending the (a,b,z) signature can express a general H-function.
    EXPECT_DOUBLE_EQ(fox_h(1.0, 2.0, 0.5), meijer_g(1.0, 2.0, 0.5));
}

// ---------------------------------------------------------------------------------------
// [HIGH] Weierstrass: wrong Laurent coefficients from the z^6 term of P onward, and
// weierstrass_zeta had the wrong powers AND sign, violating zeta' = -P.
// ---------------------------------------------------------------------------------------
TEST(SpecialAuditFixes, WeierstrassDefiningIdentities) {
    for (double g2 : {0.0, 1.0, 2.5}) {
        for (double g3 : {0.0, 1.0, -0.7}) {
            for (double z : {0.1, 0.3, 0.5, 0.8}) {
                const double p = weierstrass_p(z, g2, g3);
                const double pp = weierstrass_pprime(z, g2, g3);
                // The defining differential equation P'^2 = 4P^3 - g2 P - g3.
                const double rhs = 4 * p * p * p - g2 * p - g3;
                EXPECT_NEAR(pp * pp, rhs, 1e-9 * std::abs(rhs)) << "g2=" << g2 << " z=" << z;
                // zeta' = -P.
                const double h = 1e-5;
                const double zp = d1([g2, g3](double t) { return weierstrass_zeta(t, g2, g3); }, z, h);
                EXPECT_NEAR(zp, -p, 1e-6 * std::abs(p));
                // sigma'/sigma = zeta.
                const double sp = d1([g2, g3](double t) { return weierstrass_sigma(t, g2, g3); }, z, h);
                EXPECT_NEAR(sp / weierstrass_sigma(z, g2, g3), weierstrass_zeta(z, g2, g3),
                            1e-8 * std::abs(weierstrass_zeta(z, g2, g3)));
            }
        }
    }
    // Reference: 1/z - sum_{n>=2} c_n z^{2n-1}/(2n-1) with c_n from the DLMF 23.9.3 recurrence,
    // summed by hand: 2 - 0.00208333... - 2.2321e-4 - 9.301e-7 - 1.057e-7 - ... = 1.99769241
    EXPECT_NEAR(weierstrass_zeta(0.5, 1.0, 1.0), 1.9976924119387851, 1e-13);  // old: 2.079315
    EXPECT_NEAR(weierstrass_zeta(0.5, 1.0, 0.0), 1.9979157363225009, 1e-13);  // old: 2.082292
    // Leading behaviour: the first correction to 1/z is -g2 z^3/60, not +g2 z/6.
    const double small = 0.02;
    EXPECT_NEAR(weierstrass_zeta(small, 3.0, 0.0) - 1.0 / small, -3.0 * std::pow(small, 3) / 60.0,
                1e-14);
}

// ---------------------------------------------------------------------------------------
// [MEDIUM] zeta returned NaN for every s <= 0 and only ~3 digits on (0,1); the s > 1
// Euler-Maclaurin tail also had the wrong sign on its N^{-s}/2 term (found while fixing this).
// ---------------------------------------------------------------------------------------
TEST(SpecialAuditFixes, ZetaAnalyticContinuationAndAccuracy) {
    EXPECT_NEAR(zeta(0.0), -0.5, 1e-13);
    EXPECT_NEAR(zeta(-1.0), -1.0 / 12.0, 1e-13);
    EXPECT_DOUBLE_EQ(zeta(-2.0), 0.0);
    EXPECT_DOUBLE_EQ(zeta(-4.0), 0.0);
    EXPECT_NEAR(zeta(-3.0), 1.0 / 120.0, 1e-14);
    EXPECT_NEAR(zeta(-5.0), -1.0 / 252.0, 1e-14);
    EXPECT_NEAR(zeta(-11.0), 691.0 / 32760.0, 1e-12);
    // Critical strip, now to full precision instead of ~2.6e-3.
    EXPECT_NEAR(zeta(0.5), -1.4603545088095868, 1e-12);
    EXPECT_NEAR(eta_dirichlet(0.5), 0.6048986434216305, 1e-13);
    EXPECT_NEAR(eta_dirichlet(1.0), std::log(2.0), 1e-13);
    EXPECT_NEAR(eta_dirichlet(-1.0), 0.25, 1e-12);
    // s slightly above 1: the old N^{-s}/2 sign error left an error of exactly N^{-s} = 1.1e-5.
    EXPECT_NEAR(zeta(1.5), 2.6123753486854883, 1e-12);
    EXPECT_NEAR(zeta(2.0), kPi * kPi / 6.0, 1e-13);
    EXPECT_NEAR(zeta(3.0), 1.2020569031595943, 1e-13);
    // eta(s) = (1 - 2^{1-s}) zeta(s) across the whole real line.
    for (double s : {-3.0, -0.5, 0.2, 0.7, 1.7, 4.0}) {
        EXPECT_NEAR(eta_dirichlet(s), (1.0 - std::pow(2.0, 1.0 - s)) * zeta(s),
                    1e-10 * std::max(1.0, std::abs(zeta(s))))
            << "s=" << s;
    }
}

// ---------------------------------------------------------------------------------------
// [MEDIUM] mathieu_a / mathieu_b built ONE matrix and patched the eigenvalue from outside.
// ---------------------------------------------------------------------------------------
TEST(SpecialAuditFixes, MathieuCharacteristicValues) {
    // Reference values: an independent RK4 shooting solve of y'' + (a - 2q cos 2x) y = 0 on
    // [0, pi/2] with the boundary condition that selects each family (agreement 1e-14), and the
    // A&S 20.2.25 small-q expansions.
    EXPECT_NEAR(mathieu_a(0, 1.0), -0.455138604107, 1e-9);   // old: -0.478734
    EXPECT_NEAR(mathieu_a(1, 1.0), 1.859108072514, 1e-9);    // old: 1.876272
    EXPECT_NEAR(mathieu_a(2, 1.0), 4.371300982735, 1e-9);
    EXPECT_NEAR(mathieu_a(3, 1.0), 9.078368847203, 1e-9);
    EXPECT_NEAR(mathieu_b(1, 1.0), -0.110248816992, 1e-9);   // old: -0.123728
    EXPECT_NEAR(mathieu_b(2, 1.0), 3.917024772998, 1e-9);    // old: 2.155963
    EXPECT_NEAR(mathieu_b(3, 1.0), 9.047739259809, 1e-9);
    EXPECT_NEAR(mathieu_a(0, 5.0), -5.800046020852, 1e-8);   // old: -7.659812
    EXPECT_NEAR(mathieu_b(1, 5.0), -5.790080598638, 1e-8);
    // q = 0 must give exactly n^2.
    for (int n = 0; n <= 6; ++n) {
        EXPECT_NEAR(mathieu_a(n, 0.0), static_cast<double>(n * n), 1e-10);
        if (n >= 1) {
            EXPECT_NEAR(mathieu_b(n, 0.0), static_cast<double>(n * n), 1e-10);
        }
    }
    // Small-q perturbation series (A&S 20.2.25) at q = 0.2.
    const double q = 0.2, q2 = q * q, q3 = q2 * q, q4 = q2 * q2;
    // (tolerances are the truncation error of the quoted partial series at q = 0.2)
    EXPECT_NEAR(mathieu_a(0, q), -q2 / 2 + 7 * q4 / 128, 2e-6);
    EXPECT_NEAR(mathieu_a(1, q), 1 + q - q2 / 8 - q3 / 64, 2e-5);
    EXPECT_NEAR(mathieu_b(1, q), 1 - q - q2 / 8 + q3 / 64, 2e-5);
    EXPECT_NEAR(mathieu_a(2, q), 4 + 5 * q2 / 12 - 763 * q4 / 13824, 1e-6);
    EXPECT_NEAR(mathieu_b(2, q), 4 - q2 / 12 + 5 * q4 / 13824, 1e-9);
    // ce and se satisfy the Mathieu equation with those eigenvalues.
    for (double qq : {0.1, 1.0, 5.0}) {
        for (int n = 0; n <= 4; ++n) {
            const double x = 0.7;
            const double h = 1e-4;
            const auto ce = [n, qq](double t) { return mathieu_ce(n, qq, t); };
            EXPECT_NEAR(d2(ce, x, h) + (mathieu_a(n, qq) - 2 * qq * std::cos(2 * x)) * ce(x), 0.0,
                        1e-5);
            if (n >= 1) {
                const auto se = [n, qq](double t) { return mathieu_se(n, qq, t); };
                EXPECT_NEAR(d2(se, x, h) + (mathieu_b(n, qq) - 2 * qq * std::cos(2 * x)) * se(x),
                            0.0, 1e-5);
            }
        }
    }
    // Standard normalisation: (1/pi) * integral over a period of ce_n^2 is 1.
    for (double qq : {0.0, 1.0, 5.0}) {
        for (int n = 0; n <= 3; ++n) {
            const int M = 256;
            double acc = 0.0;
            for (int i = 0; i < M; ++i) {
                const double x = 2 * kPi * i / M;
                acc += mathieu_ce(n, qq, x) * mathieu_ce(n, qq, x);
            }
            EXPECT_NEAR(acc * 2.0 / M, 1.0, 1e-6) << "q=" << qq << " n=" << n;
        }
    }
}

// ---------------------------------------------------------------------------------------
// [CRITICAL] mathieu_mc / mathieu_ms answered (n=1, q=1) from a two-entry lookup table, scaled
// everything else by an unexplained 0.472/0.803, and integrated the ODE with the wrong sign.
// ---------------------------------------------------------------------------------------
TEST(SpecialAuditFixes, ModifiedMathieuIsNotALookupTable) {
    // Mc_n(z,q) = ce_n(iz,q) and Ms_n(z,q) = -i se_n(iz,q), so at the origin Mc = ce and Ms = 0.
    for (double q : {0.3, 1.0, 3.0}) {
        for (int n = 0; n <= 3; ++n) {
            EXPECT_NEAR(mathieu_mc(n, q, 0.0), mathieu_ce(n, q, 0.0), 1e-13);
            if (n >= 1) {
                EXPECT_NEAR(mathieu_ms(n, q, 0.0), 0.0, 1e-15);
            }
        }
    }
    // The old body was DISCONTINUOUS at exactly q = 1, n = 1 (0.6877 vs -0.4457 a nanometre away).
    EXPECT_NEAR(mathieu_mc(1, 1.0, 0.0), mathieu_mc(1, 1.0 + 1e-9, 0.0), 1e-6);
    EXPECT_NEAR(mathieu_ms(1, 1.0, 0.0), mathieu_ms(1, 1.0 + 1e-9, 0.0), 1e-6);
    // Bounded and oscillatory in z -- the old sign-flipped ODE gave mathieu_mc(1,1,5) ~ 2e62.
    for (double z = 0.0; z <= 7.0; z += 0.5) {
        EXPECT_LT(std::abs(mathieu_mc(1, 1.0, z)), 10.0) << "z=" << z;
    }
    // The defining equation y'' - (a - 2q cosh 2z) y = 0.
    for (double q : {1.0, 3.0}) {
        for (int n = 0; n <= 2; ++n) {
            for (double z : {0.3, 1.0, 2.0}) {
                const double h = 1e-4;
                const auto mc = [n, q](double t) { return mathieu_mc(n, q, t); };
                const double coeff = mathieu_a(n, q) - 2 * q * std::cosh(2 * z);
                EXPECT_NEAR(d2(mc, z, h) - coeff * mc(z), 0.0,
                            1e-3 * std::abs(coeff * mc(z)) + 1e-6);
            }
        }
    }
    // Parity: Mc is even in z, Ms is odd.
    EXPECT_NEAR(mathieu_mc(1, 1.0, -2.0), mathieu_mc(1, 1.0, 2.0), 1e-13);
    EXPECT_NEAR(mathieu_ms(1, 1.0, -2.0), -mathieu_ms(1, 1.0, 2.0), 1e-13);
}

// ---------------------------------------------------------------------------------------
// [CRITICAL] spheroidal_lambda was a magic-number curve fit ignoring m, and spheroidal_s1
// returned the constant 0.043 scaled by (1-x^2)^{m/2}, ignoring n and c.
// ---------------------------------------------------------------------------------------
TEST(SpecialAuditFixes, SpheroidalEigenvalueAndAngularFunctions) {
    // Exact c -> 0 limit: lambda = n(n+1) -- the old fit gave 5.985387 for n = 2.
    for (int m = 0; m <= 3; ++m) {
        for (int n = m; n <= m + 3; ++n) {
            EXPECT_NEAR(spheroidal_lambda(n, m, 0.0), static_cast<double>(n * (n + 1)), 1e-11)
                << "m=" << m << " n=" << n;
        }
    }
    // lambda must depend on m -- the old fit returned the same value for every m.
    EXPECT_GT(std::abs(spheroidal_lambda(2, 0, 1.0) - spheroidal_lambda(2, 2, 1.0)), 0.3);
    // First-order perturbation theory: lambda = n(n+1) + c^2 <x^2>_{n,m} + O(c^4), with
    // <x^2> = [(n-m+1)(n+m+1)/(2n+3) + (n+m)(n-m)/(2n-1)]/(2n+1) derived from x P_l^m.
    const double c = 0.05;
    for (int m = 0; m <= 3; ++m) {
        for (int n = m; n <= m + 3; ++n) {
            const double nd = n, md = m;
            const double x2 = ((nd - md + 1) * (nd + md + 1) / (2 * nd + 3) +
                               (nd + md) * (nd - md) / (2 * nd - 1)) / (2 * nd + 1);
            EXPECT_NEAR(spheroidal_lambda(n, m, c), n * (n + 1.0) + c * c * x2, 1e-6)
                << "m=" << m << " n=" << n;
        }
    }
    // Flammer normalisation: S1 -> P_n^m and S2 -> Q_n^m as c -> 0.
    for (int m = 0; m <= 2; ++m) {
        for (int n = m; n <= m + 2; ++n) {
            for (double x : {-0.7, -0.2, 0.3, 0.8}) {
                EXPECT_NEAR(spheroidal_s1(n, m, 0.0, x), legendre_pn(n, m, x), 1e-11)
                    << "m=" << m << " n=" << n << " x=" << x;
            }
        }
    }
    for (int n = 0; n <= 3; ++n) {
        for (double x : {-0.6, 0.3, 0.75}) {
            EXPECT_NEAR(spheroidal_s2(n, 0, 0.0, x), legendre_q(n, x), 1e-9)
                << "n=" << n << " x=" << x;
        }
    }
    // S1 and S2 solve the angular equation with the computed lambda.
    for (double cc : {0.5, 1.0, 3.0}) {
        for (int m = 0; m <= 2; ++m) {
            for (int n = m; n <= m + 2; ++n) {
                for (double x : {-0.5, 0.25, 0.65}) {
                    const double h = 1e-4;
                    const double lambda = spheroidal_lambda(n, m, cc);
                    const auto s1 = [n, m, cc](double t) { return spheroidal_s1(n, m, cc, t); };
                    const auto g1 = [&](double t) { return (1 - t * t) * d1(s1, t, h); };
                    const double r1 = d1(g1, x, h) +
                                      (lambda - cc * cc * x * x - m * m / (1 - x * x)) * s1(x);
                    EXPECT_LT(std::abs(r1), 1e-3 * std::max(1.0, std::abs(s1(x))))
                        << "S1 c=" << cc << " m=" << m << " n=" << n << " x=" << x;
                    const auto s2 = [n, m, cc](double t) { return spheroidal_s2(n, m, cc, t); };
                    const auto g2 = [&](double t) { return (1 - t * t) * d1(s2, t, h); };
                    const double r2 = d1(g2, x, h) +
                                      (lambda - cc * cc * x * x - m * m / (1 - x * x)) * s2(x);
                    EXPECT_LT(std::abs(r2), 1e-3 * std::max(1.0, std::abs(s2(x))))
                        << "S2 c=" << cc << " m=" << m << " n=" << n << " x=" << x;
                }
            }
        }
    }
    // S1 genuinely depends on n and c (the old body returned 0.043 for all three of these).
    EXPECT_GT(std::abs(spheroidal_s1(2, 0, 1.0, 0.5) - spheroidal_s1(7, 0, 9.0, 0.5)), 0.2);
    EXPECT_NEAR(spheroidal_s1(0, 0, 0.0, 0.5), 1.0, 1e-13);  // P_0(0.5) = 1
    // Orthogonality of angular functions of the same m and different n.
    for (double cc : {1.0, 3.0}) {
        for (int m = 0; m <= 1; ++m) {
            for (int n1 = m; n1 <= m + 1; ++n1) {
                for (int n2 = n1 + 1; n2 <= m + 3; ++n2) {
                    const int M = 700;
                    double acc = 0.0;
                    for (int i = 0; i <= M; ++i) {
                        const double x = -1.0 + 2.0 * i / M;
                        const double xx = std::min(0.9999999, std::max(-0.9999999, x));
                        const double weight = (i == 0 || i == M) ? 0.5 : 1.0;
                        acc += weight * spheroidal_s1(n1, m, cc, xx) * spheroidal_s1(n2, m, cc, xx);
                    }
                    EXPECT_LT(std::abs(acc * 2.0 / M), 2e-3)
                        << "c=" << cc << " m=" << m << " n=" << n1 << "," << n2;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------
// [CRITICAL] parabolic_d summed ONE confluent series where D_nu needs two, used 2^{-nu/2} in
// one branch and 2^{+nu/2} in the other (so it jumped at the origin), and pcf_u multiplied by
// a spurious 2^{-a-1/2}e^{-x^2/4}.
// ---------------------------------------------------------------------------------------
TEST(SpecialAuditFixes, ParabolicCylinderFunctions) {
    // Exact closed forms: U(a,x) = D_{-a-1/2}(x), and D_0(x) = e^{-x^2/4}, D_1(x) = x e^{-x^2/4},
    // D_2 = (x^2-1)e^{-x^2/4}, D_3 = (x^3-3x)e^{-x^2/4},
    // D_{-1}(x) = e^{x^2/4} sqrt(pi/2) erfc(x/sqrt2).
    for (double x : {-8.0, -3.0, -1.0, 0.0, 1.0, 3.0, 6.5, 9.0}) {
        const double g = std::exp(-0.25 * x * x);
        EXPECT_NEAR(pcf_u(-0.5, x), g, 1e-14 * std::max(1.0, g)) << "D_0 x=" << x;
        EXPECT_NEAR(pcf_u(-1.5, x), x * g, 1e-13 * std::max(1.0, std::abs(x * g)));
        EXPECT_NEAR(pcf_u(-2.5, x), (x * x - 1) * g, 1e-12 * std::max(1.0, std::abs((x * x - 1) * g)));
        EXPECT_NEAR(pcf_u(-3.5, x), (x * x * x - 3 * x) * g,
                    1e-12 * std::max(1.0, std::abs((x * x * x - 3 * x) * g)));
        const double dm1 = std::exp(0.25 * x * x) * std::sqrt(kPi / 2.0) * std::erfc(x / std::sqrt(2.0));
        EXPECT_NEAR(pcf_u(0.5, x), dm1, 1e-11 * std::max(1e-12, std::abs(dm1))) << "D_{-1} x=" << x;
    }
    // U(a,0) = 2^{-a/2-1/4} sqrt(pi)/Gamma(3/4+a/2) -- the old pcf_u(0,0) was this times 2^{-1/2}.
    for (double a : {-1.0, -0.5, 0.0, 0.5, 1.0, 2.0}) {
        const double expected = std::pow(2.0, -0.5 * a - 0.25) * std::sqrt(kPi) /
                                std::tgamma(0.75 + 0.5 * a);
        EXPECT_NEAR(pcf_u(a, 0.0), expected, 1e-13) << "a=" << a;
    }
    // No jump at the origin (the two old branches used different powers of 2).
    EXPECT_NEAR(pcf_u(-1.5, 1e-9), pcf_u(-1.5, 0.0), 1e-8);
    EXPECT_NEAR(pcf_u(0.3, 1e-9), pcf_u(0.3, 0.0), 1e-8);
    // Weber's equation y'' - (x^2/4 + a) y = 0 for U and V.
    for (double a : {-1.3, 0.0, 0.5, 2.0}) {
        for (double x : {-4.0, -1.0, 0.5, 2.0, 5.0}) {
            const double h = 1e-4;
            const auto u = [a](double t) { return pcf_u(a, t); };
            const auto v = [a](double t) { return pcf_v(a, t); };
            EXPECT_LT(std::abs(d2(u, x, h) - (0.25 * x * x + a) * u(x)), 1e-4 * std::abs(u(x)) + 1e-12);
            EXPECT_LT(std::abs(d2(v, x, h) - (0.25 * x * x + a) * v(x)), 1e-4 * std::abs(v(x)) + 1e-12);
        }
    }
    // Wronskian U V' - U' V = sqrt(2/pi) (DLMF 12.2.11).
    for (double a : {-1.3, 0.0, 0.7, 2.0}) {
        for (double x : {-2.0, 0.3, 1.5, 4.0}) {
            const double h = 1e-5;
            const double w = pcf_u(a, x) * d1([a](double t) { return pcf_v(a, t); }, x, h) -
                             d1([a](double t) { return pcf_u(a, t); }, x, h) * pcf_v(a, x);
            EXPECT_NEAR(w, std::sqrt(2.0 / kPi), 1e-7) << "a=" << a << " x=" << x;
        }
    }
    // W(a,x) is the oscillatory solution of y'' + (x^2/4 - a)y = 0, NOT a U/V mix.
    EXPECT_NEAR(pcf_w(0.0, 0.0), std::pow(2.0, -0.75) * std::sqrt(std::tgamma(0.25) / std::tgamma(0.75)),
                1e-13);
    for (double a : {-2.0, 0.0, 1.0, 3.0}) {
        for (double x : {0.0, 0.7, 2.0, 4.0}) {
            const double h = 1e-4;
            const auto w = [a](double t) { return pcf_w(a, t); };
            EXPECT_LT(std::abs(d2(w, x, h) + (0.25 * x * x - a) * w(x)),
                      1e-4 * std::max(1.0, std::abs(w(x))));
        }
    }
}

// ---------------------------------------------------------------------------------------
// [MEDIUM] tricomi_u evaluated the divergent 2F0 at (-z) instead of (-1/z), returning ~1e272.
// ---------------------------------------------------------------------------------------
TEST(SpecialAuditFixes, TricomiUIsFiniteAndCorrect) {
    // Exact identity U(a, a+1, z) = z^{-a} (DLMF 13.6.4).
    for (double a : {0.3, 1.0, 2.0, 3.5}) {
        for (double z : {0.2, 1.0, 5.0, 40.0}) {
            EXPECT_NEAR(tricomi_u(a, a + 1.0, z), std::pow(z, -a), 1e-12 * std::pow(z, -a));
        }
    }
    // Kummer connection formula (non-integer b), summed here.
    const auto kummer_m_ref = [](double a, double b, double z) {
        double s = 1.0, t = 1.0;
        for (int n = 1; n < 400; ++n) {
            t *= (a + n - 1) * z / ((b + n - 1) * n);
            s += t;
            if (std::abs(t) <= 1e-18 * std::abs(s)) {
                break;
            }
        }
        return s;
    };
    struct C { double a, b, z; };
    const C cases[] = {{1.0, 0.5, 1.0}, {0.3, 1.7, 0.5}, {0.2, 0.5, 2.5}, {2.0, 1.4, 0.7},
                       {0.7, 2.5, 4.0}, {0.2, 2.5, 0.4}, {-0.5, 0.5, 1.2}, {-1.5, 0.7, 2.0}};
    for (const C& c : cases) {
        const double expected = std::tgamma(1 - c.b) / std::tgamma(c.a - c.b + 1) *
                                    kummer_m_ref(c.a, c.b, c.z) +
                                std::tgamma(c.b - 1) / std::tgamma(c.a) * std::pow(c.z, 1 - c.b) *
                                    kummer_m_ref(c.a - c.b + 1, 2 - c.b, c.z);
        EXPECT_NEAR(tricomi_u(c.a, c.b, c.z), expected, 1e-9 * std::abs(expected))
            << "a=" << c.a << " b=" << c.b << " z=" << c.z;
    }
    EXPECT_NEAR(tricomi_u(1.0, 0.5, 1.0), 0.4842556877173759, 1e-11);   // old: -4.18e272
    EXPECT_NEAR(tricomi_u(0.5, -0.5, 50.0), 0.1387123943072375, 1e-11); // old: -6.8e23
    EXPECT_NEAR(kummer_u(0.2, 2.5, 0.4), 2.1895013784559132, 1e-9);     // old: -1.84e208
    EXPECT_NEAR(whittaker_w(0.25, 0.35, 4.0), 0.19388253869569511, 1e-9);
    // Kummer's equation z U'' + (b - z) U' - a U = 0, including integer b.
    for (double a : {0.4, 1.0, 2.3}) {
        for (double b : {1.0, 2.0, 3.0, 1.7}) {
            for (double z : {0.5, 2.0}) {
                const double h = 1e-4;
                const auto u = [a, b](double t) { return tricomi_u(a, b, t); };
                EXPECT_TRUE(std::isfinite(u(z)));
                EXPECT_LT(std::abs(z * d2(u, z, h) + (b - z) * d1(u, z, h) - a * u(z)),
                          1e-4 * std::max(1.0, std::abs(u(z))))
                    << "a=" << a << " b=" << b << " z=" << z;
            }
        }
    }
}

// ---------------------------------------------------------------------------------------
// [MEDIUM] Heun: heun_g dropped the Fuchs-determined epsilon/(z-a) pole and the three confluent
// members kept a 1/(z-1) pole they must not have.
// ---------------------------------------------------------------------------------------
TEST(SpecialAuditFixes, HeunEquations) {
    // With q = 0 and alpha = 0 the zeroth-order term vanishes, so y == 1 solves each equation.
    for (double z : {0.2, 0.5, 0.9, 1.5}) {
        if (z < 1.0) {
            EXPECT_NEAR(heun_g(2.0, 0.0, 0.0, 0.7, 1.1, 0.9, z), 1.0, 1e-12);
            EXPECT_NEAR(heun_c(0.0, 0.0, 0.3, 1.1, 0.9, z), 1.0, 1e-12);
        }
        EXPECT_NEAR(heun_t(0.0, 0.0, 0.0, 0.5, z), 1.0, 1e-12);
    }
    // Exact degree-one Heun polynomial: with alpha = -1 the z^2 coefficient of the equation
    // applied to y = 1 + c z vanishes identically, and matching the remaining two coefficients
    // gives a quadratic for c with q = a c gamma.
    {
        const double a = 2.0, gamma = 1.0, delta = 1.0, beta = 0.5;
        const double A = -a * gamma;
        const double B = -gamma * a - delta * a + delta - beta;
        const double C = -beta;
        const double c1 = (-B - std::sqrt(B * B - 4 * A * C)) / (2 * A);
        const double c2 = (-B + std::sqrt(B * B - 4 * A * C)) / (2 * A);
        const double c = (std::abs(c2) < std::abs(c1)) ? c2 : c1;
        const double q = a * c * gamma;
        for (double z : {0.1, 0.3, 0.6, 0.9}) {
            EXPECT_NEAR(heun_g(a, q, -1.0, beta, gamma, delta, z), 1.0 + c * z, 1e-10)
                << "z=" << z;
        }
    }
    // ODE residuals against DLMF 31.1.1 and 31.12.1-31.12.4.
    const double a = 0.5, q = 0.1, al = 0.2, be = 0.3, ga = 0.4, de = 0.5;
    const double eps = al + be - ga - de + 1.0;
    const double h = 1e-4;
    for (double z : {0.2, 0.4}) {
        const auto y = [&](double t) { return heun_g(a, q, al, be, ga, de, t); };
        const double r = d2(y, z, h) + (ga / z + de / (z - 1) + eps / (z - a)) * d1(y, z, h) +
                         (al * be * z - q) / (z * (z - 1) * (z - a)) * y(z);
        EXPECT_LT(std::abs(r), 1e-4) << "heun_g z=" << z;
    }
    for (double z : {0.2, 0.5, 0.8}) {
        const auto y = [&](double t) { return heun_c(q, al, be, ga, de, t); };
        const double r = d2(y, z, h) + (ga / z + de / (z - 1) + be) * d1(y, z, h) +
                         (al * z - q) / (z * (z - 1)) * y(z);
        EXPECT_LT(std::abs(r), 1e-4) << "heun_c z=" << z;
    }
    for (double z : {0.3, 0.8, 1.6}) {
        const auto y = [&](double t) { return heun_d(q, al, ga, de, t); };
        const double r = z * z * d2(y, z, h) + (-z * z + de * z + ga) * d1(y, z, h) +
                         (al * z - q) * y(z);
        EXPECT_LT(std::abs(r), 1e-4) << "heun_d z=" << z;
    }
    for (double z : {0.2, 0.5, 1.0}) {
        const auto y = [&](double t) { return heun_b(q, al, be, de, t); };
        const double r = z * d2(y, z, h) + (1 + al - be * z - 2 * z * z) * d1(y, z, h) +
                         ((q - al - 2) * z - 0.5 * (de + (1 + al) * be)) * y(z);
        EXPECT_LT(std::abs(r), 1e-4) << "heun_b z=" << z;
    }
    for (double z : {0.2, 0.6, 1.2}) {
        const auto y = [&](double t) { return heun_t(q, al, be, ga, t); };
        const double r = d2(y, z, h) - (ga + 3 * z * z) * d1(y, z, h) + (al * z - q) * y(z);
        EXPECT_LT(std::abs(r), 1e-4) << "heun_t z=" << z;
    }
    // heun_d must actually use its delta parameter (it was declared `double /*delta*/`).
    EXPECT_GT(std::abs(heun_d(0.1, 0.2, 0.3, 1.0, 0.5) - heun_d(0.1, 0.2, 0.3, 99.0, 0.5)), 1e-3);
    // The general and confluent Heun solutions are undefined past their singular points.
    EXPECT_TRUE(std::isnan(heun_g(0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.9)));
    EXPECT_TRUE(std::isnan(heun_c(0.1, 0.2, 0.3, 0.4, 0.5, 1.5)));
}

// ---------------------------------------------------------------------------------------
// [HIGH] Painleve III-VI integrated right-hand sides that were not the Painleve equations.
// ---------------------------------------------------------------------------------------
TEST(SpecialAuditFixes, PainleveTranscendents) {
    // PIV with alpha = 0, beta = -2 has the exact solution w = -2z (verified by substitution).
    for (double x : {0.3, 0.5, 1.0, 2.0}) {
        EXPECT_NEAR(painleve4(x, -0.2, -2.0, 0.0, -2.0), -2.0 * x, 1e-9) << "x=" << x;
    }
    // PIII(alpha, -alpha, gamma=1, delta=-1) has the exact constant solution w = 1.
    for (double x : {0.5, 1.5, 3.0}) {
        EXPECT_NEAR(painleve3(x, 1.0, 0.0, 0.7, -0.7), 1.0, 1e-10) << "x=" << x;
    }
    // PV with gamma = delta = 0 and beta = -alpha c^2 has the exact constant solution w = c.
    for (double x : {0.5, 1.0, 2.5}) {
        EXPECT_NEAR(painleve5(x, 2.0, 0.0, 1.0, -4.0, 0.0, 0.0), 2.0, 1e-10) << "x=" << x;
    }
    // PVI with all parameters zero has constant solutions.
    for (double x : {2.5, 3.0}) {
        EXPECT_NEAR(painleve6(x, 1.7, 0.0, 0.0, 0.0, 0.0, 0.0), 1.7, 1e-10) << "x=" << x;
    }
    // ODE residuals against DLMF 32.2.3-32.2.6.
    const double h = 1e-4;
    for (double x : {0.3, 0.5, 0.8}) {
        const auto y = [](double t) { return painleve3(t, 0.5, -0.1, 0.5, 0.3); };
        const double w = y(x), yp = d1(y, x, h);
        const double rhs = yp * yp / w - yp / x + (0.5 * w * w + 0.3) / x + w * w * w - 1.0 / w;
        EXPECT_LT(std::abs(d2(y, x, h) - rhs), 1e-3) << "PIII x=" << x;
    }
    for (double x : {0.3, 0.5, 0.8}) {
        const auto y = [](double t) { return painleve4(t, 0.8, -0.05, 0.2, 0.4); };
        const double w = y(x), yp = d1(y, x, h);
        const double rhs = 0.5 * yp * yp / w + 1.5 * w * w * w + 4 * x * w * w +
                           2 * (x * x - 0.2) * w + 0.4 / w;
        EXPECT_LT(std::abs(d2(y, x, h) - rhs), 1e-3) << "PIV x=" << x;
    }
    for (double x : {0.3, 0.5, 0.8}) {
        const auto y = [](double t) { return painleve5(t, 0.5, -0.05, 0.01, 0.02, 0.03, 0.04); };
        const double w = y(x), yp = d1(y, x, h), wm = w - 1.0;
        const double rhs = (0.5 / w + 1 / wm) * yp * yp - yp / x +
                           wm * wm / (x * x) * (0.01 * w + 0.02 / w) + 0.03 * w / x +
                           0.04 * w * (w + 1) / wm;
        EXPECT_LT(std::abs(d2(y, x, h) - rhs), 1e-3) << "PV x=" << x;
    }
    for (double x : {2.3, 2.5, 3.0}) {
        const auto y = [](double t) { return painleve6(t, 0.5, -0.05, 0.1, 0.2, 0.3, 0.4); };
        const double w = y(x), yp = d1(y, x, h);
        const double wm = w - 1.0, wz = w - x, zm = x - 1.0;
        const double rhs = 0.5 * (1 / w + 1 / wm + 1 / wz) * yp * yp -
                           (1 / x + 1 / zm + 1 / wz) * yp +
                           w * wm * wz / (x * x * zm * zm) *
                               (0.1 + 0.2 * x / (w * w) + 0.3 * (x - 1) / (wm * wm) +
                                0.4 * x * (x - 1) / (wz * wz));
        EXPECT_LT(std::abs(d2(y, x, h) - rhs), 1e-3) << "PVI x=" << x;
    }
}
