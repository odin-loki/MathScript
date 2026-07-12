#include <gtest/gtest.h>
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include "ms/special/special.hpp"

using namespace ms;

namespace {

double sph_harm_inner_product(int l1, int m1, int l2, int m2) {
    constexpr int n_theta = 40;
    constexpr int n_phi = 40;
    const double d_theta = M_PI / static_cast<double>(n_theta);
    const double d_phi = 2.0 * M_PI / static_cast<double>(n_phi);
    double sum = 0.0;
    for (int i = 0; i <= n_theta; ++i) {
        const double theta = d_theta * static_cast<double>(i);
        const double sin_theta = std::sin(theta);
        const double w_theta = (i == 0 || i == n_theta) ? 1.0 : ((i % 2 == 0) ? 2.0 : 4.0);
        for (int j = 0; j <= n_phi; ++j) {
            const double phi = d_phi * static_cast<double>(j);
            const double w_phi = (j == 0 || j == n_phi) ? 1.0 : ((j % 2 == 0) ? 2.0 : 4.0);
            const std::complex<double> y1 = sph_harm_y(l1, m1, theta, phi);
            const std::complex<double> y2 = sph_harm_y(l2, m2, theta, phi);
            const double integrand = (std::conj(y1) * y2).real() * sin_theta;
            sum += w_theta * w_phi * integrand;
        }
    }
    return sum * d_theta * d_phi / 9.0;
}

} // namespace

TEST(SpecialPolyExtTest, extended_orthogonal_polynomials) {
    EXPECT_NEAR(jacobi_p(2, 0.5, 0.5, 0.3), -0.4, 1e-6);
    EXPECT_NEAR(gegenbauer_c(2, 1.0, 0.5), 0.0, 1e-12);
    EXPECT_NEAR(laguerre_la(2, 1.0, 0.5), 1.625, 1e-6);
    EXPECT_NEAR(hermite_he(2, 0.5), -0.75, 1e-12);
    EXPECT_NEAR(chebyshev_v(2, 0.5), 1.0, 1e-6);
    EXPECT_NEAR(chebyshev_w(2, 0.5), 0.5773502691896257, 1e-6);
}

TEST(SpecialPolyExtTest, spherical_harmonic) {
    EXPECT_NEAR(sph_harm(1, 1, 0.5, 1.0), -0.08949498165189648, 1e-3);
}

TEST(SphHarmTest, assoc_legendre_matches_legendre_pn) {
    for (int l = 0; l <= 4; ++l) {
        for (int m = 0; m <= l; ++m) {
            for (double x : {0.0, 0.3, -0.5, 0.8}) {
                EXPECT_NEAR(assoc_legendre_p(l, m, x), legendre_pn(l, m, x), 1e-12)
                    << "l=" << l << " m=" << m << " x=" << x;
            }
        }
    }
}

TEST(SphHarmTest, assoc_legendre_negative_m_symmetry) {
    // P_l^{-m}(x) = (-1)^m (l-m)!/(l+m)! P_l^m(x) (DLMF §14.9.5).
    const double x = 0.4;
    for (int l = 1; l <= 4; ++l) {
        for (int m = 1; m <= l; ++m) {
            const double positive = legendre_pn(l, m, x);
            const double ratio = std::tgamma(static_cast<double>(l - m) + 1.0) /
                                 std::tgamma(static_cast<double>(l + m) + 1.0);
            const double sign = (m % 2 == 0) ? 1.0 : -1.0;
            EXPECT_NEAR(assoc_legendre_p(l, -m, x), sign * ratio * positive, 1e-12)
                << "l=" << l << " m=" << m;
        }
    }
}

TEST(SphHarmTest, y00_constant_closed_form) {
    // Y_0^0 = 1/sqrt(4π), independent of (θ, φ).
    const double expected = 1.0 / std::sqrt(4.0 * M_PI);
    for (double theta : {0.1, 0.7, 1.2, M_PI / 2.0}) {
        for (double phi : {0.0, 0.5, 1.0, 2.0}) {
            const std::complex<double> y = sph_harm_y(0, 0, theta, phi);
            EXPECT_NEAR(y.real(), expected, 1e-12);
            EXPECT_NEAR(y.imag(), 0.0, 1e-12);
            EXPECT_NEAR(sph_harm(0, 0, theta, phi), expected, 1e-12);
        }
    }
}

TEST(SphHarmTest, y10_closed_form) {
    // Y_1^0(θ, φ) = sqrt(3/(4π)) cos θ (real, φ-independent).
    for (double theta : {0.2, 0.9, 1.4}) {
        for (double phi : {0.0, 0.8, 2.5}) {
            const std::complex<double> y = sph_harm_y(1, 0, theta, phi);
            const double expected = std::sqrt(3.0 / (4.0 * M_PI)) * std::cos(theta);
            EXPECT_NEAR(y.real(), expected, 1e-12);
            EXPECT_NEAR(y.imag(), 0.0, 1e-12);
        }
    }
}

TEST(SphHarmTest, negative_m_conjugate_relation) {
    // Y_l^{-m} = (-1)^m conj(Y_l^m) for m > 0 (DLMF §14.30).
    const double theta = 0.6;
    const double phi = 1.1;
    for (int l = 1; l <= 3; ++l) {
        for (int m = 1; m <= l; ++m) {
            const std::complex<double> positive = sph_harm_y(l, m, theta, phi);
            const std::complex<double> negative = sph_harm_y(l, -m, theta, phi);
            const double sign = (m % 2 == 0) ? 1.0 : -1.0;
            EXPECT_NEAR(negative.real(), sign * positive.real(), 1e-12);
            EXPECT_NEAR(negative.imag(), -sign * positive.imag(), 1e-12);
        }
    }
}

TEST(SphHarmTest, real_wrapper_matches_complex_real_part) {
    for (int l = 0; l <= 3; ++l) {
        for (int m = -l; m <= l; ++m) {
            const std::complex<double> y = sph_harm_y(l, m, 0.7, 0.4);
            EXPECT_NEAR(sph_harm(l, m, 0.7, 0.4), y.real(), 1e-15) << "l=" << l << " m=" << m;
        }
    }
}

TEST(SphHarmTest, orthonormality_quadrature) {
    struct Pair {
        int l;
        int m;
    };
    const Pair states[] = {{0, 0}, {1, 0}, {1, 1}, {2, 0}, {2, 1}, {2, 2}};
    for (const Pair& a : states) {
        for (const Pair& b : states) {
            const double inner = sph_harm_inner_product(a.l, a.m, b.l, b.m);
            const double expected = (a.l == b.l && a.m == b.m) ? 1.0 : 0.0;
            EXPECT_NEAR(inner, expected, 2e-2) << "⟨" << a.l << "," << a.m << "|" << b.l << "," << b.m << "⟩";
        }
    }
}

TEST(SphHarmTest, invalid_orders_return_nan) {
    const std::complex<double> bad = sph_harm_y(1, 2, 0.5, 0.3);
    EXPECT_TRUE(std::isnan(bad.real()));
    EXPECT_TRUE(std::isnan(bad.imag()));
    EXPECT_TRUE(std::isnan(sph_harm(1, 2, 0.5, 0.3)));
    EXPECT_TRUE(std::isnan(assoc_legendre_p(1, 2, 0.5)));
}

TEST(SpecialEllipticTest, complete_integrals) {
    const double k = 0.5;
    EXPECT_NEAR(ellip_k(k), 1.685750354812596, 1e-6);
    EXPECT_NEAR(ellip_e(k), 1.4674622093394272, 1e-6);
    EXPECT_NEAR(ellip_pi(0.5, k), 2.4136715042011945, 1e-3);
    EXPECT_NEAR(ellip_d(k), (ellip_k(k) - ellip_e(k)) / (k * k), 1e-6);
}

TEST(SpecialEllipticTest, incomplete_integrals) {
    const double k = 0.5;
    EXPECT_NEAR(ellip_f(0.3, k), 0.30111597966406606, 1e-3);
    EXPECT_NEAR(ellip_e_inc(0.3, k), 0.2988914110164986, 1e-3);
}

TEST(SpecialEllipticTest, jacobi_elliptic_functions) {
    const double u = 0.5;
    const double k = 0.5;
    EXPECT_NEAR(jacobi_sn(u, k), 0.47508293602853646, 1e-3);
    EXPECT_NEAR(jacobi_cn(u, k), 0.8799410229637583, 1e-3);
    EXPECT_NEAR(jacobi_dn(u, k), 0.9713773988381788, 1e-3);
    EXPECT_NEAR(jacobi_sc(u, k), jacobi_sn(u, k) / jacobi_cn(u, k), 1e-6);
    EXPECT_NEAR(jacobi_sd(u, k), jacobi_sn(u, k) / jacobi_dn(u, k), 1e-6);
    EXPECT_NEAR(jacobi_cd(u, k), jacobi_cn(u, k) / jacobi_dn(u, k), 1e-6);
    EXPECT_NEAR(jacobi_ns(u, k), 1.0 / jacobi_sn(u, k), 1e-6);
}
