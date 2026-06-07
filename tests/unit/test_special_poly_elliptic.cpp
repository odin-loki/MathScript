#include <gtest/gtest.h>
#include <cmath>
#include "ms/special/special.hpp"

using namespace ms;

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
