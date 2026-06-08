#include <cmath>
#include <gtest/gtest.h>

#include "ms/special/special.hpp"

using namespace ms;

namespace {

void expect_finite(double value) {
    EXPECT_TRUE(std::isfinite(value));
}

} // namespace

TEST(SpecialSweepTest, bessel_orders_and_arguments) {
    for (int nu = 0; nu <= 5; ++nu) {
        for (double x : {0.01, 0.25, 0.75, 1.5, 3.0}) {
            expect_finite(bessel_j(nu, x));
            expect_finite(bessel_i(nu, x));
            if (x > 0.05) {
                expect_finite(bessel_k(nu, x));
                expect_finite(bessel_y(nu, x));
            }
        }
    }
}

TEST(SpecialSweepTest, orthogonal_polynomial_degrees) {
    for (int n = 0; n <= 10; ++n) {
        for (double x : {-0.95, -0.25, 0.0, 0.25, 0.95}) {
            expect_finite(legendre_p(n, x));
            expect_finite(chebyshev_t(n, x));
            expect_finite(chebyshev_u(n, x));
            expect_finite(hermite_h(n, x));
            expect_finite(laguerre_l(n, x));
            if (n >= 1) {
                expect_finite(legendre_q(n, x));
            }
        }
    }
}

TEST(SpecialSweepTest, zeta_and_polylog_values) {
    for (double s : {2.0, 3.0, 4.0, 0.5, 1.5}) {
        expect_finite(zeta(s));
        expect_finite(eta_dirichlet(s));
    }
    for (int n = 1; n <= 5; ++n) {
        for (double z : {-0.3, 0.2, 0.5, 0.8}) {
            expect_finite(polylog(n, z));
        }
    }
    expect_finite(zeta_hurwitz(2.0, 0.25));
    expect_finite(lerch_phi(0.3, 2.0, 0.5));
    expect_finite(beta_dirichlet(2.0));
    expect_finite(clausen(0.7));
}

TEST(SpecialSweepTest, elliptic_and_jacobi_variants) {
    for (double k : {0.1, 0.4, 0.7}) {
        expect_finite(ellip_k(k));
        expect_finite(ellip_e(k));
        expect_finite(ellip_d(k));
        for (double u : {0.0, 0.2, 0.6}) {
            expect_finite(jacobi_sn(u, k));
            expect_finite(jacobi_cn(u, k));
            expect_finite(jacobi_dn(u, k));
        }
    }
}

TEST(SpecialSweepTest, hypergeo_and_whittaker_grid) {
    for (double z : {0.05, 0.2, 0.4}) {
        expect_finite(hypergeo_0f1(1.5, z));
        expect_finite(hypergeo_1f1(0.5, z));
        expect_finite(hypergeo_2f1(0.2, 0.3, 1.5, z));
        expect_finite(kummer_m(0.2, 1.5, z));
        expect_finite(tricomi_u(0.5, 1.5, z));
        expect_finite(whittaker_m(0.0, 0.5, z));
        expect_finite(whittaker_w(0.0, 0.5, 1.0 + z));
    }
}

TEST(SpecialSweepTest, airy_and_kelvin_grid) {
    for (double x : {-1.0, -0.1, 0.0, 0.1, 1.0, 2.0}) {
        expect_finite(airy_ai(x));
        expect_finite(airy_bi(x));
        expect_finite(dawson(x));
        expect_finite(dawsonx(x));
    }
    for (int nu = 0; nu <= 3; ++nu) {
        expect_finite(kelvin_ber(nu, 0.3));
        expect_finite(kelvin_ker(nu, 0.5));
    }
}

TEST(SpecialSweepTest, mathieu_and_struve_grid) {
    for (int n = 1; n <= 4; ++n) {
        for (double q : {0.1, 0.3, 0.5}) {
            expect_finite(mathieu_a(n, q));
            expect_finite(mathieu_b(n, q));
            expect_finite(mathieu_ce(n, q, 0.2));
            expect_finite(mathieu_se(n, q, 0.2));
        }
    }
    for (int nu = 0; nu <= 3; ++nu) {
        for (double x : {0.2, 0.8, 1.5}) {
            expect_finite(struve_h(nu, x));
            expect_finite(struve_l(nu, x));
            expect_finite(anger_j(nu, x));
            expect_finite(weber_e(nu, x));
        }
    }
}

TEST(SpecialSweepTest, error_functions_grid) {
    for (double x : {-4.0, -1.0, 0.0, 1.0, 4.0, 8.0}) {
        expect_finite(ms::erf(x));
        expect_finite(ms::erfc(x));
        expect_finite(erfi(x));
        expect_finite(erfcx(x));
    }
}

TEST(SpecialSweepTest, bessel_zeros_kelvin_nu0_and_heun_grid) {
    for (int nu = 0; nu <= 2; ++nu) {
        for (int n = 1; n <= 3; ++n) {
            expect_finite(bessel_zero_jnu(nu, n));
            expect_finite(bessel_zero_ynu(nu, n));
        }
    }
    expect_finite(kelvin_bei(0, 0.4));
    expect_finite(kelvin_kei(0, 0.6));
    expect_finite(mathieu_a(0, 0.15));
    expect_finite(heun_g(0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.5));
    expect_finite(heun_g(0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.999999));
    expect_finite(heun_b(0.1, 0.2, 0.3, 0.4, 0.5));
    expect_finite(painleve3(0.5, 0.2, 0.0, 0.1, 0.2));
    expect_finite(painleve4(0.5, 0.2, 0.0, 0.1, 0.2));
}
