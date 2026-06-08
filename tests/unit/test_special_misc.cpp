#include <gtest/gtest.h>
#include <cmath>

#include "ms/special/special.hpp"

using namespace ms;

namespace {

void expect_finite(double value) {
    EXPECT_TRUE(std::isfinite(value));
}

} // namespace

TEST(SpecialMiscTest, extended_error_functions) {
    expect_finite(erfi(0.5));
    expect_finite(erfcx(1.0));
    expect_finite(dawsonx(0.5));
    expect_finite(digamma(2.0));
}

TEST(SpecialMiscTest, bessel_variants) {
    expect_finite(bessel_y(1, 1.0));
    expect_finite(bessel_i(1, 1.0));
    EXPECT_NEAR(bessel_h(0, 1.0), bessel_j0(1.0) + bessel_y0(1.0), 1e-6);
    EXPECT_NEAR(bessel_hy(0, 1.0), bessel_j0(1.0) - bessel_y0(1.0), 1e-6);
    EXPECT_NEAR(bessel_l(0, 1.0), bessel_j0(1.0), 1e-6);
    EXPECT_NEAR(bessel_lu(0, 1.0), bessel_y0(1.0), 1e-6);
}

TEST(SpecialMiscTest, struve_and_legendre) {
    expect_finite(struve_l(0, 1.0));
    expect_finite(legendre_q(1, 0.5));
    expect_finite(legendre_pn(2, 1, 0.5));
}

TEST(SpecialMiscTest, airy_and_hyper_indexed) {
    expect_finite(airy_bi(0.0));
    expect_finite(airy_bip(1.0));
    expect_finite(hypergeo_0f1n(1, 2.0, 0.5));
    expect_finite(hypergeo_1f1n(1, 1.0, 0.5));
}

TEST(SpecialMiscTest, mathieu_and_spheroidal) {
    expect_finite(mathieu_se(1, 0.1, 0.5));
    expect_finite(spheroidal_s2(1, 1, 5.0, 0.5));
}

TEST(SpecialMiscTest, jacobi_ratios) {
    const double u = 0.5;
    const double k = 0.5;
    EXPECT_NEAR(jacobi_am(u, k), std::atan2(jacobi_sn(u, k), jacobi_cn(u, k)), 1e-3);
    EXPECT_NEAR(jacobi_nd(u, k), jacobi_cn(u, k) / jacobi_dn(u, k), 1e-6);
    EXPECT_NEAR(jacobi_nc(u, k), jacobi_cn(u, k) / jacobi_sn(u, k), 1e-6);
}
