#include <gtest/gtest.h>
#include <cmath>
#include "ms/special/special.hpp"

using namespace ms;

TEST(SpecialThetaTest, jacobi_theta_functions) {
    const double q = 0.3;
    const double z = 0.5;
    EXPECT_NEAR(theta1(z, q), 0.5773940463248446, 1e-3);
    EXPECT_NEAR(theta2(z, q), 1.3075255735032947, 1e-3);
    EXPECT_NEAR(theta3(z, q), 1.317400827096804, 1e-3);
    EXPECT_NEAR(theta4(z, q), 0.6691160041441827, 1e-3);
    EXPECT_NEAR(theta1_prime(z, q), 1.2663806017958978, 1e-3);
    EXPECT_NEAR(jacobi_theta(3, z, 0.2), 1.5020847332591976, 1e-3);
}

TEST(SpecialWeierstrassTest, short_series) {
    // AUDIT FIX: these four values used to encode the WRONG Laurent coefficients (g2^2 z^6/960
    // in P instead of /1200, and a weierstrass_zeta whose leading correction had the wrong power
    // AND sign). They are now the values of the DLMF 23.9.3 series, and the tolerances are
    // tightened from 1e-3/1e-2 to 1e-12 -- the old numbers no longer pass for zeta.
    const double z = 0.5;
    const double g2 = 1.0;
    const double g3 = 0.0;
    EXPECT_NEAR(weierstrass_p(z, g2, g3), 4.0125130270962277, 1e-12);
    EXPECT_NEAR(weierstrass_pprime(z, g2, g3), -15.949843624719085, 1e-11);
    EXPECT_NEAR(weierstrass_zeta(z, g2, g3), 1.9979157363225009, 1e-12);
    EXPECT_NEAR(weierstrass_sigma(z, g2, g3), 0.49986977955668571, 1e-13);
}

TEST(SpecialZetaTest, zeta_and_related) {
    EXPECT_NEAR(zeta(2.0), 1.6449340668482264, 1e-9);
    EXPECT_NEAR(zeta(0.5), -1.4603545088095868, 5e-3);
    EXPECT_NEAR(zeta_hurwitz(2.0, 0.3), 12.245364546107732, 1e-3);
    EXPECT_NEAR(eta_dirichlet(2.0), 0.8224670334241132, 1e-6);
    EXPECT_NEAR(beta_dirichlet(2.0), 0.915965594127219, 1e-3);
    EXPECT_NEAR(polylog(2, 0.5), 0.5822405264650125, 1e-3);
    EXPECT_NEAR(clausen(0.5), 0.8483118777036792, 1e-3);
    EXPECT_NEAR(lerch_phi(0.5, 2.0, 0.3), 11.47083462974499, 1e-3);
}

TEST(SpecialMathieuTest, characteristic_and_periodic) {
    const double q = 0.1;
    // AUDIT FIX: tolerances tightened from 1e-3 / 7e-2 to 1e-9 now that the four Mathieu
    // recurrence matrices are built correctly instead of one matrix patched from outside.
    EXPECT_NEAR(mathieu_a(1, q), 1.0987343129634084, 1e-9);
    EXPECT_NEAR(mathieu_b(1, q), 0.8987655569943626, 1e-9);
    EXPECT_NEAR(mathieu_ce(1, q, 0.5), 0.8765747185555349, 1e-9);
}

TEST(SpecialMathieuTest, modified_functions) {
    // AUDIT FIX: the two numbers previously pinned here (0.6877267249062208 and
    // 0.40462549819476684) were the hard-coded literals that modified_mathieu_origin returned
    // for exactly this (n = 1, q = 1) input; every other argument got the Fourier value times an
    // unexplained 0.472 / 0.803. With Mc_n(z,q) = ce_n(iz,q) and Ms_n(z,q) = -i se_n(iz,q),
    // Mc(0) is ce(0) and Ms(0) is 0 -- and the function is continuous in q.
    const double q = 1.0;
    EXPECT_NEAR(mathieu_mc(1, q, 0.0), mathieu_ce(1, q, 0.0), 1e-13);
    EXPECT_NEAR(mathieu_mc(1, q, 0.0), 0.85659846555688657, 1e-9);
    EXPECT_NEAR(mathieu_ms(1, q, 0.0), 0.0, 1e-15);
    EXPECT_NEAR(mathieu_mc(1, q + 1e-9, 0.0), mathieu_mc(1, q, 0.0), 1e-6);
}

TEST(SpecialWaveTest, spheroidal_and_parabolic) {
    // AUDIT FIX: the previous spheroidal expectations came from the magic-number curve fit
    // `n(n+1) - c^2 + 3.0986774 c - 0.0146127` and from spheroidal_s1's hard-coded 0.043; both
    // are now the eigenvalue/eigenvector of the real Bouwkamp tridiagonal problem. The pcf_w
    // expectation asserted the old U/V mix, which is not the oscillatory W of DLMF 12.14.
    EXPECT_NEAR(spheroidal_lambda(1, 1, 5.0), 5.3504222984641334, 1e-9);
    EXPECT_NEAR(spheroidal_s1(1, 1, 5.0, 0.5), -0.56016304181378862, 1e-9);
    EXPECT_NEAR(pcf_w(0.5, 1.0), 0.47413985718435603, 1e-9);
    // W solves y'' + (x^2/4 - a) y = 0; U solves y'' - (x^2/4 + a) y = 0.
    const double h = 1e-4;
    const double w2 = (pcf_w(0.5, 1.0 + h) - 2.0 * pcf_w(0.5, 1.0) + pcf_w(0.5, 1.0 - h)) / (h * h);
    EXPECT_NEAR(w2, (0.5 - 0.25 * 1.0) * pcf_w(0.5, 1.0), 1e-6);
}
