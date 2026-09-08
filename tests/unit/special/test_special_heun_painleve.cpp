#include <gtest/gtest.h>
#include "ms/special/special.hpp"

using namespace ms;

TEST(SpecialHeunTest, general_and_confluent) {
    const double a = 0.5;
    const double q = 0.1;
    const double alpha = 0.2;
    const double beta = 0.3;
    const double gamma = 0.4;
    const double delta = 0.5;
    const double z = 0.2;

    // AUDIT FIX: every one of these five values came from integrating an equation OUTSIDE the
    // Heun family (heun_g was missing the Fuchs-determined epsilon/(z-a) pole; the three
    // confluent members each kept a 1/(z-1) pole they must not have). They are re-pinned against
    // the DLMF 31.1.1 / 31.12.1-31.12.4 equations, whose residuals are checked directly in
    // tests/unit/special/test_special_audit_fixes.cpp.
    EXPECT_NEAR(heun_g(a, q, alpha, beta, gamma, delta, z), 1.1275084129139172, 1e-9);
    EXPECT_NEAR(heun_c(q, alpha, beta, gamma, delta, z), 0.95102150922133522, 1e-9);
    EXPECT_NEAR(heun_d(q, alpha * beta, gamma, delta, z), 1.0849641043699225, 1e-9);
    EXPECT_NEAR(heun_b(q, alpha, beta, delta, z), 1.0951885848132246, 1e-9);
    EXPECT_NEAR(heun_t(q, alpha, beta, gamma, z), 1.0017871617706366, 1e-9);
    EXPECT_NEAR(heun_g(a, q, alpha, beta, gamma, delta, 1e-6), 1.0, 1e-12);
}

TEST(SpecialPainleveTest, first_and_second) {
    EXPECT_NEAR(painleve1(0.5, 0.0, 0.0), -0.020821712245422376, 1e-6);
    EXPECT_NEAR(painleve1(1.0, 0.0, 0.0), -0.16372821373317917, 1e-6);
    EXPECT_NEAR(painleve2(0.5, 0.0, -0.5, 0.0), -0.2530083397481052, 1e-6);
    EXPECT_NEAR(painleve1(0.0, 0.0, 0.0), 0.0, 1e-15);
}

TEST(SpecialPainleveTest, third_through_sixth) {
    // AUDIT FIX: the previous values solved right-hand sides that were not the Painleve
    // equations at all (PIII had (alpha w^2 + beta)/z^2 and no gamma w^3 + delta/w; PIV had the
    // wrong coefficient on w'^2 and none of its cubic/quadratic terms; PV and PVI omitted their
    // defining pole structure). Re-pinned against DLMF 32.2.3-32.2.6, whose residuals and known
    // exact solutions are checked in test_special_audit_fixes.cpp.
    EXPECT_NEAR(painleve3(0.5, 0.5, -0.1, 0.5, 0.3), 0.48771921611212105, 1e-9);
    EXPECT_NEAR(painleve4(0.5, 0.8, -0.05, 0.2, 0.4), 0.91945144765805531, 1e-9);
    EXPECT_NEAR(painleve5(0.5, 0.5, -0.05, 0.01, 0.02, 0.03, 0.04), 0.49531801548760035, 1e-9);
    EXPECT_NEAR(painleve6(2.5, 0.5, -0.05, 0.1, 0.2, 0.3, 0.4), 0.49834436151961337, 1e-9);
}
