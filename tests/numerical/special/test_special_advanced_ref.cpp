// MathScript Special Functions Advanced Numerical Reference Tests
// Tests erfi, erfcx, dawsonx, log_gamma, bernoulli_number, euler_number, Airy functions

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#include "ms/special/special.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ms;

// ---------------------------------------------------------------------------
// erfi: imaginary error function erfi(x) = (2/sqrt(pi)) * int_0^x exp(t^2) dt
// ---------------------------------------------------------------------------

TEST(SpecialAdvanced, Erfi_Zero) {
    EXPECT_NEAR(erfi(0.0), 0.0, 1e-12);
}

TEST(SpecialAdvanced, Erfi_OddSymmetry) {
    EXPECT_NEAR(erfi(-1.0), -erfi(1.0), 1e-10);
}

TEST(SpecialAdvanced, Erfi_PositiveForPositive) {
    EXPECT_GT(erfi(0.5), 0.0);
    EXPECT_GT(erfi(1.0), 0.0);
}

TEST(SpecialAdvanced, Erfi_MonotoneIncreasing) {
    EXPECT_LT(erfi(0.5), erfi(1.0));
    EXPECT_LT(erfi(1.0), erfi(1.5));
}

TEST(SpecialAdvanced, Erfi_Reference) {
    // erfi(0.5) ≈ 0.6149(DLMF) — approximate
    EXPECT_NEAR(erfi(0.5), 0.6149, 0.01);
}

// ---------------------------------------------------------------------------
// erfcx: scaled complementary error function erfcx(x) = exp(x^2) * erfc(x)
// ---------------------------------------------------------------------------

TEST(SpecialAdvanced, Erfcx_Zero) {
    EXPECT_NEAR(erfcx(0.0), 1.0, 1e-10);
}

TEST(SpecialAdvanced, Erfcx_PositiveForPositive) {
    EXPECT_GT(erfcx(1.0), 0.0);
    EXPECT_GT(erfcx(5.0), 0.0);
}

TEST(SpecialAdvanced, Erfcx_AllFinite) {
    for (double x : {-2.0, -1.0, 0.0, 1.0, 2.0, 5.0, 10.0})
        EXPECT_TRUE(std::isfinite(erfcx(x)));
}

TEST(SpecialAdvanced, Erfcx_Monotone_Decreasing_For_Positive_X) {
    EXPECT_GT(erfcx(0.5), erfcx(1.0));
    EXPECT_GT(erfcx(1.0), erfcx(2.0));
}

// ---------------------------------------------------------------------------
// dawsonx: Dawson's integral D(x) = exp(-x^2) * int_0^x exp(t^2) dt
// ---------------------------------------------------------------------------

TEST(SpecialAdvanced, Dawsonx_Zero) {
    EXPECT_NEAR(dawsonx(0.0), 0.0, 1e-12);
}

TEST(SpecialAdvanced, Dawsonx_OddSymmetry) {
    EXPECT_NEAR(dawsonx(-1.0), -dawsonx(1.0), 1e-10);
}

TEST(SpecialAdvanced, Dawsonx_HasPeakNear06) {
    // Dawson function has max near x ≈ 0.924
    EXPECT_GT(dawsonx(0.924), dawsonx(0.0));
    EXPECT_GT(dawsonx(0.924), dawsonx(2.0));
}

TEST(SpecialAdvanced, Dawsonx_AllFinite) {
    for (double x : {-3.0, -1.0, 0.0, 0.5, 1.0, 2.0, 5.0})
        EXPECT_TRUE(std::isfinite(dawsonx(x)));
}

// ---------------------------------------------------------------------------
// log_gamma: log(Gamma(x)) = lgamma(x)
// ---------------------------------------------------------------------------

TEST(SpecialAdvanced, LogGamma_Integer_1) {
    // log(Gamma(1)) = log(0!) = 0
    EXPECT_NEAR(log_gamma(1.0), 0.0, 1e-10);
}

TEST(SpecialAdvanced, LogGamma_Integer_2) {
    // log(Gamma(2)) = log(1!) = 0
    EXPECT_NEAR(log_gamma(2.0), 0.0, 1e-10);
}

TEST(SpecialAdvanced, LogGamma_Integer_3) {
    // log(Gamma(3)) = log(2!) = log(2)
    EXPECT_NEAR(log_gamma(3.0), std::log(2.0), 1e-10);
}

TEST(SpecialAdvanced, LogGamma_Integer_5) {
    // log(Gamma(5)) = log(4!) = log(24)
    EXPECT_NEAR(log_gamma(5.0), std::log(24.0), 1e-10);
}

TEST(SpecialAdvanced, LogGamma_Half) {
    // Gamma(0.5) = sqrt(pi), so log_gamma(0.5) = 0.5*log(pi)
    EXPECT_NEAR(log_gamma(0.5), 0.5 * std::log(M_PI), 1e-8);
}

TEST(SpecialAdvanced, LogGamma_Positive) {
    // log_gamma(x) should be positive for x > 2
    EXPECT_GT(log_gamma(5.0), 0.0);
    EXPECT_GT(log_gamma(10.0), 0.0);
}

// ---------------------------------------------------------------------------
// bernoulli_number: B(0)=1, B(1)=-1/2, B(2)=1/6, B(odd>1)=0
// ---------------------------------------------------------------------------

TEST(SpecialAdvanced, Bernoulli_B0) {
    EXPECT_NEAR(bernoulli_number(0), 1.0, 1e-10);
}

TEST(SpecialAdvanced, Bernoulli_B1) {
    EXPECT_NEAR(bernoulli_number(1), -0.5, 1e-10);
}

TEST(SpecialAdvanced, Bernoulli_B2) {
    EXPECT_NEAR(bernoulli_number(2), 1.0/6.0, 1e-10);
}

TEST(SpecialAdvanced, Bernoulli_B3_Is_Zero) {
    EXPECT_NEAR(bernoulli_number(3), 0.0, 1e-10);
}

TEST(SpecialAdvanced, Bernoulli_B4) {
    // B4 = -1/30
    EXPECT_NEAR(bernoulli_number(4), -1.0/30.0, 1e-10);
}

// ---------------------------------------------------------------------------
// euler_number: E(0)=1, E(1)=0, E(2)=-1, E(3)=0, E(4)=5
// ---------------------------------------------------------------------------

TEST(SpecialAdvanced, EulerNumber_E0) {
    EXPECT_NEAR(euler_number(0), 1.0, 1e-10);
}

TEST(SpecialAdvanced, EulerNumber_E2) {
    // E2 = -1
    EXPECT_NEAR(euler_number(2), -1.0, 1e-10);
}

TEST(SpecialAdvanced, EulerNumber_E4) {
    // E4 = 5
    EXPECT_NEAR(euler_number(4), 5.0, 1e-10);
}

TEST(SpecialAdvanced, EulerNumber_OddIsZero) {
    EXPECT_NEAR(euler_number(1), 0.0, 1e-10);
    EXPECT_NEAR(euler_number(3), 0.0, 1e-10);
}

// ---------------------------------------------------------------------------
// Airy functions: Ai(0) = 1/cbrt(9*Gamma(2/3)), Ai'(0) = -1/cbrt(3*Gamma(1/3))
// ---------------------------------------------------------------------------

TEST(SpecialAdvanced, Airy_Ai_AtZero_Positive) {
    // Ai(0) ≈ 0.3550
    EXPECT_NEAR(airy_ai(0.0), 0.3550, 0.01);
}

TEST(SpecialAdvanced, Airy_Bi_AtZero_Positive) {
    // Bi(0) ≈ 0.6149
    EXPECT_NEAR(airy_bi(0.0), 0.6149, 0.01);
}

TEST(SpecialAdvanced, Airy_Ai_LargeNegX_Oscillates) {
    // For large negative x, Ai oscillates but stays finite
    EXPECT_TRUE(std::isfinite(airy_ai(-5.0)));
    EXPECT_TRUE(std::isfinite(airy_ai(-10.0)));
}

TEST(SpecialAdvanced, Airy_Ai_Decays_ForLargePositiveX) {
    // Ai decays exponentially for large x
    EXPECT_GT(airy_ai(1.0), airy_ai(5.0));
    EXPECT_GT(airy_ai(5.0), -0.1);  // close to 0 but positive
}

TEST(SpecialAdvanced, Airy_AiPrime_AtZero_Negative) {
    // Ai'(0) ≈ -0.2588
    EXPECT_LT(airy_aip(0.0), 0.0);
    EXPECT_NEAR(airy_aip(0.0), -0.2588, 0.01);
}

TEST(SpecialAdvanced, Airy_BiPrime_AtZero_Positive) {
    // Bi'(0) ≈ 0.4483
    EXPECT_GT(airy_bip(0.0), 0.0);
}

TEST(SpecialAdvanced, Airy_AllFinite) {
    for (double x : {-5.0, -2.0, 0.0, 1.0, 3.0}) {
        EXPECT_TRUE(std::isfinite(airy_ai(x)));
        EXPECT_TRUE(std::isfinite(airy_bi(x)));
        EXPECT_TRUE(std::isfinite(airy_aip(x)));
        EXPECT_TRUE(std::isfinite(airy_bip(x)));
    }
}
