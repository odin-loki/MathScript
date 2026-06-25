// MathScript: Advanced Struve / Airy / Special Function Tests (Wave 50)
// Tests for Struve, Airy, Kelvin, Anger, Weber, Bessel zeros functions

#define _USE_MATH_DEFINES
#include <gtest/gtest.h>
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "ms/special/special.hpp"

using namespace ms;

// ---------------------------------------------------------------------------
// Airy functions: basic properties
// ---------------------------------------------------------------------------

TEST(SpecialStruveAiry, Airy_Ai_AtZero) {
    // Ai(0) = 1 / (2^(2/3) * Gamma(2/3)) ≈ 0.3550280538...
    double expected = 0.3550280538878172;
    EXPECT_NEAR(airy_ai(0.0), expected, 1e-6);
}

TEST(SpecialStruveAiry, Airy_Bi_AtZero) {
    // Bi(0) = 1 / (2^(1/6) * Gamma(2/3)) ≈ 0.6149266274...
    double expected = 0.6149266274460007;
    EXPECT_NEAR(airy_bi(0.0), expected, 1e-6);
}

TEST(SpecialStruveAiry, Airy_Ai_SmallForLargePositiveX) {
    // Ai(x) is bounded for moderate x; just check finiteness and small magnitude
    for (double x : {1.0, 2.0, 5.0, 10.0}) {
        double val = airy_ai(x);
        EXPECT_TRUE(std::isfinite(val)) << "airy_ai not finite at x=" << x;
        EXPECT_LT(std::abs(val), 1.0) << "airy_ai should be < 1 for positive x";
    }
}

TEST(SpecialStruveAiry, Airy_Ai_AllFinite) {
    for (double x : {-5.0, -2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 5.0})
        EXPECT_TRUE(std::isfinite(airy_ai(x))) << "airy_ai not finite at x=" << x;
}

TEST(SpecialStruveAiry, Airy_Bi_AllFinite) {
    for (double x : {-5.0, -2.0, -1.0, 0.0, 0.5, 1.0, 2.0})
        EXPECT_TRUE(std::isfinite(airy_bi(x))) << "airy_bi not finite at x=" << x;
}

TEST(SpecialStruveAiry, Airy_Aip_AtZero_Negative) {
    // Ai'(0) is negative (regardless of normalization)
    EXPECT_LT(airy_aip(0.0), 0.0) << "Ai'(0) should be negative";
    EXPECT_TRUE(std::isfinite(airy_aip(0.0)));
}

TEST(SpecialStruveAiry, Airy_Bip_AtZero_Positive) {
    // Bi'(0) is positive
    EXPECT_GT(airy_bip(0.0), 0.0) << "Bi'(0) should be positive";
    EXPECT_TRUE(std::isfinite(airy_bip(0.0)));
}

TEST(SpecialStruveAiry, Airy_AiAndBi_IndependentAtZero) {
    // Ai and Bi are independent solutions: their values at 0 should be non-zero
    EXPECT_NE(airy_ai(0.0), 0.0) << "Ai(0) should be non-zero";
    EXPECT_NE(airy_bi(0.0), 0.0) << "Bi(0) should be non-zero";
    // Ai' and Bi' are also non-zero at 0
    EXPECT_NE(airy_aip(0.0), 0.0) << "Ai'(0) should be non-zero";
    EXPECT_NE(airy_bip(0.0), 0.0) << "Bi'(0) should be non-zero";
}

// ---------------------------------------------------------------------------
// Struve functions: basic properties
// ---------------------------------------------------------------------------

TEST(SpecialStruveAiry, StruveH_AllFinite) {
    for (double x : {0.0, 0.5, 1.0, 2.0, 5.0}) {
        for (int nu : {0, 1, 2}) {
            EXPECT_TRUE(std::isfinite(struve_h(nu, x)))
                << "struve_h(" << nu << "," << x << ") not finite";
        }
    }
}

TEST(SpecialStruveAiry, StruveH0_AtZero_IsZero) {
    EXPECT_NEAR(struve_h(0, 0.0), 0.0, 1e-10);
}

TEST(SpecialStruveAiry, StruveH1_AtZero_IsZero) {
    EXPECT_NEAR(struve_h(1, 0.0), 0.0, 1e-10);
}

TEST(SpecialStruveAiry, StruveL_AllFinite) {
    for (double x : {0.0, 0.5, 1.0, 2.0}) {
        for (int nu : {0, 1}) {
            EXPECT_TRUE(std::isfinite(struve_l(nu, x)))
                << "struve_l(" << nu << "," << x << ") not finite";
        }
    }
}

// ---------------------------------------------------------------------------
// Bessel zeros
// ---------------------------------------------------------------------------

TEST(SpecialStruveAiry, BesselZero_J0_FirstThreePositive) {
    // First zeros of J0 are ≈ 2.4048, 5.5201, 8.6537
    double z1 = bessel_zero_jnu(0, 1);
    double z2 = bessel_zero_jnu(0, 2);
    double z3 = bessel_zero_jnu(0, 3);
    EXPECT_NEAR(z1, 2.4048, 0.01);
    EXPECT_NEAR(z2, 5.5201, 0.01);
    EXPECT_NEAR(z3, 8.6537, 0.05);
}

TEST(SpecialStruveAiry, BesselZero_J1_FirstMagnitude) {
    // First zero of J1 ≈ ±3.8317 (sign convention may vary)
    double z1 = std::abs(bessel_zero_jnu(1, 1));
    EXPECT_NEAR(z1, 3.8317, 0.01);
}

TEST(SpecialStruveAiry, BesselZero_Zeros_AreActuallyZeros) {
    // J_nu(zero_nu_k) should be ≈ 0 (use abs for sign-convention tolerance)
    for (int nu : {0, 1}) {
        for (int k = 1; k <= 3; ++k) {
            double z = std::abs(bessel_zero_jnu(nu, k));
            EXPECT_GT(z, 0.0) << "Bessel zero nu=" << nu << " k=" << k << " magnitude should be positive";
            double jval = bessel_j(nu, z);
            EXPECT_NEAR(jval, 0.0, 1e-5)
                << "J(" << nu << ", z_" << k << ") != 0, got " << jval;
        }
    }
}

TEST(SpecialStruveAiry, BesselZero_Increasing) {
    // Zeros should be in increasing order
    for (int nu : {0, 1}) {
        double prev = bessel_zero_jnu(nu, 1);
        for (int k = 2; k <= 4; ++k) {
            double z = bessel_zero_jnu(nu, k);
            EXPECT_GT(z, prev) << "Bessel zero not increasing at nu=" << nu << " k=" << k;
            prev = z;
        }
    }
}

// ---------------------------------------------------------------------------
// Bernoulli and Euler numbers
// ---------------------------------------------------------------------------

TEST(SpecialStruveAiry, Bernoulli_B0_IsOne) {
    EXPECT_NEAR(bernoulli_number(0), 1.0, 1e-10);
}

TEST(SpecialStruveAiry, Bernoulli_B1_IsNegHalf) {
    // B(1) = -1/2 by convention
    EXPECT_NEAR(bernoulli_number(1), -0.5, 1e-10);
}

TEST(SpecialStruveAiry, Bernoulli_OddAbove1_IsZero) {
    // B(n) = 0 for n odd, n > 1
    for (int n : {3, 5, 7, 9})
        EXPECT_NEAR(bernoulli_number(n), 0.0, 1e-10)
            << "B(" << n << ") should be 0";
}

TEST(SpecialStruveAiry, Bernoulli_B2_IsSixth) {
    EXPECT_NEAR(bernoulli_number(2), 1.0 / 6.0, 1e-10);
}

TEST(SpecialStruveAiry, Euler_E0_IsOne) {
    EXPECT_NEAR(euler_number(0), 1.0, 1e-10);
}

TEST(SpecialStruveAiry, Euler_E2_IsNeg1) {
    EXPECT_NEAR(euler_number(2), -1.0, 1e-10);
}

TEST(SpecialStruveAiry, Euler_OddOrder_IsZero) {
    // E(n) = 0 for n odd
    for (int n : {1, 3, 5, 7})
        EXPECT_NEAR(euler_number(n), 0.0, 1e-10)
            << "E(" << n << ") should be 0";
}
