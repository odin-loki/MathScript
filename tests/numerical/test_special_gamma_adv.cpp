// MathScript: Advanced Gamma/Beta/Digamma Tests (Wave 47)
// Tests for gamma function identities, beta function, digamma, log_gamma

#define _USE_MATH_DEFINES
#include <gtest/gtest.h>
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "ms/special/special.hpp"

using namespace ms;

// ---------------------------------------------------------------------------
// gamma_func: additional identities
// ---------------------------------------------------------------------------

TEST(SpecialGammaAdv, Gamma_Positive_AllFinite) {
    for (double x : {0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0}) {
        EXPECT_TRUE(std::isfinite(gamma_func(x))) << "gamma not finite at x=" << x;
    }
}

TEST(SpecialGammaAdv, Gamma_StirlingApprox_LargeX) {
    // For large x, Gamma(x) ~ sqrt(2*pi/x) * (x/e)^x
    // At x=10: Gamma(10) = 9! = 362880
    EXPECT_NEAR(gamma_func(10.0), 362880.0, 10.0);
}

TEST(SpecialGammaAdv, Gamma_Reflection_Formula) {
    // Gamma(x) * Gamma(1-x) = pi / sin(pi*x)  for non-integer x
    for (double x : {0.3, 0.7, 0.2, 0.8}) {
        double lhs = gamma_func(x) * gamma_func(1.0 - x);
        double rhs = M_PI / std::sin(M_PI * x);
        EXPECT_NEAR(lhs, rhs, 1e-9) << "Reflection formula failed at x=" << x;
    }
}

TEST(SpecialGammaAdv, Gamma_Recurrence_ForHalfIntegers) {
    // Gamma(n + 1/2) = (2n-1)!! * sqrt(pi) / 2^n
    // Gamma(1/2) = sqrt(pi)
    // Gamma(3/2) = sqrt(pi)/2
    // Gamma(5/2) = 3*sqrt(pi)/4
    EXPECT_NEAR(gamma_func(0.5), std::sqrt(M_PI), 1e-10);
    EXPECT_NEAR(gamma_func(1.5), std::sqrt(M_PI) / 2.0, 1e-10);
    EXPECT_NEAR(gamma_func(2.5), 3.0 * std::sqrt(M_PI) / 4.0, 1e-9);
}

TEST(SpecialGammaAdv, Gamma_GrowsMonotonically_PositiveBeyond1) {
    // For x >= 2, Gamma(x) is monotone increasing
    for (double x = 2.0; x < 7.0; x += 0.5) {
        EXPECT_LT(gamma_func(x), gamma_func(x + 0.5))
            << "Gamma not monotone at x=" << x;
    }
}

// ---------------------------------------------------------------------------
// log_gamma
// ---------------------------------------------------------------------------

TEST(SpecialGammaAdv, LogGamma_MatchesLogOfGamma) {
    for (double x : {0.5, 1.0, 2.0, 3.0, 5.0}) {
        double direct = std::log(gamma_func(x));
        double lg = log_gamma(x);
        EXPECT_NEAR(lg, direct, 1e-9) << "log_gamma mismatch at x=" << x;
    }
}

TEST(SpecialGammaAdv, LogGamma_Positive_ForX_GT_1) {
    // log_gamma(x) >= 0 for x >= 1 (since gamma >= 1 for integer x >= 1)
    for (double x : {2.0, 3.0, 4.0, 5.0}) {
        EXPECT_GE(log_gamma(x), 0.0) << "log_gamma should be >= 0 for x >= 2";
    }
}

TEST(SpecialGammaAdv, LogGamma_AllFinite) {
    for (double x : {0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0}) {
        EXPECT_TRUE(std::isfinite(log_gamma(x))) << "log_gamma not finite at x=" << x;
    }
}

// ---------------------------------------------------------------------------
// beta_func
// ---------------------------------------------------------------------------

TEST(SpecialGammaAdv, Beta_Symmetric) {
    // B(a, b) = B(b, a)
    for (auto [a, b] : std::vector<std::pair<double,double>>{{0.5, 1.5}, {1.0, 2.0}, {2.0, 3.0}}) {
        EXPECT_NEAR(beta_func(a, b), beta_func(b, a), 1e-12)
            << "Beta not symmetric for a=" << a << " b=" << b;
    }
}

TEST(SpecialGammaAdv, Beta_OneArgOne_IsReciprocal) {
    // B(1, b) = 1/b for b > 0
    for (double b : {1.0, 2.0, 3.0, 5.0}) {
        EXPECT_NEAR(beta_func(1.0, b), 1.0 / b, 1e-10)
            << "B(1,b) != 1/b at b=" << b;
    }
}

TEST(SpecialGammaAdv, Beta_RelationToGamma) {
    // B(a, b) = Gamma(a) * Gamma(b) / Gamma(a+b)
    for (auto [a, b] : std::vector<std::pair<double,double>>{{0.5, 0.5}, {1.0, 2.0}, {2.0, 3.0}}) {
        double expected = gamma_func(a) * gamma_func(b) / gamma_func(a + b);
        EXPECT_NEAR(beta_func(a, b), expected, 1e-9)
            << "Beta vs Gamma relation failed at a=" << a << " b=" << b;
    }
}

TEST(SpecialGammaAdv, Beta_HalfHalf_IsPi) {
    // B(1/2, 1/2) = pi
    EXPECT_NEAR(beta_func(0.5, 0.5), M_PI, 1e-9);
}

TEST(SpecialGammaAdv, Beta_Positive_ForPositiveArgs) {
    for (auto [a, b] : std::vector<std::pair<double,double>>{{0.5, 1.5}, {1.0, 1.0}, {2.0, 2.0}}) {
        EXPECT_GT(beta_func(a, b), 0.0);
    }
}

// ---------------------------------------------------------------------------
// digamma
// ---------------------------------------------------------------------------

TEST(SpecialGammaAdv, Digamma_AtOne_IsNegEulerMascheroni) {
    // psi(1) = -gamma_EM ≈ -0.5772156649
    double euler_mascheroni = 0.5772156649015328;
    EXPECT_NEAR(digamma(1.0), -euler_mascheroni, 1e-8);
}

TEST(SpecialGammaAdv, Digamma_RecurrenceRelation) {
    // psi(x+1) = psi(x) + 1/x
    for (double x : {0.5, 1.0, 1.5, 2.0, 3.0}) {
        EXPECT_NEAR(digamma(x + 1.0), digamma(x) + 1.0 / x, 1e-9)
            << "Digamma recurrence failed at x=" << x;
    }
}

TEST(SpecialGammaAdv, Digamma_IntegerValues) {
    // psi(n) = H_{n-1} - gamma_EM (harmonic number)
    // psi(2) = 1 - gamma_EM ≈ 0.4228
    double euler_mascheroni = 0.5772156649015328;
    EXPECT_NEAR(digamma(2.0), 1.0 - euler_mascheroni, 1e-8);
    // psi(3) = 1 + 1/2 - gamma_EM ≈ 0.9228
    EXPECT_NEAR(digamma(3.0), 1.5 - euler_mascheroni, 1e-8);
}

TEST(SpecialGammaAdv, Digamma_Monotone_Increasing) {
    for (double x = 1.0; x < 5.0; x += 0.5) {
        EXPECT_LT(digamma(x), digamma(x + 0.5))
            << "Digamma should be monotone increasing";
    }
}

TEST(SpecialGammaAdv, Digamma_AllFinite) {
    for (double x : {0.5, 1.0, 2.0, 5.0, 10.0}) {
        EXPECT_TRUE(std::isfinite(digamma(x)));
    }
}
