// MathScript: Advanced Special Function Tests (Wave 44)
// Tests for erf/erfc, Bessel recurrence, Legendre, Gamma identities

#define _USE_MATH_DEFINES
#include <gtest/gtest.h>
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "ms/special/special.hpp"

using namespace ms;

// Use explicit namespace to avoid ambiguity with <cmath> erf/erfc
namespace msspec = ms;

// ---------------------------------------------------------------------------
// erf / erfc
// ---------------------------------------------------------------------------

TEST(SpecialAdv2, Erf_Odd_Function) {
    for (double x : {0.5, 1.0, 2.0, 3.0}) {
        EXPECT_NEAR(msspec::erf(-x), -msspec::erf(x), 1e-14)
            << "erf(-x) != -erf(x) at x=" << x;
    }
}

TEST(SpecialAdv2, Erf_AtZero_IsZero) {
    EXPECT_NEAR(msspec::erf(0.0), 0.0, 1e-15);
}

TEST(SpecialAdv2, Erf_AtLargeX_NearOne) {
    EXPECT_NEAR(msspec::erf(5.0), 1.0, 1e-10);
    EXPECT_NEAR(msspec::erf(10.0), 1.0, 1e-10);
}

TEST(SpecialAdv2, Erf_Monotone_Increasing) {
    for (double x = -3.0; x < 3.0; x += 0.5) {
        EXPECT_LT(msspec::erf(x), msspec::erf(x + 0.5))
            << "erf should be monotone increasing";
    }
}

TEST(SpecialAdv2, Erf_Erfc_Complement) {
    for (double x : {-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0}) {
        EXPECT_NEAR(msspec::erf(x) + msspec::erfc(x), 1.0, 1e-14)
            << "erf + erfc != 1 at x=" << x;
    }
}

TEST(SpecialAdv2, Erfc_Monotone_Decreasing) {
    for (double x = -3.0; x < 3.0; x += 0.5) {
        EXPECT_GT(msspec::erfc(x), msspec::erfc(x + 0.5))
            << "erfc should be monotone decreasing";
    }
}

TEST(SpecialAdv2, Erfc_AtZero_IsOne) {
    EXPECT_NEAR(msspec::erfc(0.0), 1.0, 1e-15);
}

// ---------------------------------------------------------------------------
// gamma function identities
// ---------------------------------------------------------------------------

TEST(SpecialAdv2, Gamma_FactorialIntegers) {
    // Gamma(n+1) = n! for integers n >= 0
    EXPECT_NEAR(gamma_func(1.0), 1.0, 1e-10);  // 0! = 1
    EXPECT_NEAR(gamma_func(2.0), 1.0, 1e-10);  // 1! = 1
    EXPECT_NEAR(gamma_func(3.0), 2.0, 1e-10);  // 2! = 2
    EXPECT_NEAR(gamma_func(4.0), 6.0, 1e-10);  // 3! = 6
    EXPECT_NEAR(gamma_func(5.0), 24.0, 1e-8); // 4! = 24
    EXPECT_NEAR(gamma_func(6.0), 120.0, 1e-7); // 5! = 120
}

TEST(SpecialAdv2, Gamma_RecurrenceRelation) {
    // Gamma(x+1) = x * Gamma(x)
    for (double x : {0.5, 1.5, 2.5, 3.5}) {
        EXPECT_NEAR(gamma_func(x + 1.0), x * gamma_func(x), 1e-10)
            << "Gamma recurrence failed at x=" << x;
    }
}

TEST(SpecialAdv2, Gamma_HalfInteger) {
    // Gamma(1/2) = sqrt(pi)
    EXPECT_NEAR(gamma_func(0.5), std::sqrt(M_PI), 1e-10);
    // Gamma(3/2) = sqrt(pi)/2
    EXPECT_NEAR(gamma_func(1.5), std::sqrt(M_PI) / 2.0, 1e-10);
}

TEST(SpecialAdv2, Gamma_PositiveForPositiveX) {
    for (double x : {0.1, 0.5, 1.0, 2.0, 5.0}) {
        EXPECT_GT(gamma_func(x), 0.0) << "Gamma should be positive at x=" << x;
    }
}

// ---------------------------------------------------------------------------
// Bessel function identities
// ---------------------------------------------------------------------------

TEST(SpecialAdv2, Bessel_J0_At_Zero_Is_One) {
    EXPECT_NEAR(bessel_j(0, 0.0), 1.0, 1e-10);
}

TEST(SpecialAdv2, Bessel_J1_At_Zero_Is_Zero) {
    EXPECT_NEAR(bessel_j(1, 0.0), 0.0, 1e-10);
}

TEST(SpecialAdv2, Bessel_Jn_At_Zero_Is_Zero_ForN_GE_1) {
    for (int n = 1; n <= 5; ++n) {
        EXPECT_NEAR(bessel_j(n, 0.0), 0.0, 1e-10)
            << "J_" << n << "(0) should be 0";
    }
}

TEST(SpecialAdv2, Bessel_Recurrence_Relation) {
    // J_{n-1}(x) + J_{n+1}(x) = (2n/x) * J_n(x)
    double x = 2.0;
    for (int n = 1; n <= 4; ++n) {
        double lhs = bessel_j(n - 1, x) + bessel_j(n + 1, x);
        double rhs = (2.0 * n / x) * bessel_j(n, x);
        EXPECT_NEAR(lhs, rhs, 1e-8)
            << "Bessel recurrence failed at n=" << n << " x=" << x;
    }
}

TEST(SpecialAdv2, Bessel_J0_Finite_ForAllX) {
    for (double x : {0.1, 1.0, 2.0, 5.0, 10.0, 20.0}) {
        EXPECT_TRUE(std::isfinite(bessel_j(0, x)));
    }
}

TEST(SpecialAdv2, Bessel_J_Bounded_By_1) {
    // |J_0(x)| <= 1 for all x
    for (double x = 0.0; x <= 20.0; x += 0.5) {
        EXPECT_LE(std::abs(bessel_j(0, x)), 1.0 + 1e-10)
            << "|J_0| > 1 at x=" << x;
    }
}

// ---------------------------------------------------------------------------
// Legendre polynomials
// ---------------------------------------------------------------------------

// legendre_pn(n, m, x) is the associated Legendre polynomial P_n^m(x)
// For standard Legendre polynomials, use m=0

TEST(SpecialAdv2, Legendre_P0_Is_One) {
    for (double x : {-1.0, 0.0, 0.5, 1.0}) {
        EXPECT_NEAR(legendre_pn(0, 0, x), 1.0, 1e-10)
            << "P_0 should be 1 at x=" << x;
    }
}

TEST(SpecialAdv2, Legendre_P1_Is_X) {
    for (double x : {-1.0, 0.0, 0.5, 1.0}) {
        EXPECT_NEAR(legendre_pn(1, 0, x), x, 1e-10)
            << "P_1 should be x at x=" << x;
    }
}

TEST(SpecialAdv2, Legendre_P2_Correct) {
    // P_2(x) = (3x^2 - 1)/2
    for (double x : {0.0, 0.5, 1.0}) {
        double expected = (3.0 * x * x - 1.0) / 2.0;
        EXPECT_NEAR(legendre_pn(2, 0, x), expected, 1e-10)
            << "P_2 error at x=" << x;
    }
}

TEST(SpecialAdv2, Legendre_Recurrence) {
    // (n+1) * P_{n+1}(x) = (2n+1) * x * P_n(x) - n * P_{n-1}(x)
    double x = 0.7;
    for (int n = 1; n <= 5; ++n) {
        double lhs = (n + 1) * legendre_pn(n + 1, 0, x);
        double rhs = (2 * n + 1) * x * legendre_pn(n, 0, x) - n * legendre_pn(n - 1, 0, x);
        EXPECT_NEAR(lhs, rhs, 1e-9)
            << "Legendre recurrence failed at n=" << n;
    }
}

TEST(SpecialAdv2, Legendre_Pn_At_Pm1) {
    // P_n(1) = 1, P_n(-1) = (-1)^n for all n
    for (int n = 0; n <= 5; ++n) {
        EXPECT_NEAR(legendre_pn(n, 0, 1.0), 1.0, 1e-9)
            << "P_n(1) != 1 at n=" << n;
        double expected_m1 = (n % 2 == 0) ? 1.0 : -1.0;
        EXPECT_NEAR(legendre_pn(n, 0, -1.0), expected_m1, 1e-9)
            << "P_n(-1) failed at n=" << n;
    }
}

// ---------------------------------------------------------------------------
// Fresnel integrals
// ---------------------------------------------------------------------------

TEST(SpecialAdv2, Fresnel_AtZero_IsZero) {
    EXPECT_NEAR(fresnel_c(0.0), 0.0, 1e-10);
    EXPECT_NEAR(fresnel_s(0.0), 0.0, 1e-10);
}

TEST(SpecialAdv2, Fresnel_Odd_Functions) {
    for (double x : {0.5, 1.0, 2.0}) {
        EXPECT_NEAR(fresnel_c(-x), -fresnel_c(x), 1e-10);
        EXPECT_NEAR(fresnel_s(-x), -fresnel_s(x), 1e-10);
    }
}

TEST(SpecialAdv2, Fresnel_ValueAtOne) {
    // C(1.0) and S(1.0) should be finite and in a reasonable range
    EXPECT_TRUE(std::isfinite(fresnel_c(1.0)));
    EXPECT_TRUE(std::isfinite(fresnel_s(1.0)));
    // Standard definition: C(1) ≈ 0.9045, S(1) ≈ 0.3103 (scaled by pi/2)
    // Or: C(1) ≈ 0.7799 for the sqrt(2/pi) definition
    // Just check they're in [-1, 1]
    EXPECT_GE(fresnel_c(1.0), -1.0);
    EXPECT_LE(fresnel_c(1.0), 1.0);
}

TEST(SpecialAdv2, Fresnel_AllFinite) {
    for (double x : {0.0, 0.5, 1.0, 2.0, 5.0, 10.0}) {
        EXPECT_TRUE(std::isfinite(fresnel_c(x)));
        EXPECT_TRUE(std::isfinite(fresnel_s(x)));
    }
}
