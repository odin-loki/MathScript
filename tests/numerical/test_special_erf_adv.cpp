// MathScript Advanced Special Functions: erf, erfc, erfi, Fresnel, Dawson (Wave 55)

#define _USE_MATH_DEFINES
#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#include "ms/special/special.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Use ms:: prefix for all special function calls to avoid ambiguity with <cmath>

// ---------------------------------------------------------------------------
// erf: error function
// ---------------------------------------------------------------------------

TEST(SpecialErfAdv, Erf_AtZero_IsZero) {
    EXPECT_NEAR(ms::erf(0.0), 0.0, 1e-10);
}

TEST(SpecialErfAdv, Erf_AtOne_KnownValue) {
    // ms::erf(1) ≈ 0.8427007929...
    EXPECT_NEAR(ms::erf(1.0), 0.8427007929, 1e-6);
}

TEST(SpecialErfAdv, Erf_OddFunction) {
    // ms::erf(-x) = -ms::erf(x)
    for (double x : {0.5, 1.0, 1.5, 2.0}) {
        EXPECT_NEAR(ms::erf(-x), -ms::erf(x), 1e-10) << "erf not odd at x=" << x;
    }
}

TEST(SpecialErfAdv, Erf_Monotone_Increasing) {
    double prev = ms::erf(-2.0);
    for (double x : {-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0}) {
        double curr = ms::erf(x);
        EXPECT_GE(curr, prev - 1e-10);
        prev = curr;
    }
}

TEST(SpecialErfAdv, Erf_LargeArg_NearOne) {
    EXPECT_NEAR(ms::erf(5.0), 1.0, 1e-11);
}

TEST(SpecialErfAdv, Erf_AllFinite) {
    for (double x : {-3.0, -1.0, 0.0, 1.0, 3.0}) {
        EXPECT_TRUE(std::isfinite(ms::erf(x)));
    }
}

// ---------------------------------------------------------------------------
// erfc: complementary error function
// ---------------------------------------------------------------------------

TEST(SpecialErfAdv, Erfc_PlusErf_IsTwo_At_NegX) {
    // ms::erfc(-x) + ms::erf(-x) = 1 → ms::erfc(x) = 1 - ms::erf(x)
    for (double x : {0.0, 0.5, 1.0, 2.0}) {
        EXPECT_NEAR(ms::erfc(x) + ms::erf(x), 1.0, 1e-10) << "erf + erfc != 1 at x=" << x;
    }
}

TEST(SpecialErfAdv, Erfc_AtZero_IsOne) {
    EXPECT_NEAR(ms::erfc(0.0), 1.0, 1e-10);
}

TEST(SpecialErfAdv, Erfc_LargeArg_NearZero) {
    EXPECT_NEAR(ms::erfc(5.0), 0.0, 1e-11);
}

TEST(SpecialErfAdv, Erfc_Monotone_Decreasing) {
    double prev = ms::erfc(-2.0);
    for (double x : {-1.5, -1.0, 0.0, 0.5, 1.0, 2.0}) {
        double curr = ms::erfc(x);
        EXPECT_LE(curr, prev + 1e-10);
        prev = curr;
    }
}

TEST(SpecialErfAdv, Erfc_AllFinite) {
    for (double x : {0.0, 1.0, 2.0, 5.0}) EXPECT_TRUE(std::isfinite(ms::erfc(x)));
}

// ---------------------------------------------------------------------------
// erfi: imaginary error function
// ---------------------------------------------------------------------------

TEST(SpecialErfAdv, Erfi_AtZero_IsZero) {
    EXPECT_NEAR(ms::erfi(0.0), 0.0, 1e-10);
}

TEST(SpecialErfAdv, Erfi_OddFunction) {
    for (double x : {0.3, 0.7, 1.0}) {
        EXPECT_NEAR(ms::erfi(-x), -ms::erfi(x), 1e-10);
    }
}

TEST(SpecialErfAdv, Erfi_AllFinite) {
    for (double x : {0.0, 0.5, 1.0, 1.5}) {
        EXPECT_TRUE(std::isfinite(ms::erfi(x)));
    }
}

TEST(SpecialErfAdv, Erfi_Monotone_Increasing) {
    double prev = ms::erfi(0.0);
    for (double x : {0.25, 0.5, 0.75, 1.0, 1.25}) {
        double curr = ms::erfi(x);
        EXPECT_GE(curr, prev - 1e-10);
        prev = curr;
    }
}

// ---------------------------------------------------------------------------
// Dawson function
// ---------------------------------------------------------------------------

TEST(SpecialErfAdv, Dawson_AtZero_IsZero) {
    EXPECT_NEAR(ms::dawson(0.0), 0.0, 1e-10);
}

TEST(SpecialErfAdv, Dawson_AllFinite) {
    for (double x : {0.0, 0.5, 1.0, 2.0, 5.0}) {
        EXPECT_TRUE(std::isfinite(ms::dawson(x)));
    }
}

TEST(SpecialErfAdv, Dawson_MaxNearOne) {
    // Dawson function has maximum near x=0.92; peak value ≈ 0.5
    double peak = 0.0;
    for (double x = 0.0; x <= 2.0; x += 0.01) {
        peak = std::max(peak, ms::dawson(x));
    }
    EXPECT_GT(peak, 0.3) << "Dawson should peak around 0.5";
    EXPECT_LT(peak, 0.7) << "Dawson peak should be below 0.7";
}

// ---------------------------------------------------------------------------
// Fresnel integrals
// ---------------------------------------------------------------------------

TEST(SpecialErfAdv, FresnelC_AtZero_IsZero) {
    EXPECT_NEAR(ms::fresnel_c(0.0), 0.0, 1e-10);
}

TEST(SpecialErfAdv, FresnelS_AtZero_IsZero) {
    EXPECT_NEAR(ms::fresnel_s(0.0), 0.0, 1e-10);
}

TEST(SpecialErfAdv, FresnelC_AllFinite) {
    for (double x : {0.0, 0.5, 1.0, 2.0, 5.0}) {
        EXPECT_TRUE(std::isfinite(ms::fresnel_c(x)));
    }
}

TEST(SpecialErfAdv, FresnelS_AllFinite) {
    for (double x : {0.0, 0.5, 1.0, 2.0, 5.0}) {
        EXPECT_TRUE(std::isfinite(ms::fresnel_s(x)));
    }
}

TEST(SpecialErfAdv, FresnelC_Oscillates_LargeX) {
    // For large x, Fresnel C approaches 0.5; check it's bounded
    double fc5 = ms::fresnel_c(5.0);
    EXPECT_GT(fc5, -1.0);
    EXPECT_LT(fc5, 1.0);
}

TEST(SpecialErfAdv, FresnelS_Oscillates_LargeX) {
    double fs5 = ms::fresnel_s(5.0);
    EXPECT_GT(fs5, -1.0);
    EXPECT_LT(fs5, 1.0);
}

TEST(SpecialErfAdv, Fresnel_OddFunctions) {
    // C(-x) = -C(x), S(-x) = -S(x)
    for (double x : {0.5, 1.0, 2.0}) {
        EXPECT_NEAR(ms::fresnel_c(-x), -ms::fresnel_c(x), 1e-8);
        EXPECT_NEAR(ms::fresnel_s(-x), -ms::fresnel_s(x), 1e-8);
    }
}
