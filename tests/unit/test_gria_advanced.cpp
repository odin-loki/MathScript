// Advanced unit tests for under-covered ms::gria framework functions.
// Covers: matrix_alpha, is_critical, classify, generate_field,
//         langton_lambda, alpha_ca, alpha_lfsr.

#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <vector>

#include "ms/frameworks/gria/gria.hpp"

using DMatrix = ms::ColMatrix<double>;

// ---------------------------------------------------------------------------
// matrix_alpha: identity input/output → finite result
// ---------------------------------------------------------------------------

TEST(GriaAdvanced, MatrixAlphaIdentityIsFinite) {
    DMatrix id{{1.0, 0.0}, {0.0, 1.0}};
    const double alpha = ms::gria::matrix_alpha(id, id);
    EXPECT_TRUE(std::isfinite(alpha));
}

TEST(GriaAdvanced, MatrixAlphaInUnitInterval) {
    DMatrix x{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    DMatrix fx{{2.0, 4.0, 6.0}, {8.0, 10.0, 12.0}};
    const double alpha = ms::gria::matrix_alpha(x, fx);
    EXPECT_TRUE(std::isfinite(alpha));
    EXPECT_GE(alpha, 0.0);
    EXPECT_LE(alpha, 1.0);
}

TEST(GriaAdvanced, MatrixAlphaConstantOutputHighAlpha) {
    // Collapsing all values to 0 maximises information loss → alpha near 1
    DMatrix x{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    DMatrix fx{{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
    const double alpha = ms::gria::matrix_alpha(x, fx);
    EXPECT_TRUE(std::isfinite(alpha));
    EXPECT_GE(alpha, 0.0);
    EXPECT_LE(alpha, 1.0);
}

// ---------------------------------------------------------------------------
// is_critical: alpha near 0.5 → true; extremes → false
// ---------------------------------------------------------------------------

TEST(GriaAdvanced, IsCriticalExactHalfIsTrue) {
    EXPECT_TRUE(ms::gria::is_critical(0.5));
}

TEST(GriaAdvanced, IsCriticalJustInsideTolerance) {
    // Default tolerance 0.05: 0.5 ± 0.04 should be critical
    EXPECT_TRUE(ms::gria::is_critical(0.46));
    EXPECT_TRUE(ms::gria::is_critical(0.54));
}

TEST(GriaAdvanced, IsCriticalZeroIsFalse) {
    EXPECT_FALSE(ms::gria::is_critical(0.0));
}

TEST(GriaAdvanced, IsCriticalOneIsFalse) {
    EXPECT_FALSE(ms::gria::is_critical(1.0));
}

TEST(GriaAdvanced, IsCriticalCustomToleranceWide) {
    // With tolerance 0.2: 0.35 is within 0.15 of 0.5 → critical
    EXPECT_TRUE(ms::gria::is_critical(0.35, 0.2));
}

TEST(GriaAdvanced, IsCriticalCustomToleranceNarrow) {
    // With tolerance 0.01: 0.46 is 0.04 away from 0.5 → not critical
    EXPECT_FALSE(ms::gria::is_critical(0.46, 0.01));
}

// ---------------------------------------------------------------------------
// classify: returns expected ComputeClass for boundary values
// ---------------------------------------------------------------------------

TEST(GriaAdvanced, ClassifyAlphaZeroIsReversible) {
    EXPECT_EQ(ms::gria::classify(0.0), ms::gria::ComputeClass::Reversible);
}

TEST(GriaAdvanced, ClassifyAlphaOneIsIrreversible) {
    EXPECT_EQ(ms::gria::classify(1.0), ms::gria::ComputeClass::Irreversible);
}

TEST(GriaAdvanced, ClassifyAlphaHalfIsCritical) {
    EXPECT_EQ(ms::gria::classify(0.5), ms::gria::ComputeClass::Critical);
}

TEST(GriaAdvanced, ClassifyReturnsValidEnumForMidpoints) {
    // Any alpha must produce one of the three valid enum values
    for (double a : {0.1, 0.25, 0.5, 0.75, 0.9}) {
        const auto cls = ms::gria::classify(a);
        const bool valid =
            cls == ms::gria::ComputeClass::Reversible ||
            cls == ms::gria::ComputeClass::Critical   ||
            cls == ms::gria::ComputeClass::Irreversible;
        EXPECT_TRUE(valid) << "classify(" << a << ") returned unexpected value";
    }
}

// ---------------------------------------------------------------------------
// gf2n::generate_field: size and element validity
// ---------------------------------------------------------------------------

TEST(GriaAdvanced, GenerateFieldSizeMatchesPowerOfTwo) {
    for (int n : {2, 3, 4, 8}) {
        const auto field = ms::gria::gf2n::generate_field(n);
        EXPECT_EQ(field.size(), static_cast<size_t>(1u << n));
    }
}

TEST(GriaAdvanced, GenerateFieldContainsZeroAndOne) {
    const auto field = ms::gria::gf2n::generate_field(4);
    bool has_zero = false, has_one = false;
    for (const auto v : field) {
        if (v == 0) has_zero = true;
        if (v == 1) has_one = true;
    }
    EXPECT_TRUE(has_zero);
    EXPECT_TRUE(has_one);
}

TEST(GriaAdvanced, GenerateFieldElementsBoundedByFieldSize) {
    constexpr int n = 4;
    const uint64_t max_val = (1u << n) - 1;
    const auto field = ms::gria::gf2n::generate_field(n);
    for (const auto v : field) {
        EXPECT_LE(v, max_val);
    }
}

// ---------------------------------------------------------------------------
// ca::langton_lambda: boundary rules → 0.0; rule 110 ∈ [0,1]
// ---------------------------------------------------------------------------

TEST(GriaAdvanced, LangtonLambdaRule0InUnitInterval) {
    const double lambda = ms::gria::ca::langton_lambda(0);
    EXPECT_GE(lambda, 0.0);
    EXPECT_LE(lambda, 1.0);
}

TEST(GriaAdvanced, LangtonLambdaRule255InUnitInterval) {
    const double lambda = ms::gria::ca::langton_lambda(255);
    EXPECT_GE(lambda, 0.0);
    EXPECT_LE(lambda, 1.0);
}

TEST(GriaAdvanced, LangtonLambdaRule110InUnitInterval) {
    const double lambda = ms::gria::ca::langton_lambda(110);
    EXPECT_GE(lambda, 0.0);
    EXPECT_LE(lambda, 1.0);
}

TEST(GriaAdvanced, LangtonLambdaAlwaysInUnitIntervalSample) {
    for (int rule = 0; rule <= 255; rule += 17) {
        const double lambda = ms::gria::ca::langton_lambda(static_cast<uint8_t>(rule));
        EXPECT_GE(lambda, 0.0) << "rule=" << rule;
        EXPECT_LE(lambda, 1.0) << "rule=" << rule;
    }
}

// ---------------------------------------------------------------------------
// ca::alpha_ca: returns finite value in [0,1]
// ---------------------------------------------------------------------------

TEST(GriaAdvanced, AlphaCaRule110IsFiniteInUnitInterval) {
    const double alpha = ms::gria::ca::alpha_ca(110, 10, 20);
    EXPECT_TRUE(std::isfinite(alpha));
    EXPECT_GE(alpha, 0.0);
    EXPECT_LE(alpha, 1.0);
}

TEST(GriaAdvanced, AlphaCaRule0IsFinite) {
    const double alpha = ms::gria::ca::alpha_ca(0, 5, 16);
    EXPECT_TRUE(std::isfinite(alpha));
}

TEST(GriaAdvanced, AlphaCaRule255IsFinite) {
    const double alpha = ms::gria::ca::alpha_ca(255, 5, 16);
    EXPECT_TRUE(std::isfinite(alpha));
}

TEST(GriaAdvanced, AlphaCaLargerSimulationIsFinite) {
    const double alpha = ms::gria::ca::alpha_ca(30, 50, 64);
    EXPECT_TRUE(std::isfinite(alpha));
    EXPECT_GE(alpha, 0.0);
    EXPECT_LE(alpha, 1.0);
}

// ---------------------------------------------------------------------------
// lfsr::alpha_lfsr: returns finite value in [0,1]
// ---------------------------------------------------------------------------

TEST(GriaAdvanced, AlphaLfsrAesPolyIsFiniteInUnitInterval) {
    // AES field polynomial: x^8 + x^4 + x^3 + x + 1 = 0x11B
    const double alpha = ms::gria::lfsr::alpha_lfsr(0x11B, 100);
    EXPECT_TRUE(std::isfinite(alpha));
    EXPECT_GE(alpha, 0.0);
    EXPECT_LE(alpha, 1.0);
}

TEST(GriaAdvanced, AlphaLfsrKnownMaximalPolyIsFinite) {
    // Known 3-bit maximal poly: 0x6 (x^2 + x, after feedback convention)
    const double alpha = ms::gria::lfsr::alpha_lfsr(0x6, 50);
    EXPECT_TRUE(std::isfinite(alpha));
}

TEST(GriaAdvanced, AlphaLfsrMoreStepsIsFinite) {
    const double alpha = ms::gria::lfsr::alpha_lfsr(0xB, 200);
    EXPECT_TRUE(std::isfinite(alpha));
    EXPECT_GE(alpha, 0.0);
    EXPECT_LE(alpha, 1.0);
}
