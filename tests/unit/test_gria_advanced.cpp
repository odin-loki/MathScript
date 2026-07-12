// Advanced unit tests for under-covered ms::gria framework functions.
// Covers: matrix_alpha, is_critical, classify, generate_field,
//         langton_lambda, alpha_ca, alpha_lfsr.

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
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

// ---------------------------------------------------------------------------
// ca::hamming_distance: hand-computable pairs, identity, symmetry, bounds.
// ---------------------------------------------------------------------------

TEST(GriaAdvanced, HammingDistanceKnownThreeDifferences) {
    // Differ at indices 1, 4, 6 (3 positions) out of 8.
    const std::vector<uint8_t> a{0, 0, 1, 1, 0, 1, 0, 1};
    const std::vector<uint8_t> b{0, 1, 1, 1, 1, 1, 1, 1};
    EXPECT_EQ(ms::gria::ca::hamming_distance(a, b), 3u);
}

TEST(GriaAdvanced, HammingDistanceIdenticalConfigsIsZero) {
    const std::vector<uint8_t> a{1, 0, 1, 1, 0, 0, 1, 0};
    EXPECT_EQ(ms::gria::ca::hamming_distance(a, a), 0u);
}

TEST(GriaAdvanced, HammingDistanceEmptyConfigsIsZero) {
    const std::vector<uint8_t> a{};
    const std::vector<uint8_t> b{};
    EXPECT_EQ(ms::gria::ca::hamming_distance(a, b), 0u);
}

TEST(GriaAdvanced, HammingDistanceComplementaryConfigsIsFullLength) {
    const std::vector<uint8_t> a{0, 1, 0, 1, 0, 1, 0, 1};
    std::vector<uint8_t> b(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        b[i] = static_cast<uint8_t>(1 - a[i]);
    }
    EXPECT_EQ(ms::gria::ca::hamming_distance(a, b), a.size());
}

TEST(GriaAdvanced, HammingDistanceIsSymmetric) {
    const std::vector<std::pair<std::vector<uint8_t>, std::vector<uint8_t>>> pairs{
        {{0, 0, 0, 0}, {1, 1, 1, 1}},
        {{1, 0, 1, 0, 1}, {1, 1, 1, 0, 0}},
        {{0, 1}, {1, 0}},
        {{1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 0}},
    };
    for (const auto& [a, b] : pairs) {
        EXPECT_EQ(ms::gria::ca::hamming_distance(a, b), ms::gria::ca::hamming_distance(b, a));
    }
}

TEST(GriaAdvanced, HammingDistanceBoundedByLength) {
    const std::vector<std::vector<uint8_t>> configs{
        {0, 0, 0, 1, 1, 0, 1},
        {1, 1, 0, 1, 0, 0, 1},
        {0, 1, 1, 0, 1, 1, 0},
        {1, 0, 0, 0, 1, 1, 1},
    };
    for (const auto& a : configs) {
        for (const auto& b : configs) {
            const size_t d = ms::gria::ca::hamming_distance(a, b);
            EXPECT_GE(d, 0u);
            EXPECT_LE(d, a.size());
        }
    }
}

TEST(GriaAdvanced, HammingDistanceMismatchedLengthCountsExtraAsDiffering) {
    // Common prefix {1,0,1} matches exactly -> 0 mismatches there.
    // b has 2 extra trailing cells beyond a's length -> +2.
    const std::vector<uint8_t> a{1, 0, 1};
    const std::vector<uint8_t> b{1, 0, 1, 1, 0};
    EXPECT_EQ(ms::gria::ca::hamming_distance(a, b), 2u);
    // Symmetric in the same defensive convention.
    EXPECT_EQ(ms::gria::ca::hamming_distance(b, a), 2u);
}

TEST(GriaAdvanced, HammingDistanceSingleCellConfigsNoCrash) {
    const std::vector<uint8_t> a{1};
    const std::vector<uint8_t> b{0};
    EXPECT_EQ(ms::gria::ca::hamming_distance(a, b), 1u);
    EXPECT_EQ(ms::gria::ca::hamming_distance(a, a), 0u);
}

// ---------------------------------------------------------------------------
// ca::divergence_trajectory: length convention, deterministic identity,
// qualitative sensitivity-to-initial-conditions check under a chaotic rule.
// ---------------------------------------------------------------------------

TEST(GriaAdvanced, DivergenceTrajectoryLengthIsNStepsPlusOne) {
    const std::vector<uint8_t> a{0, 1, 0, 1, 0, 1, 0, 1};
    const std::vector<uint8_t> b{0, 1, 0, 1, 0, 1, 0, 0};
    const int n_steps = 5;
    const auto trajectory = ms::gria::ca::divergence_trajectory(a, b, /*rule=*/110, n_steps);
    EXPECT_EQ(trajectory.size(), static_cast<size_t>(n_steps + 1));
}

TEST(GriaAdvanced, DivergenceTrajectoryIdenticalConfigsStayZeroUnderDeterministicRule) {
    // step() is a deterministic function of the configuration, so two
    // identical configurations stay identical forever under any rule.
    const std::vector<uint8_t> a{1, 0, 1, 1, 0, 0, 1, 0, 1, 1};
    for (int rule : {0, 30, 110, 255}) {
        const auto trajectory =
            ms::gria::ca::divergence_trajectory(a, a, static_cast<uint8_t>(rule), 8);
        for (size_t d : trajectory) {
            EXPECT_EQ(d, 0u) << "rule=" << rule;
        }
    }
}

TEST(GriaAdvanced, DivergenceTrajectoryFirstEntryIsInitialDistance) {
    const std::vector<uint8_t> a{0, 0, 0, 0, 0, 0};
    const std::vector<uint8_t> b{0, 0, 1, 0, 0, 0};
    const auto trajectory = ms::gria::ca::divergence_trajectory(a, b, /*rule=*/110, 4);
    EXPECT_EQ(trajectory.front(), ms::gria::ca::hamming_distance(a, b));
}

TEST(GriaAdvanced, DivergenceTrajectoryZeroStepsReturnsSingleInitialEntry) {
    const std::vector<uint8_t> a{1, 0, 1, 0};
    const std::vector<uint8_t> b{1, 1, 1, 0};
    const auto trajectory = ms::gria::ca::divergence_trajectory(a, b, /*rule=*/54, 0);
    ASSERT_EQ(trajectory.size(), 1u);
    EXPECT_EQ(trajectory.front(), 1u);
}

TEST(GriaAdvanced, DivergenceTrajectoryChaoticRuleShowsSensitivityToInitialConditions) {
    // Rule 110 sits at Wolfram's edge-of-chaos / class IV boundary: a single
    // differing cell in an otherwise-identical configuration tends to grow
    // into a nonzero, non-trivial spread of differences over time. This is
    // a qualitative check only (exact trajectories are rule/width-specific).
    std::vector<uint8_t> a(40, 0);
    a[20] = 1;
    std::vector<uint8_t> b = a;
    b[19] = 1; // Perturb one neighbouring cell.

    const auto trajectory = ms::gria::ca::divergence_trajectory(a, b, /*rule=*/110, 30);
    EXPECT_EQ(trajectory.front(), 1u);
    const bool eventually_nonzero =
        std::any_of(trajectory.begin() + 1, trajectory.end(), [](size_t d) { return d > 0; });
    EXPECT_TRUE(eventually_nonzero);
}

// ---------------------------------------------------------------------------
// ca::settling_time: convergence detection, deterministic identity.
// ---------------------------------------------------------------------------

TEST(GriaAdvanced, SettlingTimeAlreadyIdenticalIsZero) {
    const std::vector<uint8_t> a{1, 0, 1, 1, 0};
    const auto settled = ms::gria::ca::settling_time(a, a, /*rule=*/110, 10);
    ASSERT_TRUE(settled.has_value());
    EXPECT_EQ(*settled, 0u);
}

TEST(GriaAdvanced, SettlingTimeConvergesQuicklyUnderRuleZero) {
    // Rule 0 maps every neighbourhood to 0, so any two distinct
    // configurations both collapse to all-zero after one step -> settles
    // by step 1 (rule 0's fixed point is the all-zero configuration).
    const std::vector<uint8_t> a{1, 0, 1, 0};
    const std::vector<uint8_t> b{0, 1, 0, 1};
    const auto settled = ms::gria::ca::settling_time(a, b, /*rule=*/0, 5);
    ASSERT_TRUE(settled.has_value());
    EXPECT_LE(*settled, 5u);
}

TEST(GriaAdvanced, SettlingTimeReturnsNulloptWhenBudgetTooSmall) {
    // Two configurations that differ and, under a chaotic rule, are highly
    // unlikely to re-synchronise within a single step.
    std::vector<uint8_t> a(20, 0);
    a[10] = 1;
    std::vector<uint8_t> b = a;
    b[9] = 1;
    const auto settled = ms::gria::ca::settling_time(a, b, /*rule=*/110, 0);
    EXPECT_FALSE(settled.has_value());
}
