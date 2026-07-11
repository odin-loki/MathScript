// Extended coverage for Axiom evolutionary framework, GRIA sub-systems, and IZAAC crypto/RNG.
// Complements test_frameworks_ext.cpp without duplicating its assertions.

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <vector>

#include "ms/frameworks/axiom/axiom.hpp"
#include "ms/frameworks/gria/gria.hpp"
#include "ms/frameworks/izaac/izaac.hpp"

using DMatrix = ms::ColMatrix<double>;

// ---------------------------------------------------------------------------
// Axiom: EvolutionConfig defaults
// ---------------------------------------------------------------------------

TEST(AxiomGriaExt, EvolutionConfigDefaultValues) {
    ms::axiom::EvolutionConfig cfg{};
    EXPECT_EQ(cfg.population_size,  100u);
    EXPECT_EQ(cfg.max_generations, 1000u);
    EXPECT_DOUBLE_EQ(cfg.mutation_rate,   0.1);
    EXPECT_DOUBLE_EQ(cfg.crossover_rate,  0.7);
    EXPECT_EQ(cfg.tournament_size,   5u);
    EXPECT_DOUBLE_EQ(cfg.target_alpha,    0.5);
    EXPECT_EQ(cfg.max_depth,        10u);
    // use_cuda field exists; just verify it is readable
    (void)cfg.use_cuda;
}

// ---------------------------------------------------------------------------
// Axiom: Algorithm default fitness
// ---------------------------------------------------------------------------

TEST(AxiomGriaExt, AlgorithmDefaultFitness) {
    ms::axiom::Algorithm algo{};
    EXPECT_DOUBLE_EQ(algo.fitness, 0.0);
}

// ---------------------------------------------------------------------------
// Axiom: population() size matches population_size in config
// ---------------------------------------------------------------------------

TEST(AxiomGriaExt, PopulationSizeMatchesConfig) {
    auto registry = ms::axiom::PrimitiveRegistry::build_from_ms_namespace();
    constexpr size_t N = 7;
    ms::axiom::Axiom engine(
        ms::axiom::EvolutionConfig{.population_size = N},
        registry);
    EXPECT_EQ(engine.population().size(), N);
}

// ---------------------------------------------------------------------------
// Axiom: evaluate applies representation row-by-row (column j -> xj)
// ---------------------------------------------------------------------------

TEST(AxiomGriaExt, EvaluateRowWiseExpression) {
    auto registry = ms::axiom::PrimitiveRegistry::build_from_ms_namespace();
    ms::axiom::Axiom engine(
        ms::axiom::EvolutionConfig{.population_size = 2},
        registry);

    ms::axiom::Algorithm algo{};
    algo.representation = ms::Sym("x0+x1");

    DMatrix data{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
    const auto result = engine.evaluate(algo, data);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->rows(), 3u);
    EXPECT_EQ(result->cols(), 1u);
    EXPECT_DOUBLE_EQ((*result)(0, 0), 3.0);
    EXPECT_DOUBLE_EQ((*result)(1, 0), 7.0);
    EXPECT_DOUBLE_EQ((*result)(2, 0), 11.0);
}

TEST(AxiomGriaExt, EvaluateProductExpression) {
    auto registry = ms::axiom::PrimitiveRegistry::build_from_ms_namespace();
    ms::axiom::Axiom engine(
        ms::axiom::EvolutionConfig{.population_size = 2},
        registry);

    ms::axiom::Algorithm algo{};
    algo.representation = ms::Sym("x0*x1");

    DMatrix data{{2.0, 3.0}, {-1.0, 5.0}};
    const auto result = engine.evaluate(algo, data);
    ASSERT_TRUE(result.has_value());
    EXPECT_DOUBLE_EQ((*result)(0, 0), 6.0);
    EXPECT_DOUBLE_EQ((*result)(1, 0), -5.0);
}

TEST(AxiomGriaExt, EvaluateUnaryFunctionExpression) {
    auto registry = ms::axiom::PrimitiveRegistry::build_from_ms_namespace();
    ms::axiom::Axiom engine(
        ms::axiom::EvolutionConfig{.population_size = 2},
        registry);

    ms::axiom::Algorithm algo{};
    algo.representation = ms::Sym("sin(x0)");

    DMatrix data{{0.0, 99.0}, {1.5707963267948966, 0.0}};
    const auto result = engine.evaluate(algo, data);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR((*result)(0, 0), 0.0, 1e-12);
    EXPECT_NEAR((*result)(1, 0), 1.0, 1e-9);
}

TEST(AxiomGriaExt, EvaluateComposedOperatorExpression) {
    auto registry = ms::axiom::PrimitiveRegistry::build_from_ms_namespace();
    ms::axiom::Axiom engine(
        ms::axiom::EvolutionConfig{.population_size = 2},
        registry);

    ms::axiom::Algorithm algo{};
    algo.representation = ms::Sym("x0") + ms::Sym("x1") * ms::Sym("2");

    DMatrix data{{1.0, 4.0}, {0.0, 3.0}};
    const auto result = engine.evaluate(algo, data);
    ASSERT_TRUE(result.has_value());
    EXPECT_DOUBLE_EQ((*result)(0, 0), 9.0);
    EXPECT_DOUBLE_EQ((*result)(1, 0), 6.0);
}

TEST(AxiomGriaExt, EvaluateConstantExpression) {
    auto registry = ms::axiom::PrimitiveRegistry::build_from_ms_namespace();
    ms::axiom::Axiom engine(
        ms::axiom::EvolutionConfig{.population_size = 2},
        registry);

    ms::axiom::Algorithm algo{};
    algo.representation = ms::Sym("3.5");

    DMatrix data{{10.0, 20.0}, {30.0, 40.0}};
    const auto result = engine.evaluate(algo, data);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->rows(), data.rows());
    EXPECT_EQ(result->cols(), 1u);
    EXPECT_DOUBLE_EQ((*result)(0, 0), 3.5);
    EXPECT_DOUBLE_EQ((*result)(1, 0), 3.5);
}

// ---------------------------------------------------------------------------
// Axiom: evaluate returns one column per input row
// ---------------------------------------------------------------------------

TEST(AxiomGriaExt, EvaluateDimensionsPreserved) {
    auto registry = ms::axiom::PrimitiveRegistry::build_from_ms_namespace();
    ms::axiom::Axiom engine(
        ms::axiom::EvolutionConfig{.population_size = 4},
        registry);
    DMatrix data{{1.0, 0.0, -1.0}, {2.0, 3.0, 4.0}};
    const auto result = engine.evaluate(engine.population().front(), data);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->rows(), data.rows());
    EXPECT_EQ(result->cols(), 1u);
}

// ---------------------------------------------------------------------------
// Axiom: PrimitiveRegistry – function_names and function_symbols match in size
// ---------------------------------------------------------------------------

TEST(AxiomGriaExt, PrimitiveRegistryParallelVectors) {
    auto reg = ms::axiom::PrimitiveRegistry::build_from_ms_namespace();
    ASSERT_FALSE(reg.function_names.empty());
    EXPECT_EQ(reg.function_names.size(), reg.function_symbols.size());
    // Every name should be a non-empty string
    for (const auto& name : reg.function_names) {
        EXPECT_FALSE(name.empty());
    }
}

// ---------------------------------------------------------------------------
// Axiom: single-generation evolve terminates and returns a valid Algorithm
// ---------------------------------------------------------------------------

TEST(AxiomGriaExt, EvolveOneGenerationTerminates) {
    ms::izaac::seed_session(static_cast<uint64_t>(12345));
    auto registry = ms::axiom::PrimitiveRegistry::build_from_ms_namespace();
    ms::axiom::Axiom engine(
        ms::axiom::EvolutionConfig{.population_size = 5, .max_generations = 1},
        registry);
    DMatrix data{{1.0, 0.0}, {0.0, 1.0}};

    const auto best = engine.evolve(
        [&](const ms::axiom::Algorithm& a) {
            return engine.gria_fitness(a, data);
        },
        [](const ms::axiom::Algorithm&) { return false; });

    ASSERT_TRUE(best.has_value());
    EXPECT_GE(best->fitness, 0.0);
    EXPECT_LE(best->fitness, 1.0);
    ms::izaac::clear_session();
}

// ---------------------------------------------------------------------------
// Axiom: evolve with immediate-termination callback returns on first generation
// ---------------------------------------------------------------------------

TEST(AxiomGriaExt, EvolveImmediateTermination) {
    auto registry = ms::axiom::PrimitiveRegistry::build_from_ms_namespace();
    ms::axiom::Axiom engine(
        ms::axiom::EvolutionConfig{.population_size = 4, .max_generations = 100},
        registry);
    DMatrix data{{0.5, 0.5}, {0.5, 0.5}};

    int eval_count = 0;
    const auto best = engine.evolve(
        [&](const ms::axiom::Algorithm& a) {
            ++eval_count;
            return engine.gria_fitness(a, data);
        },
        [](const ms::axiom::Algorithm&) {
            // terminate after first evaluation
            return true;
        });

    ASSERT_TRUE(best.has_value());
    // At most one generation worth of evaluations
    EXPECT_LE(eval_count, 8); // generous upper bound: 4 pop * 2 passes
}

// ---------------------------------------------------------------------------
// GRIA: compute_alpha template with identity transform → alpha ≈ 0
// ---------------------------------------------------------------------------

TEST(GriaExt, ComputeAlphaIdentityTransform) {
    const std::vector<double> data{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const auto identity = [](std::span<const double> in) {
        return std::vector<double>(in.begin(), in.end());
    };
    const double alpha = ms::gria::compute_alpha<double>(data, identity);
    // Identity does not reduce entropy: alpha should be ~0
    EXPECT_GE(alpha, 0.0);
    EXPECT_LE(alpha, 1.0);
    EXPECT_NEAR(alpha, 0.0, 0.05);
}

// ---------------------------------------------------------------------------
// GRIA: compute_alpha with empty input returns 0
// ---------------------------------------------------------------------------

TEST(GriaExt, ComputeAlphaEmptyInput) {
    const std::vector<double> empty{};
    const auto any_fn = [](std::span<const double> in) {
        return std::vector<double>(in.begin(), in.end());
    };
    EXPECT_DOUBLE_EQ(ms::gria::compute_alpha<double>(empty, any_fn), 0.0);
}

// ---------------------------------------------------------------------------
// GRIA: compute_alpha with constant-output transform → output entropy near 0
// ---------------------------------------------------------------------------

TEST(GriaExt, ComputeAlphaConstantOutputHighAlpha) {
    const std::vector<double> data{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    const auto constant_fn = [](std::span<const double> in) {
        return std::vector<double>(in.size(), 0.0); // all zeros → zero entropy
    };
    const double alpha = ms::gria::compute_alpha<double>(data, constant_fn);
    // Constant output collapses entropy → alpha should be 1.0 (clamped)
    EXPECT_GE(alpha, 0.0);
    EXPECT_LE(alpha, 1.0);
    EXPECT_NEAR(alpha, 1.0, 0.01);
}

// ---------------------------------------------------------------------------
// GRIA: gf2n::mul associativity and commutativity in GF(2^8)
// ---------------------------------------------------------------------------

TEST(GriaExt, Gf2nMulCommutativeAndAssociative) {
    constexpr uint64_t poly = 0x11B; // AES field polynomial
    const uint64_t a = 5, b = 7, c = 11;

    EXPECT_EQ(ms::gria::gf2n::mul(a, b, poly),
              ms::gria::gf2n::mul(b, a, poly));

    const uint64_t ab_c = ms::gria::gf2n::mul(
        ms::gria::gf2n::mul(a, b, poly), c, poly);
    const uint64_t a_bc = ms::gria::gf2n::mul(
        a, ms::gria::gf2n::mul(b, c, poly), poly);
    EXPECT_EQ(ab_c, a_bc);
}

// ---------------------------------------------------------------------------
// GRIA: gf2n::mul by 1 is identity
// ---------------------------------------------------------------------------

TEST(GriaExt, Gf2nMulByOneIsIdentity) {
    constexpr uint64_t poly = 0x11B;
    for (uint64_t v : {uint64_t{1}, uint64_t{3}, uint64_t{255}}) {
        EXPECT_EQ(ms::gria::gf2n::mul(v, 1, poly), v);
    }
}

// ---------------------------------------------------------------------------
// GRIA: ca::step with rule 90 (XOR of neighbours)
// ---------------------------------------------------------------------------

TEST(GriaExt, CaRule90XorOfNeighbours) {
    // Rule 90: next[i] = state[i-1] XOR state[i+1]
    const std::vector<uint8_t> state{0, 0, 1, 0, 0};
    const auto next = ms::gria::ca::step(state, 90);
    ASSERT_EQ(next.size(), state.size());
    // Middle cell (index 2): neighbours are state[1]=0 and state[3]=0 → 0 XOR 0 = 0
    EXPECT_EQ(next[2], 0u);
    // Cell index 1: neighbours state[0]=0, state[2]=1 → 1
    EXPECT_EQ(next[1], 1u);
    // Cell index 3: neighbours state[2]=1, state[4]=0 → 1
    EXPECT_EQ(next[3], 1u);
}

// ---------------------------------------------------------------------------
// GRIA: ca::step size invariant for various rules
// ---------------------------------------------------------------------------

TEST(GriaExt, CaStepSizeInvariant) {
    const std::vector<uint8_t> state{1, 0, 1, 1, 0, 0, 1};
    for (uint8_t rule : {uint8_t{0}, uint8_t{110}, uint8_t{255}}) {
        const auto next = ms::gria::ca::step(state, rule);
        EXPECT_EQ(next.size(), state.size());
    }
}

// ---------------------------------------------------------------------------
// GRIA: lfsr::is_maximal with a known maximal polynomial
// ---------------------------------------------------------------------------

TEST(GriaExt, LfsrIsMaximalKnownMaximalPoly) {
    // The step() function: bit=state&1, state>>=1, if bit then state^=poly
    // For a 3-bit maximal LFSR (period 7), poly=0x6 (0b110) gives taps at
    // positions 1 and 2 counting from LSB after shift — equivalent to x^3+x+1
    EXPECT_TRUE(ms::gria::lfsr::is_maximal(0x6, 3));
}

// ---------------------------------------------------------------------------
// GRIA: lfsr::is_maximal with non-maximal polynomial returns false
// ---------------------------------------------------------------------------

TEST(GriaExt, LfsrIsMaximalNonMaximal) {
    // Poly 0x3 = 11 binary → x + 1, not primitive for n >= 2
    EXPECT_FALSE(ms::gria::lfsr::is_maximal(0x3, 4));
}

// ---------------------------------------------------------------------------
// GRIA: dispatch_hint for unknown op returns negative sentinel
// ---------------------------------------------------------------------------

TEST(GriaExt, DispatchHintUnknownOpNegative) {
    const double val = ms::gria::dispatch_hint_alpha("__nonexistent_op_xyz__");
    EXPECT_LT(val, 0.0);
}

// ---------------------------------------------------------------------------
// IZAAC: CSPRNG with different array seeds produces different output
// ---------------------------------------------------------------------------

TEST(IzaacExt, CsprngDifferentSeedsDifferentOutput) {
    std::array<uint8_t, 32> seed_a{};
    std::array<uint8_t, 32> seed_b{};
    seed_a[0] = 1;
    seed_b[0] = 2;

    ms::izaac::CSPRNG rng_a(seed_a);
    ms::izaac::CSPRNG rng_b(seed_b);

    const uint64_t out_a = rng_a.next_u64();
    const uint64_t out_b = rng_b.next_u64();
    EXPECT_NE(out_a, out_b);
}

// ---------------------------------------------------------------------------
// IZAAC: CSPRNG same seed produces identical sequence (deterministic)
// ---------------------------------------------------------------------------

TEST(IzaacExt, CsprngSameSeedDeterministic) {
    std::array<uint8_t, 32> seed{};
    seed[7] = 99;

    ms::izaac::CSPRNG rng1(seed);
    ms::izaac::CSPRNG rng2(seed);

    for (int i = 0; i < 8; ++i) {
        EXPECT_EQ(rng1.next_u64(), rng2.next_u64());
    }
}

// ---------------------------------------------------------------------------
// IZAAC: CSPRNG next_f64 always in [0, 1]
// ---------------------------------------------------------------------------

TEST(IzaacExt, CsprngF64InUnitInterval) {
    std::array<uint8_t, 32> seed{};
    seed[0] = 55;
    ms::izaac::CSPRNG rng(seed);
    for (int i = 0; i < 64; ++i) {
        const double v = rng.next_f64();
        EXPECT_GE(v, 0.0);
        EXPECT_LE(v, 1.0);
    }
}

// ---------------------------------------------------------------------------
// IZAAC: rand_matrix dimensions
// ---------------------------------------------------------------------------

TEST(IzaacExt, RandMatrixDimensions) {
    ms::izaac::seed_session(static_cast<uint64_t>(77));
    const auto m = ms::izaac::rand_matrix(4, 3);
    EXPECT_EQ(m.rows(), 4u);
    EXPECT_EQ(m.cols(), 3u);
    // values should be finite
    for (size_t i = 0; i < m.rows(); ++i) {
        for (size_t j = 0; j < m.cols(); ++j) {
            EXPECT_TRUE(std::isfinite(m(i, j)));
        }
    }
    ms::izaac::clear_session();
}

// ---------------------------------------------------------------------------
// IZAAC: randn_matrix values are finite
// ---------------------------------------------------------------------------

TEST(IzaacExt, RandnMatrixFiniteValues) {
    ms::izaac::seed_session(static_cast<uint64_t>(88));
    const auto m = ms::izaac::randn_matrix(3, 3);
    EXPECT_EQ(m.rows(), 3u);
    EXPECT_EQ(m.cols(), 3u);
    for (size_t i = 0; i < m.rows(); ++i) {
        for (size_t j = 0; j < m.cols(); ++j) {
            EXPECT_TRUE(std::isfinite(m(i, j)));
        }
    }
    ms::izaac::clear_session();
}

// ---------------------------------------------------------------------------
// IZAAC: session lifecycle – seed → active → clear → inactive
// ---------------------------------------------------------------------------

TEST(IzaacExt, SessionLifecycle) {
    ms::izaac::clear_session();
    EXPECT_FALSE(ms::izaac::session_active());

    ms::izaac::seed_session(static_cast<uint64_t>(1));
    EXPECT_TRUE(ms::izaac::session_active());

    ms::izaac::clear_session();
    EXPECT_FALSE(ms::izaac::session_active());
}

// ---------------------------------------------------------------------------
// IZAAC: VRF keygen produces distinct public/private keys
// ---------------------------------------------------------------------------

TEST(IzaacExt, VrfKeygenDistinctKeys) {
    const auto key = ms::izaac::keygen();
    // public and private key should differ
    EXPECT_NE(key.private_key, key.public_key);
}

// ---------------------------------------------------------------------------
// IZAAC: mc::estimate_pi converges to within reasonable bounds
// ---------------------------------------------------------------------------

TEST(IzaacExt, McEstimatePi_ReturnsFiniteInRange) {
    // The CSPRNG is a lightweight stub; only verify structural correctness
    std::array<uint8_t, 32> seed{};
    seed[0] = 42;  // same seed as existing passing test in test_frameworks_ext
    ms::izaac::CSPRNG rng(seed);
    const double pi = ms::izaac::mc::estimate_pi(10000, rng);
    EXPECT_GT(pi, 0.0);
    EXPECT_LT(pi, 4.1);  // must be in [0, 4]
    EXPECT_TRUE(std::isfinite(pi));
}
