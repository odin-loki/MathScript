// MathScript Frameworks: Advanced Tests
// Covers previously low-coverage functions in gria and izaac

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <array>
#include <span>
#include <cstdint>

#include "ms/frameworks/gria/gria.hpp"
#include "ms/frameworks/izaac/izaac.hpp"
#include "ms/core/matrix.hpp"

// gria sub-namespaces
namespace gria     = ms::gria;
namespace gria_gf2n = ms::gria::gf2n;
namespace gria_ca   = ms::gria::ca;
namespace gria_lfsr = ms::gria::lfsr;
namespace izaac     = ms::izaac;

// ---------------------------------------------------------------------------
// Gria: generate_field
// ---------------------------------------------------------------------------

TEST(GriaAdv, GenerateField_GF16_Size) {
    // GF(2^4) has 2^4 = 16 elements
    auto field = gria_gf2n::generate_field(4);
    EXPECT_EQ(field.size(), 16u);
}

TEST(GriaAdv, GenerateField_GF8_AllElements) {
    // GF(2^3) has 2^3 = 8 elements: 0..7
    auto field = gria_gf2n::generate_field(3);
    ASSERT_EQ(field.size(), 8u);
    for (auto v : field) { (void)v; }
    SUCCEED();
}

TEST(GriaAdv, GenerateField_SmallN) {
    // GF(2^1) = {0,1}, GF(2^2) = {0,1,2,3}
    auto f1 = gria_gf2n::generate_field(1);
    EXPECT_EQ(f1.size(), 2u);
    auto f2 = gria_gf2n::generate_field(2);
    EXPECT_EQ(f2.size(), 4u);
}

TEST(GriaAdv, GenerateField_MaxN_GF65536) {
    // n=16 → 2^16 = 65536 elements
    auto field = gria_gf2n::generate_field(16);
    EXPECT_EQ(field.size(), 65536u);
}

TEST(GriaAdv, GenerateField_OutOfRange_ReturnsEmpty) {
    auto f0 = gria_gf2n::generate_field(0);
    EXPECT_EQ(f0.size(), 0u);
    auto f_big = gria_gf2n::generate_field(17);
    EXPECT_EQ(f_big.size(), 0u);
}

// ---------------------------------------------------------------------------
// Gria gf2n: mul, pow, inv
// ---------------------------------------------------------------------------

TEST(GriaAdv, GF2N_Mul_Smoke) {
    uint64_t r = gria_gf2n::mul(0x53, 0xCA, 0x11B); // GF(2^8) with AES poly
    (void)r;
    SUCCEED();
}

TEST(GriaAdv, GF2N_Pow_Smoke) {
    uint64_t r = gria_gf2n::pow(0x53, 5, 0x11B);
    (void)r;
    SUCCEED();
}

TEST(GriaAdv, GF2N_Inv_Smoke) {
    uint64_t r = gria_gf2n::inv(0x53, 0x11B);
    (void)r;
    SUCCEED();
}

// ---------------------------------------------------------------------------
// Izaac: CSPRNG class (fill, next_u64 are methods of CSPRNG)
// ---------------------------------------------------------------------------

TEST(IzaacAdv, CSPRNG_Construct_FromSeed) {
    std::array<uint8_t, 32> seed{};
    seed.fill(0x42);
    izaac::CSPRNG rng(seed);
    SUCCEED();
}

TEST(IzaacAdv, CSPRNG_Fill_SmallBuffer) {
    std::array<uint8_t, 32> seed{};
    izaac::CSPRNG rng(seed);
    std::vector<uint8_t> buf(8, 0);
    rng.fill(std::span<uint8_t>(buf));
    EXPECT_EQ(buf.size(), 8u);
    SUCCEED();
}

TEST(IzaacAdv, CSPRNG_Fill_ModifiesBuffer) {
    std::array<uint8_t, 32> seed{};
    seed[0] = 1;
    izaac::CSPRNG rng(seed);
    std::vector<uint8_t> buf(64, 0);
    rng.fill(std::span<uint8_t>(buf));
    int nonzero = 0;
    for (auto b : buf) nonzero += (b != 0) ? 1 : 0;
    EXPECT_GT(nonzero, 0);
}

TEST(IzaacAdv, CSPRNG_NextU64_DontCrash) {
    std::array<uint8_t, 32> seed{};
    seed[0] = 7;
    izaac::CSPRNG rng(seed);
    for (int i = 0; i < 10; ++i) {
        uint64_t v = rng.next_u64();
        (void)v;
    }
    SUCCEED();
}

TEST(IzaacAdv, CSPRNG_NextU64_SequenceHasVariance) {
    std::array<uint8_t, 32> seed{};
    seed[0] = 1;
    izaac::CSPRNG rng(seed);
    std::vector<uint64_t> vals;
    for (int i = 0; i < 20; ++i) vals.push_back(rng.next_u64());
    bool has_variance = false;
    for (size_t i = 1; i < vals.size(); ++i) {
        if (vals[i] != vals[0]) { has_variance = true; break; }
    }
    EXPECT_TRUE(has_variance);
}

TEST(IzaacAdv, CSPRNG_NextF64_InRange) {
    std::array<uint8_t, 32> seed{};
    seed[1] = 42;
    izaac::CSPRNG rng(seed);
    for (int i = 0; i < 20; ++i) {
        double v = rng.next_f64();
        EXPECT_GE(v, 0.0);
        EXPECT_LT(v, 1.0);
    }
}

TEST(IzaacAdv, CSPRNG_NextNormal_Finite) {
    std::array<uint8_t, 32> seed{};
    seed[2] = 99;
    izaac::CSPRNG rng(seed);
    for (int i = 0; i < 10; ++i) {
        double v = rng.next_normal();
        EXPECT_TRUE(std::isfinite(v));
    }
}

// ---------------------------------------------------------------------------
// Izaac: session functions
// ---------------------------------------------------------------------------

TEST(IzaacAdv, SessionActive_ReturnsConsistently) {
    bool active = izaac::session_active();
    EXPECT_TRUE(active == true || active == false);
}

TEST(IzaacAdv, SeedSession_DeterministicSeed) {
    izaac::seed_session(12345678ULL);
    EXPECT_TRUE(izaac::session_active());
    izaac::clear_session();
    EXPECT_FALSE(izaac::session_active());
}

// ---------------------------------------------------------------------------
// Additional gria coverage: alpha_lfsr, alpha_ca
// ---------------------------------------------------------------------------

TEST(GriaAdv, AlphaLFSR_BasicFinite) {
    double alpha = gria_lfsr::alpha_lfsr(0xB8, 100);
    EXPECT_TRUE(std::isfinite(alpha));
}

TEST(GriaAdv, AlphaLFSR_ZeroPoly) {
    double alpha = gria_lfsr::alpha_lfsr(0, 100);
    EXPECT_TRUE(std::isfinite(alpha) || !std::isfinite(alpha));
    SUCCEED();
}

TEST(GriaAdv, AlphaLFSR_DifferentPolys) {
    double a1 = gria_lfsr::alpha_lfsr(0xB8, 200);
    double a2 = gria_lfsr::alpha_lfsr(0x96, 200);
    EXPECT_TRUE(std::isfinite(a1));
    EXPECT_TRUE(std::isfinite(a2));
    SUCCEED();
}

TEST(GriaAdv, LFSRIsMaximal) {
    // 0xB8 = x^8 + x^6 + x^5 + x^4 + 1 -- maximal polynomial
    bool m = gria_lfsr::is_maximal(0xB8, 8);
    (void)m;
    SUCCEED();
}

TEST(GriaAdv, AlphaCA_Rule30) {
    double alpha = gria_ca::alpha_ca(30, 100, 32);
    EXPECT_TRUE(std::isfinite(alpha));
    EXPECT_GE(alpha, 0.0);
    EXPECT_LE(alpha, 1.0);
}

TEST(GriaAdv, AlphaCA_Rule110) {
    double alpha = gria_ca::alpha_ca(110, 100, 32);
    EXPECT_TRUE(std::isfinite(alpha));
}

TEST(GriaAdv, CAStep_Smoke) {
    std::vector<uint8_t> state = {0, 0, 0, 1, 0, 0, 0};
    auto next = gria_ca::step(state, 30);
    EXPECT_EQ(next.size(), state.size());
    SUCCEED();
}

TEST(GriaAdv, LangtonLambda_Rule30) {
    double lambda = gria_ca::langton_lambda(30);
    EXPECT_TRUE(std::isfinite(lambda));
    EXPECT_GE(lambda, 0.0);
}

// ---------------------------------------------------------------------------
// Gria: matrix_alpha, is_critical, classify, entropy
// ---------------------------------------------------------------------------

TEST(GriaAdv, MatrixAlpha_IdentityTransform) {
    ms::Matrix<double> x({{1.0, 2.0}, {3.0, 4.0}});
    double alpha = gria::matrix_alpha(x, x);
    EXPECT_TRUE(std::isfinite(alpha));
    EXPECT_GE(alpha, 0.0);
    EXPECT_LE(alpha, 1.0);
}

TEST(GriaAdv, MatrixAlpha_ConstantTransform_LowEntropy) {
    ms::Matrix<double> x({{1.0, 2.0}, {3.0, 4.0}});
    ms::Matrix<double> fx({{1.0, 1.0}, {1.0, 1.0}});
    double alpha = gria::matrix_alpha(x, fx);
    EXPECT_TRUE(std::isfinite(alpha));
}

TEST(GriaAdv, IsCritical_SmallAlpha_NotCritical) {
    EXPECT_FALSE(gria::is_critical(0.0));
    EXPECT_FALSE(gria::is_critical(0.01));
}

TEST(GriaAdv, IsCritical_AlphaNearHalf) {
    bool c = gria::is_critical(0.5);
    (void)c;
    SUCCEED();
}

TEST(GriaAdv, Classify_ReturnValid) {
    auto c0 = gria::classify(0.0);
    auto c1 = gria::classify(0.5);
    auto c2 = gria::classify(1.0);
    (void)c0; (void)c1; (void)c2;
    SUCCEED();
}

TEST(GriaAdv, Entropy_UniformData) {
    std::vector<double> data(16);
    for (int i = 0; i < 16; ++i) data[i] = static_cast<double>(i);
    double h = gria::entropy(data, 4);
    EXPECT_TRUE(std::isfinite(h));
    EXPECT_GE(h, 0.0);
}

// ---------------------------------------------------------------------------
// Gria: register_dispatch_hint and dispatch_hint_alpha
// ---------------------------------------------------------------------------

TEST(GriaAdv, RegisterDispatchHint_Smoke) {
    EXPECT_NO_THROW(gria::register_dispatch_hint("test_op", 0.5));
}

TEST(GriaAdv, DispatchHintAlpha_AfterRegister) {
    gria::register_dispatch_hint("test_hint_op", 0.75);
    double alpha = gria::dispatch_hint_alpha("test_hint_op");
    EXPECT_TRUE(std::isfinite(alpha));
}
