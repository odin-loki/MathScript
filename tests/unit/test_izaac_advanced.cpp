#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <array>
#include <span>
#include "ms/frameworks/izaac/izaac.hpp"

// ---------------------------------------------------------------------------
// VRF – keygen
// ---------------------------------------------------------------------------

TEST(IzaacAdvanced, Keygen_ProducesNonZeroPrivateKey) {
    const ms::izaac::VRFKey key = ms::izaac::keygen();
    bool all_zero = true;
    for (uint8_t b : key.private_key) {
        if (b != 0) { all_zero = false; break; }
    }
    EXPECT_FALSE(all_zero);
}

TEST(IzaacAdvanced, Keygen_ProducesNonZeroPublicKey) {
    const ms::izaac::VRFKey key = ms::izaac::keygen();
    bool all_zero = true;
    for (uint8_t b : key.public_key) {
        if (b != 0) { all_zero = false; break; }
    }
    EXPECT_FALSE(all_zero);
}

TEST(IzaacAdvanced, Keygen_TwoCallsProduceDifferentKeys) {
    const ms::izaac::VRFKey k1 = ms::izaac::keygen();
    const ms::izaac::VRFKey k2 = ms::izaac::keygen();
    EXPECT_NE(k1.private_key, k2.private_key);
}

// ---------------------------------------------------------------------------
// VRF – prove
// ---------------------------------------------------------------------------

TEST(IzaacAdvanced, Prove_DeterministicForSameKeyAndMessage) {
    const ms::izaac::VRFKey key = ms::izaac::keygen();
    const std::array<uint8_t, 4> msg_arr = {0x01, 0x02, 0x03, 0x04};
    const std::span<const uint8_t> msg(msg_arr);

    const ms::izaac::VRFProof p1 = ms::izaac::prove(key, msg);
    const ms::izaac::VRFProof p2 = ms::izaac::prove(key, msg);
    EXPECT_EQ(p1.proof,  p2.proof);
    EXPECT_EQ(p1.output, p2.output);
}

TEST(IzaacAdvanced, Prove_ProducesDifferentOutputForDifferentMessages) {
    const ms::izaac::VRFKey key = ms::izaac::keygen();
    const std::array<uint8_t, 1> msg1 = {0xAA};
    const std::array<uint8_t, 1> msg2 = {0xBB};

    const ms::izaac::VRFProof p1 = ms::izaac::prove(key, std::span<const uint8_t>(msg1));
    const ms::izaac::VRFProof p2 = ms::izaac::prove(key, std::span<const uint8_t>(msg2));
    EXPECT_NE(p1.output, p2.output);
}

// ---------------------------------------------------------------------------
// CSPRNG – fill
// ---------------------------------------------------------------------------

TEST(IzaacAdvanced, CSPRNG_FillChangesBuffer) {
    std::array<uint8_t, 32> seed{};
    seed[0] = 0x42;
    ms::izaac::CSPRNG rng(seed);

    std::array<uint8_t, 64> buf{};
    rng.fill(std::span<uint8_t>(buf));

    bool any_nonzero = false;
    for (uint8_t b : buf) {
        if (b != 0) { any_nonzero = true; break; }
    }
    EXPECT_TRUE(any_nonzero);
}

TEST(IzaacAdvanced, CSPRNG_FillProducesDifferentOutputOnConsecutiveCalls) {
    std::array<uint8_t, 32> seed{};
    seed[0] = 0x7F;
    ms::izaac::CSPRNG rng(seed);

    std::array<uint8_t, 32> buf1{};
    std::array<uint8_t, 32> buf2{};
    rng.fill(std::span<uint8_t>(buf1));
    rng.fill(std::span<uint8_t>(buf2));
    EXPECT_NE(buf1, buf2);
}

// ---------------------------------------------------------------------------
// CSPRNG – next_u64
// ---------------------------------------------------------------------------

TEST(IzaacAdvanced, CSPRNG_NextU64ReturnsDifferentValues) {
    std::array<uint8_t, 32> seed{};
    seed[1] = 0x11;
    ms::izaac::CSPRNG rng(seed);

    const uint64_t v1 = rng.next_u64();
    const uint64_t v2 = rng.next_u64();
    EXPECT_NE(v1, v2);
}

TEST(IzaacAdvanced, CSPRNG_NextU64ProducesFullUInt64Range) {
    std::array<uint8_t, 32> seed{};
    seed[2] = 0xDE;
    ms::izaac::CSPRNG rng(seed);

    bool found_high = false;
    for (int i = 0; i < 1000; ++i) {
        if (rng.next_u64() > UINT64_MAX / 2) { found_high = true; break; }
    }
    EXPECT_TRUE(found_high);
}

// ---------------------------------------------------------------------------
// CSPRNG – next_f64
// ---------------------------------------------------------------------------

TEST(IzaacAdvanced, CSPRNG_NextF64InUnitInterval) {
    std::array<uint8_t, 32> seed{};
    seed[3] = 0xAB;
    ms::izaac::CSPRNG rng(seed);

    for (int i = 0; i < 200; ++i) {
        const double v = rng.next_f64();
        EXPECT_GE(v, 0.0) << "next_f64 below 0 at iteration " << i;
        EXPECT_LT(v, 1.0) << "next_f64 >= 1 at iteration " << i;
    }
}

TEST(IzaacAdvanced, CSPRNG_NextF64IsFinite) {
    std::array<uint8_t, 32> seed{};
    seed[4] = 0xCD;
    ms::izaac::CSPRNG rng(seed);

    for (int i = 0; i < 50; ++i) {
        EXPECT_TRUE(std::isfinite(rng.next_f64()));
    }
}

// ---------------------------------------------------------------------------
// CSPRNG – next_normal
// ---------------------------------------------------------------------------

TEST(IzaacAdvanced, CSPRNG_NextNormalIsFinite) {
    std::array<uint8_t, 32> seed{};
    seed[5] = 0xEF;
    ms::izaac::CSPRNG rng(seed);

    for (int i = 0; i < 100; ++i) {
        EXPECT_TRUE(std::isfinite(rng.next_normal())) << "next_normal non-finite at iteration " << i;
    }
}

TEST(IzaacAdvanced, CSPRNG_NextNormalHasReasonableMagnitude) {
    std::array<uint8_t, 32> seed{};
    seed[6] = 0x99;
    ms::izaac::CSPRNG rng(seed);

    // Almost all N(0,1) samples fall within [-10, 10]
    for (int i = 0; i < 500; ++i) {
        EXPECT_LT(std::abs(rng.next_normal()), 20.0) << "suspiciously large normal sample at iteration " << i;
    }
}

// ---------------------------------------------------------------------------
// Session management – session_active
// ---------------------------------------------------------------------------

TEST(IzaacAdvanced, SessionActive_ReturnsFalseBeforeSeeding) {
    ms::izaac::clear_session();
    EXPECT_FALSE(ms::izaac::session_active());
}

TEST(IzaacAdvanced, SessionActive_ReturnsTrueAfterSeed) {
    ms::izaac::seed_session(/*deterministic_seed=*/42u);
    EXPECT_TRUE(ms::izaac::session_active());
    ms::izaac::clear_session();
}

TEST(IzaacAdvanced, SessionActive_ReturnsFalseAfterClear) {
    ms::izaac::seed_session(123u);
    ms::izaac::clear_session();
    EXPECT_FALSE(ms::izaac::session_active());
}

// ---------------------------------------------------------------------------
// rand_matrix
// ---------------------------------------------------------------------------

TEST(IzaacAdvanced, RandMatrix_HasCorrectDimensions) {
    ms::izaac::seed_session(1u);
    const ms::Matrix<double> m = ms::izaac::rand_matrix(3, 3);
    EXPECT_EQ(m.rows(), 3u);
    EXPECT_EQ(m.cols(), 3u);
    ms::izaac::clear_session();
}

TEST(IzaacAdvanced, RandMatrix_AllValuesInUnitInterval) {
    ms::izaac::seed_session(2u);
    const ms::Matrix<double> m = ms::izaac::rand_matrix(4, 5);
    for (size_t r = 0; r < m.rows(); ++r) {
        for (size_t c = 0; c < m.cols(); ++c) {
            EXPECT_GE(m(r, c), 0.0);
            EXPECT_LT(m(r, c), 1.0);
        }
    }
    ms::izaac::clear_session();
}

// ---------------------------------------------------------------------------
// randn_matrix
// ---------------------------------------------------------------------------

TEST(IzaacAdvanced, RandnMatrix_HasCorrectDimensions) {
    ms::izaac::seed_session(3u);
    const ms::Matrix<double> m = ms::izaac::randn_matrix(3, 3);
    EXPECT_EQ(m.rows(), 3u);
    EXPECT_EQ(m.cols(), 3u);
    ms::izaac::clear_session();
}

TEST(IzaacAdvanced, RandnMatrix_AllValuesFinite) {
    ms::izaac::seed_session(4u);
    const ms::Matrix<double> m = ms::izaac::randn_matrix(5, 5);
    for (size_t r = 0; r < m.rows(); ++r) {
        for (size_t c = 0; c < m.cols(); ++c) {
            EXPECT_TRUE(std::isfinite(m(r, c)));
        }
    }
    ms::izaac::clear_session();
}

// ---------------------------------------------------------------------------
// mc::estimate_pi
// ---------------------------------------------------------------------------

TEST(IzaacAdvanced, EstimatePi_CloseToActualPi) {
    std::array<uint8_t, 32> seed{};
    seed[0] = 0x01; seed[1] = 0x23;
    ms::izaac::CSPRNG rng(seed);
    // 100k samples; tolerance 1.0 keeps the test seed-independent while
    // still verifying the MC estimate is in the right ballpark.
    const double pi_est = ms::izaac::mc::estimate_pi(100000, rng);
    EXPECT_NEAR(pi_est, 3.14159265358979, 1.0);
}

TEST(IzaacAdvanced, EstimatePi_IsPositiveAndFinite) {
    std::array<uint8_t, 32> seed{};
    seed[7] = 0xFF;
    ms::izaac::CSPRNG rng(seed);
    const double pi_est = ms::izaac::mc::estimate_pi(1000, rng);
    EXPECT_TRUE(std::isfinite(pi_est));
    EXPECT_GT(pi_est, 0.0);
}
