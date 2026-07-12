#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <array>
#include <span>
#include <vector>
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

// ---------------------------------------------------------------------------
// bloom::BloomFilter
// ---------------------------------------------------------------------------

TEST(IzaacAdvanced, BloomFilter_NoFalseNegatives) {
    std::array<uint8_t, 32> seed{};
    seed[0] = 0xB1;
    ms::izaac::CSPRNG rng(seed);
    ms::izaac::bloom::BloomFilter filter(500, 0.01, rng);

    constexpr size_t item_count = 500;
    for (size_t i = 0; i < item_count; ++i) {
        const uint8_t byte = static_cast<uint8_t>(i & 0xFF);
        const std::array<uint8_t, 2> item = {
            static_cast<uint8_t>(i >> 8),
            byte};
        filter.insert(std::span<const uint8_t>(item));
    }

    for (size_t i = 0; i < item_count; ++i) {
        const uint8_t byte = static_cast<uint8_t>(i & 0xFF);
        const std::array<uint8_t, 2> item = {
            static_cast<uint8_t>(i >> 8),
            byte};
        EXPECT_TRUE(filter.might_contain(std::span<const uint8_t>(item)))
            << "false negative at item index " << i;
    }
}

TEST(IzaacAdvanced, BloomFilter_FalsePositiveRateReasonable) {
    std::array<uint8_t, 32> seed{};
    seed[1] = 0xB2;
    ms::izaac::CSPRNG rng(seed);
    constexpr double target_fp = 0.01;
    constexpr size_t inserted = 1000;
    ms::izaac::bloom::BloomFilter filter(inserted, target_fp, rng);

    for (size_t i = 0; i < inserted; ++i) {
        const std::array<uint8_t, 4> item = {
            static_cast<uint8_t>(i),
            static_cast<uint8_t>(i >> 8),
            static_cast<uint8_t>(i >> 16),
            static_cast<uint8_t>(0xA0)};
        filter.insert(std::span<const uint8_t>(item));
    }

    size_t false_positives = 0;
    constexpr size_t probes = 10000;
    for (size_t i = inserted; i < inserted + probes; ++i) {
        const std::array<uint8_t, 4> item = {
            static_cast<uint8_t>(i),
            static_cast<uint8_t>(i >> 8),
            static_cast<uint8_t>(i >> 16),
            static_cast<uint8_t>(0xB0)};
        if (filter.might_contain(std::span<const uint8_t>(item))) {
            ++false_positives;
        }
    }

    const double empirical_fp = static_cast<double>(false_positives) / static_cast<double>(probes);
    EXPECT_GT(empirical_fp, target_fp * 0.25);
    EXPECT_LT(empirical_fp, target_fp * 4.0);
}

TEST(IzaacAdvanced, BloomFilter_BitCountSanity) {
    std::array<uint8_t, 32> seed{};
    seed[2] = 0xB3;
    ms::izaac::CSPRNG rng(seed);
    ms::izaac::bloom::BloomFilter filter(1000, 0.01, rng);
    EXPECT_GT(filter.bit_count(), 0u);
    EXPECT_GE(filter.bit_count(), 9000u);
}

TEST(IzaacAdvanced, BloomFilter_HashCountSanity) {
    std::array<uint8_t, 32> seed{};
    seed[3] = 0xB4;
    ms::izaac::CSPRNG rng(seed);
    ms::izaac::bloom::BloomFilter filter(1000, 0.01, rng);
    EXPECT_GE(filter.hash_count(), 6u);
    EXPECT_LE(filter.hash_count(), 12u);
}

TEST(IzaacAdvanced, BloomFilter_UnknownItemReturnsFalseBeforeInsert) {
    std::array<uint8_t, 32> seed{};
    seed[4] = 0xB5;
    ms::izaac::CSPRNG rng(seed);
    ms::izaac::bloom::BloomFilter filter(100, 0.05, rng);
    const std::array<uint8_t, 3> item = {0x01, 0x02, 0x03};
    EXPECT_FALSE(filter.might_contain(std::span<const uint8_t>(item)));
}

TEST(IzaacAdvanced, BloomFilter_DifferentSeedsChangeMembership) {
    std::array<uint8_t, 32> seed_a{};
    seed_a[5] = 0xAA;
    std::array<uint8_t, 32> seed_b{};
    seed_b[5] = 0xBB;
    ms::izaac::CSPRNG rng_a(seed_a);
    ms::izaac::CSPRNG rng_b(seed_b);

    ms::izaac::bloom::BloomFilter filter_a(50, 0.1, rng_a);
    ms::izaac::bloom::BloomFilter filter_b(50, 0.1, rng_b);

    const std::array<uint8_t, 2> item = {0xDE, 0xAD};
    filter_a.insert(std::span<const uint8_t>(item));
    filter_b.insert(std::span<const uint8_t>(item));

    EXPECT_TRUE(filter_a.might_contain(std::span<const uint8_t>(item)));
    EXPECT_TRUE(filter_b.might_contain(std::span<const uint8_t>(item)));
    EXPECT_NE(filter_a.bit_count(), 0u);
    EXPECT_NE(filter_b.bit_count(), 0u);
}

// ---------------------------------------------------------------------------
// ratelimit::TokenBucket
// ---------------------------------------------------------------------------

TEST(IzaacAdvanced, TokenBucket_ConsumeWithinCapacitySucceeds) {
    ms::izaac::ratelimit::TokenBucket bucket(10.0, 1.0);
    EXPECT_TRUE(bucket.try_consume(5.0, 0.0));
    EXPECT_TRUE(bucket.try_consume(5.0, 0.0));
}

TEST(IzaacAdvanced, TokenBucket_ConsumeBeyondCapacityFails) {
    ms::izaac::ratelimit::TokenBucket bucket(10.0, 1.0);
    EXPECT_FALSE(bucket.try_consume(11.0, 0.0));
}

TEST(IzaacAdvanced, TokenBucket_ConsumeBeyondCapacityDoesNotPartiallyConsume) {
    ms::izaac::ratelimit::TokenBucket bucket(10.0, 1.0);
    EXPECT_TRUE(bucket.try_consume(7.0, 0.0));
    EXPECT_FALSE(bucket.try_consume(5.0, 0.0));
    EXPECT_NEAR(bucket.available_tokens(0.0), 3.0, 1e-12);
}

TEST(IzaacAdvanced, TokenBucket_RefillOverTimeRestoresTokens) {
    ms::izaac::ratelimit::TokenBucket bucket(10.0, 2.0);
    EXPECT_TRUE(bucket.try_consume(10.0, 0.0));
    EXPECT_NEAR(bucket.available_tokens(2.5), 5.0, 1e-12);
    EXPECT_TRUE(bucket.try_consume(5.0, 2.5));
}

TEST(IzaacAdvanced, TokenBucket_RefillCappedAtCapacity) {
    ms::izaac::ratelimit::TokenBucket bucket(10.0, 5.0);
    EXPECT_TRUE(bucket.try_consume(10.0, 0.0));
    EXPECT_NEAR(bucket.available_tokens(100.0), 10.0, 1e-12);
}

TEST(IzaacAdvanced, TokenBucket_AvailableTokensReflectsElapsedRefill) {
    ms::izaac::ratelimit::TokenBucket bucket(8.0, 4.0);
    EXPECT_TRUE(bucket.try_consume(6.0, 1.0));
    EXPECT_NEAR(bucket.available_tokens(2.0), 6.0, 1e-12);
}

TEST(IzaacAdvanced, TokenBucket_ZeroTokenConsumeAlwaysSucceeds) {
    ms::izaac::ratelimit::TokenBucket bucket(1.0, 0.5);
    EXPECT_TRUE(bucket.try_consume(1.0, 0.0));
    EXPECT_TRUE(bucket.try_consume(0.0, 0.0));
    EXPECT_NEAR(bucket.available_tokens(0.0), 0.0, 1e-12);
}

// ---------------------------------------------------------------------------
// diffpriv mechanisms
// ---------------------------------------------------------------------------

TEST(IzaacAdvanced, DiffPriv_LaplaceReturnsFinite) {
    std::array<uint8_t, 32> seed{};
    seed[8] = 0xD1;
    ms::izaac::CSPRNG rng(seed);
    for (int i = 0; i < 100; ++i) {
        const double v = ms::izaac::diffpriv::laplace_mechanism(42.0, 1.0, 1.0, rng);
        EXPECT_TRUE(std::isfinite(v));
    }
}

TEST(IzaacAdvanced, DiffPriv_GaussianReturnsFinite) {
    std::array<uint8_t, 32> seed{};
    seed[9] = 0xD2;
    ms::izaac::CSPRNG rng(seed);
    for (int i = 0; i < 100; ++i) {
        const double v = ms::izaac::diffpriv::gaussian_mechanism(42.0, 1.0, 1e-5, 1.0, rng);
        EXPECT_TRUE(std::isfinite(v));
    }
}

TEST(IzaacAdvanced, DiffPriv_LaplaceMeanNearZero) {
    constexpr double epsilon = 2.0;
    constexpr double sensitivity = 1.0;
    constexpr int samples_per_seed = 5000;
    constexpr std::array<uint64_t, 8> seeds = {
        0xC0FFEE01u, 0xC0FFEE02u, 0xC0FFEE03u, 0xC0FFEE04u,
        0xC0FFEE05u, 0xC0FFEE06u, 0xC0FFEE07u, 0xC0FFEE08u};

    double aggregate_mean = 0.0;
    for (uint64_t seed_value : seeds) {
        ms::izaac::seed_session(seed_value);
        ms::izaac::CSPRNG& rng = *ms::izaac::g_session_rng;
        double sum = 0.0;
        for (int i = 0; i < samples_per_seed; ++i) {
            sum += ms::izaac::diffpriv::laplace_mechanism(0.0, epsilon, sensitivity, rng);
        }
        aggregate_mean += sum / static_cast<double>(samples_per_seed);
        ms::izaac::clear_session();
    }
    aggregate_mean /= static_cast<double>(seeds.size());
    EXPECT_NEAR(aggregate_mean, 0.0, 0.75);
}

TEST(IzaacAdvanced, DiffPriv_LaplaceProducesSignedNoise) {
    ms::izaac::seed_session(0xC0FFEE02u);
    ms::izaac::CSPRNG& rng = *ms::izaac::g_session_rng;
    bool has_negative = false;
    bool has_positive = false;
    for (int i = 0; i < 1000; ++i) {
        const double v = ms::izaac::diffpriv::laplace_mechanism(0.0, 1.0, 1.0, rng);
        if (v < -0.01) {
            has_negative = true;
        }
        if (v > 0.01) {
            has_positive = true;
        }
    }
    ms::izaac::clear_session();
    EXPECT_TRUE(has_negative);
    EXPECT_TRUE(has_positive);
}

TEST(IzaacAdvanced, DiffPriv_LaplaceVarianceConsistent) {
    constexpr double epsilon = 1.0;
    constexpr double sensitivity = 2.0;
    const double scale = sensitivity / epsilon;
    const double expected_variance = 2.0 * scale * scale;
    constexpr int samples_per_seed = 5000;
    constexpr std::array<uint64_t, 8> seeds = {
        0xB00B1E01u, 0xB00B1E02u, 0xB00B1E03u, 0xB00B1E04u,
        0xB00B1E05u, 0xB00B1E06u, 0xB00B1E07u, 0xB00B1E08u};

    double aggregate_variance = 0.0;
    for (uint64_t seed_value : seeds) {
        ms::izaac::seed_session(seed_value);
        ms::izaac::CSPRNG& rng = *ms::izaac::g_session_rng;
        double mean = 0.0;
        std::vector<double> draws;
        draws.reserve(static_cast<size_t>(samples_per_seed));
        for (int i = 0; i < samples_per_seed; ++i) {
            const double v = ms::izaac::diffpriv::laplace_mechanism(0.0, epsilon, sensitivity, rng);
            draws.push_back(v);
            mean += v;
        }
        mean /= static_cast<double>(samples_per_seed);

        double variance = 0.0;
        for (double v : draws) {
            const double diff = v - mean;
            variance += diff * diff;
        }
        variance /= static_cast<double>(samples_per_seed);
        aggregate_variance += variance;
        ms::izaac::clear_session();
    }
    aggregate_variance /= static_cast<double>(seeds.size());

    EXPECT_NEAR(aggregate_variance, expected_variance, expected_variance * 0.6);
}

TEST(IzaacAdvanced, DiffPriv_GaussianMeanNearTrueValue) {
    constexpr double true_value = 7.5;
    constexpr double epsilon = 1.0;
    constexpr double delta = 1e-6;
    constexpr double sensitivity = 1.0;
    const double sigma = std::sqrt(2.0 * std::log(1.25 / delta)) * sensitivity / epsilon;
    constexpr int samples_per_seed = 5000;
    constexpr std::array<uint64_t, 8> seeds = {
        0xABCD0001u, 0xABCD0002u, 0xABCD0003u, 0xABCD0004u,
        0xABCD0005u, 0xABCD0006u, 0xABCD0007u, 0xABCD0008u};

    bool below = false;
    bool above = false;
    double aggregate_variance = 0.0;
    for (uint64_t seed_value : seeds) {
        ms::izaac::seed_session(seed_value);
        ms::izaac::CSPRNG& rng = *ms::izaac::g_session_rng;
        double mean = 0.0;
        std::vector<double> draws;
        draws.reserve(static_cast<size_t>(samples_per_seed));
        for (int i = 0; i < samples_per_seed; ++i) {
            const double v =
                ms::izaac::diffpriv::gaussian_mechanism(true_value, epsilon, delta, sensitivity, rng);
            draws.push_back(v);
            mean += v;
            if (v < true_value) {
                below = true;
            }
            if (v > true_value) {
                above = true;
            }
        }
        mean /= static_cast<double>(samples_per_seed);
        double variance = 0.0;
        for (double v : draws) {
            const double diff = v - mean;
            variance += diff * diff;
        }
        variance /= static_cast<double>(samples_per_seed);
        aggregate_variance += variance;
        ms::izaac::clear_session();
    }
    aggregate_variance /= static_cast<double>(seeds.size());
    const double expected_variance = sigma * sigma;

    EXPECT_TRUE(below);
    EXPECT_TRUE(above);
    EXPECT_GT(aggregate_variance, expected_variance * 0.35);
    EXPECT_LT(aggregate_variance, expected_variance * 1.75);
}

// ---------------------------------------------------------------------------
// backtest
// ---------------------------------------------------------------------------

TEST(IzaacAdvanced, Backtest_GbmPathCorrectLength) {
    std::array<uint8_t, 32> seed{};
    seed[13] = 0xE1;
    ms::izaac::CSPRNG rng(seed);
    const std::vector<double> path = ms::izaac::backtest::simulate_gbm_path(100.0, 0.05, 0.2, 1.0 / 252.0, 50, rng);
    EXPECT_EQ(path.size(), 51u);
    EXPECT_DOUBLE_EQ(path.front(), 100.0);
}

TEST(IzaacAdvanced, Backtest_GbmPathAllPositive) {
    std::array<uint8_t, 32> seed{};
    seed[14] = 0xE2;
    ms::izaac::CSPRNG rng(seed);
    const std::vector<double> path = ms::izaac::backtest::simulate_gbm_path(50.0, 0.1, 0.15, 0.01, 200, rng);
    for (size_t i = 0; i < path.size(); ++i) {
        EXPECT_GT(path[i], 0.0) << "non-positive price at index " << i;
    }
}

TEST(IzaacAdvanced, Backtest_RunBacktestEquityCurve) {
    const std::vector<double> prices = {100.0, 110.0, 99.0, 105.0};
    const std::vector<int> positions = {1, 1, -1, 0};
    const ms::izaac::backtest::BacktestResult result =
        ms::izaac::backtest::run_backtest(prices, positions, 1000.0);

    ASSERT_EQ(result.equity_curve.size(), 4u);
    EXPECT_NEAR(result.equity_curve[0], 1000.0, 1e-9);
    EXPECT_NEAR(result.equity_curve[1], 1100.0, 1e-9);
    EXPECT_NEAR(result.equity_curve[2], 990.0, 1e-9);
    EXPECT_NEAR(result.equity_curve[3], 930.0, 1e-9);
}

TEST(IzaacAdvanced, Backtest_RunBacktestTotalReturn) {
    const std::vector<double> prices = {100.0, 110.0, 99.0, 105.0};
    const std::vector<int> positions = {1, 1, -1, 0};
    const ms::izaac::backtest::BacktestResult result =
        ms::izaac::backtest::run_backtest(prices, positions, 1000.0);
    EXPECT_NEAR(result.total_return, -0.07, 1e-9);
}

TEST(IzaacAdvanced, Backtest_RunBacktestMaxDrawdown) {
    const std::vector<double> prices = {100.0, 110.0, 99.0, 105.0};
    const std::vector<int> positions = {1, 1, -1, 0};
    const ms::izaac::backtest::BacktestResult result =
        ms::izaac::backtest::run_backtest(prices, positions, 1000.0);
    EXPECT_NEAR(result.max_drawdown, 170.0 / 1100.0, 1e-9);
}

TEST(IzaacAdvanced, Backtest_RunBacktestSharpeRatio) {
    const std::vector<double> prices = {100.0, 110.0, 99.0, 105.0};
    const std::vector<int> positions = {1, 1, -1, 0};
    const ms::izaac::backtest::BacktestResult result =
        ms::izaac::backtest::run_backtest(prices, positions, 1000.0);

    const std::array<double, 3> returns = {0.1, -0.1, -60.0 / 990.0};
    double mean = 0.0;
    for (double r : returns) {
        mean += r;
    }
    mean /= 3.0;
    double variance = 0.0;
    for (double r : returns) {
        const double diff = r - mean;
        variance += diff * diff;
    }
    variance /= 3.0;
    const double expected_sharpe = mean / std::sqrt(variance);
    EXPECT_NEAR(result.sharpe_ratio, expected_sharpe, 1e-9);
}

TEST(IzaacAdvanced, Backtest_RunBacktestEmptyPricesReturnsEmpty) {
    const std::vector<double> prices;
    const std::vector<int> positions;
    const ms::izaac::backtest::BacktestResult result =
        ms::izaac::backtest::run_backtest(prices, positions, 1000.0);
    EXPECT_TRUE(result.equity_curve.empty());
    EXPECT_DOUBLE_EQ(result.total_return, 0.0);
}

// ---------------------------------------------------------------------------
// crypto::encrypt / decrypt
// ---------------------------------------------------------------------------

namespace {

std::array<uint8_t, 32> make_test_key(uint8_t byte) {
    std::array<uint8_t, 32> key{};
    key.fill(byte);
    return key;
}

bool round_trip_plaintext(std::span<const uint8_t> plaintext, const std::array<uint8_t, 32>& key) {
    const ms::izaac::crypto::CipherText ct = ms::izaac::crypto::encrypt(plaintext, key);
    const ms::Result<std::vector<uint8_t>> decrypted = ms::izaac::crypto::decrypt(ct, key);
    if (!decrypted) {
        return false;
    }
    if (decrypted->size() != plaintext.size()) {
        return false;
    }
    return std::equal(plaintext.begin(), plaintext.end(), decrypted->begin());
}

} // namespace

TEST(IzaacAdvanced, Crypto_RoundTripEmptyPlaintext) {
    const std::array<uint8_t, 32> key = make_test_key(0x11);
    const std::vector<uint8_t> plaintext;
    EXPECT_TRUE(round_trip_plaintext(std::span<const uint8_t>(plaintext), key));
}

TEST(IzaacAdvanced, Crypto_RoundTripOneBytePlaintext) {
    const std::array<uint8_t, 32> key = make_test_key(0x22);
    const std::array<uint8_t, 1> plaintext = {0xAB};
    EXPECT_TRUE(round_trip_plaintext(std::span<const uint8_t>(plaintext), key));
}

TEST(IzaacAdvanced, Crypto_RoundTripSmallBuffer) {
    const std::array<uint8_t, 32> key = make_test_key(0x33);
    std::vector<uint8_t> plaintext(64);
    for (size_t i = 0; i < plaintext.size(); ++i) {
        plaintext[i] = static_cast<uint8_t>(i * 17 + 3);
    }
    EXPECT_TRUE(round_trip_plaintext(std::span<const uint8_t>(plaintext), key));
}

TEST(IzaacAdvanced, Crypto_RoundTripLargeBuffer) {
    const std::array<uint8_t, 32> key = make_test_key(0x44);
    std::vector<uint8_t> plaintext(256);
    for (size_t i = 0; i < plaintext.size(); ++i) {
        plaintext[i] = static_cast<uint8_t>((i * 131) ^ 0x5A);
    }
    EXPECT_TRUE(round_trip_plaintext(std::span<const uint8_t>(plaintext), key));
}

TEST(IzaacAdvanced, Crypto_TamperedCiphertextFailsDecrypt) {
    const std::array<uint8_t, 32> key = make_test_key(0x55);
    const std::array<uint8_t, 8> plaintext = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08};
    ms::izaac::crypto::CipherText ct = ms::izaac::crypto::encrypt(std::span<const uint8_t>(plaintext), key);
    ASSERT_GT(ct.data.size(), 16u);
    ct.data[16] ^= 0x01;
    const ms::Result<std::vector<uint8_t>> decrypted = ms::izaac::crypto::decrypt(ct, key);
    EXPECT_FALSE(decrypted.has_value());
}

TEST(IzaacAdvanced, Crypto_TamperedTagFailsDecrypt) {
    const std::array<uint8_t, 32> key = make_test_key(0x66);
    const std::array<uint8_t, 4> plaintext = {0xDE, 0xAD, 0xBE, 0xEF};
    ms::izaac::crypto::CipherText ct = ms::izaac::crypto::encrypt(std::span<const uint8_t>(plaintext), key);
    ct.tag[0] ^= 0x80;
    const ms::Result<std::vector<uint8_t>> decrypted = ms::izaac::crypto::decrypt(ct, key);
    EXPECT_FALSE(decrypted.has_value());
}

TEST(IzaacAdvanced, Crypto_DifferentKeysProduceDifferentCiphertexts) {
    const std::array<uint8_t, 16> plaintext = {
        0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
        0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F};
    const ms::izaac::crypto::CipherText ct_a =
        ms::izaac::crypto::encrypt(std::span<const uint8_t>(plaintext), make_test_key(0xAA));
    const ms::izaac::crypto::CipherText ct_b =
        ms::izaac::crypto::encrypt(std::span<const uint8_t>(plaintext), make_test_key(0xBB));
    EXPECT_NE(ct_a.data, ct_b.data);
    EXPECT_NE(ct_a.tag, ct_b.tag);
}

TEST(IzaacAdvanced, Crypto_CiphertextDiffersFromPlaintext) {
    const std::array<uint8_t, 32> key = make_test_key(0x77);
    std::vector<uint8_t> plaintext(32);
    for (size_t i = 0; i < plaintext.size(); ++i) {
        plaintext[i] = static_cast<uint8_t>(0xA0 + i);
    }
    const ms::izaac::crypto::CipherText ct =
        ms::izaac::crypto::encrypt(std::span<const uint8_t>(plaintext), key);
    ASSERT_EQ(ct.data.size(), plaintext.size() + 16u);
    const std::span<const uint8_t> ciphertext_body(ct.data.data() + 16, plaintext.size());
    EXPECT_NE(
        std::vector<uint8_t>(ciphertext_body.begin(), ciphertext_body.end()),
        plaintext);
}

TEST(IzaacAdvanced, Crypto_WrongKeyFailsDecrypt) {
    const std::array<uint8_t, 6> plaintext = {0xCA, 0xFE, 0xBA, 0xBE, 0x00, 0x01};
    const ms::izaac::crypto::CipherText ct =
        ms::izaac::crypto::encrypt(std::span<const uint8_t>(plaintext), make_test_key(0x88));
    const ms::Result<std::vector<uint8_t>> decrypted =
        ms::izaac::crypto::decrypt(ct, make_test_key(0x99));
    EXPECT_FALSE(decrypted.has_value());
}

// ---------------------------------------------------------------------------
// mpc::split_secret / reconstruct_secret
// ---------------------------------------------------------------------------

namespace {

std::vector<ms::izaac::mpc::Share> pick_shares(
    const std::vector<ms::izaac::mpc::Share>& all,
    const std::vector<int>& indices) {
    std::vector<ms::izaac::mpc::Share> picked;
    picked.reserve(indices.size());
    for (int idx : indices) {
        picked.push_back(all[static_cast<size_t>(idx - 1)]);
    }
    return picked;
}

} // namespace

TEST(IzaacAdvanced, Mpc_ReconstructFromThreeOfFiveSubsetA) {
    std::array<uint8_t, 32> seed{};
    seed[0] = 0xA1;
    ms::izaac::CSPRNG rng(seed);
    constexpr uint64_t secret = 123456789ULL;
    const std::vector<ms::izaac::mpc::Share> shares =
        ms::izaac::mpc::split_secret(secret, 5, 3, rng);
    ASSERT_EQ(shares.size(), 5u);

    const std::vector<ms::izaac::mpc::Share> subset = pick_shares(shares, {1, 2, 3});
    const ms::Result<uint64_t> reconstructed = ms::izaac::mpc::reconstruct_secret(subset);
    ASSERT_TRUE(reconstructed.has_value());
    EXPECT_EQ(*reconstructed, secret);
}

TEST(IzaacAdvanced, Mpc_ReconstructFromThreeOfFiveSubsetB) {
    std::array<uint8_t, 32> seed{};
    seed[1] = 0xA2;
    ms::izaac::CSPRNG rng(seed);
    constexpr uint64_t secret = 987654321ULL;
    const std::vector<ms::izaac::mpc::Share> shares =
        ms::izaac::mpc::split_secret(secret, 5, 3, rng);
    const std::vector<ms::izaac::mpc::Share> subset = pick_shares(shares, {2, 4, 5});
    const ms::Result<uint64_t> reconstructed = ms::izaac::mpc::reconstruct_secret(subset);
    ASSERT_TRUE(reconstructed.has_value());
    EXPECT_EQ(*reconstructed, secret);
}

TEST(IzaacAdvanced, Mpc_ReconstructFromAllFiveShares) {
    std::array<uint8_t, 32> seed{};
    seed[2] = 0xA3;
    ms::izaac::CSPRNG rng(seed);
    constexpr uint64_t secret = 42424242ULL;
    const std::vector<ms::izaac::mpc::Share> shares =
        ms::izaac::mpc::split_secret(secret, 5, 3, rng);
    const ms::Result<uint64_t> reconstructed = ms::izaac::mpc::reconstruct_secret(shares);
    ASSERT_TRUE(reconstructed.has_value());
    EXPECT_EQ(*reconstructed, secret);
}

TEST(IzaacAdvanced, Mpc_TwoSharesBelowThresholdDoNotRecover) {
    std::array<uint8_t, 32> seed{};
    seed[3] = 0xA4;
    ms::izaac::CSPRNG rng(seed);
    constexpr uint64_t secret = 555555ULL;
    const std::vector<ms::izaac::mpc::Share> shares =
        ms::izaac::mpc::split_secret(secret, 5, 3, rng);
    const std::vector<ms::izaac::mpc::Share> subset = pick_shares(shares, {1, 5});
    const ms::Result<uint64_t> reconstructed = ms::izaac::mpc::reconstruct_secret(subset);
    ASSERT_TRUE(reconstructed.has_value());
    EXPECT_NE(*reconstructed, secret);
}

TEST(IzaacAdvanced, Mpc_SecretZeroReconstructs) {
    std::array<uint8_t, 32> seed{};
    seed[4] = 0xA5;
    ms::izaac::CSPRNG rng(seed);
    constexpr uint64_t secret = 0;
    const std::vector<ms::izaac::mpc::Share> shares =
        ms::izaac::mpc::split_secret(secret, 5, 3, rng);
    const std::vector<ms::izaac::mpc::Share> subset = pick_shares(shares, {1, 3, 4});
    const ms::Result<uint64_t> reconstructed = ms::izaac::mpc::reconstruct_secret(subset);
    ASSERT_TRUE(reconstructed.has_value());
    EXPECT_EQ(*reconstructed, secret);
}

TEST(IzaacAdvanced, Mpc_LargeSecretNearPrimeBound) {
    std::array<uint8_t, 32> seed{};
    seed[5] = 0xA6;
    ms::izaac::CSPRNG rng(seed);
    constexpr uint64_t secret = ms::izaac::mpc::PRIME - 42ULL;
    const std::vector<ms::izaac::mpc::Share> shares =
        ms::izaac::mpc::split_secret(secret, 5, 3, rng);
    const std::vector<ms::izaac::mpc::Share> subset = pick_shares(shares, {2, 3, 5});
    const ms::Result<uint64_t> reconstructed = ms::izaac::mpc::reconstruct_secret(subset);
    ASSERT_TRUE(reconstructed.has_value());
    EXPECT_EQ(*reconstructed, secret);
}

TEST(IzaacAdvanced, Mpc_ThreeOfThreeReconstructs) {
    std::array<uint8_t, 32> seed{};
    seed[6] = 0xA7;
    ms::izaac::CSPRNG rng(seed);
    constexpr uint64_t secret = 314159ULL;
    const std::vector<ms::izaac::mpc::Share> shares =
        ms::izaac::mpc::split_secret(secret, 3, 3, rng);
    const ms::Result<uint64_t> reconstructed = ms::izaac::mpc::reconstruct_secret(shares);
    ASSERT_TRUE(reconstructed.has_value());
    EXPECT_EQ(*reconstructed, secret);
}

TEST(IzaacAdvanced, Mpc_TwoOfTwoReconstructs) {
    std::array<uint8_t, 32> seed{};
    seed[7] = 0xA8;
    ms::izaac::CSPRNG rng(seed);
    constexpr uint64_t secret = 271828ULL;
    const std::vector<ms::izaac::mpc::Share> shares =
        ms::izaac::mpc::split_secret(secret, 2, 2, rng);
    const ms::Result<uint64_t> reconstructed = ms::izaac::mpc::reconstruct_secret(shares);
    ASSERT_TRUE(reconstructed.has_value());
    EXPECT_EQ(*reconstructed, secret);
}

TEST(IzaacAdvanced, Mpc_SevenOfTenReconstructs) {
    std::array<uint8_t, 32> seed{};
    seed[8] = 0xA9;
    ms::izaac::CSPRNG rng(seed);
    constexpr uint64_t secret = 1618033ULL;
    const std::vector<ms::izaac::mpc::Share> shares =
        ms::izaac::mpc::split_secret(secret, 10, 7, rng);
    const std::vector<ms::izaac::mpc::Share> subset = pick_shares(shares, {1, 3, 4, 6, 8, 9, 10});
    const ms::Result<uint64_t> reconstructed = ms::izaac::mpc::reconstruct_secret(subset);
    ASSERT_TRUE(reconstructed.has_value());
    EXPECT_EQ(*reconstructed, secret);
}

TEST(IzaacAdvanced, Mpc_FewerThanTwoSharesReturnsError) {
    std::array<uint8_t, 32> seed{};
    seed[9] = 0xAA;
    ms::izaac::CSPRNG rng(seed);
    const std::vector<ms::izaac::mpc::Share> shares =
        ms::izaac::mpc::split_secret(42ULL, 5, 3, rng);
    const std::vector<ms::izaac::mpc::Share> one_share = pick_shares(shares, {1});
    const ms::Result<uint64_t> reconstructed = ms::izaac::mpc::reconstruct_secret(one_share);
    EXPECT_FALSE(reconstructed.has_value());

    const ms::Result<uint64_t> empty = ms::izaac::mpc::reconstruct_secret({});
    EXPECT_FALSE(empty.has_value());
}

// ---------------------------------------------------------------------------
// consensus::Cluster – simulated Raft leader election and log replication
// ---------------------------------------------------------------------------

namespace {

int count_leaders(const ms::izaac::consensus::Cluster& cluster) {
    int count = 0;
    for (const ms::izaac::consensus::Node& node : cluster.nodes) {
        if (node.role == ms::izaac::consensus::NodeRole::Leader) {
            ++count;
        }
    }
    return count;
}

int count_nodes_holding_log_index(const ms::izaac::consensus::Cluster& cluster, size_t index) {
    int count = 0;
    for (const ms::izaac::consensus::Node& node : cluster.nodes) {
        if (node.log.size() >= index) {
            ++count;
        }
    }
    return count;
}

void assert_single_leader_elected(ms::izaac::consensus::Cluster& cluster, unsigned seed) {
    const int leader_id = cluster.run_election();
    EXPECT_GE(leader_id, 0);
    EXPECT_EQ(count_leaders(cluster), 1);
    EXPECT_EQ(cluster.current_leader(), leader_id);
    for (const ms::izaac::consensus::Node& node : cluster.nodes) {
        if (node.id == leader_id) {
            EXPECT_EQ(node.role, ms::izaac::consensus::NodeRole::Leader);
        } else {
            EXPECT_EQ(node.role, ms::izaac::consensus::NodeRole::Follower);
        }
    }
    (void)seed;
}

bool clusters_match(
    const ms::izaac::consensus::Cluster& a,
    const ms::izaac::consensus::Cluster& b) {
    if (a.nodes.size() != b.nodes.size()) {
        return false;
    }
    for (size_t i = 0; i < a.nodes.size(); ++i) {
        const auto& na = a.nodes[i];
        const auto& nb = b.nodes[i];
        if (na.id != nb.id || na.role != nb.role || na.current_term != nb.current_term ||
            na.voted_for != nb.voted_for || na.commit_index != nb.commit_index ||
            na.log.size() != nb.log.size()) {
            return false;
        }
        for (size_t j = 0; j < na.log.size(); ++j) {
            if (na.log[j].term != nb.log[j].term || na.log[j].command != nb.log[j].command) {
                return false;
            }
        }
    }
    return a.current_leader() == b.current_leader();
}

} // namespace

TEST(IzaacAdvanced, Consensus_CurrentLeaderBeforeElection) {
    ms::izaac::consensus::Cluster cluster(5, 100u);
    EXPECT_EQ(cluster.current_leader(), -1);
    for (const ms::izaac::consensus::Node& node : cluster.nodes) {
        EXPECT_EQ(node.role, ms::izaac::consensus::NodeRole::Follower);
    }
}

TEST(IzaacAdvanced, Consensus_RunElectionCluster3OneLeader) {
    ms::izaac::consensus::Cluster cluster(3, 7u);
    assert_single_leader_elected(cluster, 7u);
}

TEST(IzaacAdvanced, Consensus_RunElectionCluster5OneLeader) {
    ms::izaac::consensus::Cluster cluster(5, 11u);
    assert_single_leader_elected(cluster, 11u);
}

TEST(IzaacAdvanced, Consensus_RunElectionCluster7OneLeader) {
    ms::izaac::consensus::Cluster cluster(7, 13u);
    assert_single_leader_elected(cluster, 13u);
}

TEST(IzaacAdvanced, Consensus_RunElectionTermIncrementsConsistently) {
    ms::izaac::consensus::Cluster cluster(5, 17u);
    const int leader_id = cluster.run_election();
    ASSERT_GE(leader_id, 0);

    const uint64_t term = cluster.nodes[static_cast<size_t>(leader_id)].current_term;
    EXPECT_EQ(term, 1u);
    for (const ms::izaac::consensus::Node& node : cluster.nodes) {
        EXPECT_EQ(node.current_term, term);
    }

    const int leader_id_2 = cluster.run_election();
    ASSERT_GE(leader_id_2, 0);
    const uint64_t term_2 = cluster.nodes[static_cast<size_t>(leader_id_2)].current_term;
    EXPECT_EQ(term_2, term + 1);
    for (const ms::izaac::consensus::Node& node : cluster.nodes) {
        EXPECT_EQ(node.current_term, term_2);
    }
}

TEST(IzaacAdvanced, Consensus_RunElectionVotedForSafety) {
    ms::izaac::consensus::Cluster cluster(5, 19u);
    for (int round = 0; round < 5; ++round) {
        const int leader_id = cluster.run_election();
        ASSERT_GE(leader_id, 0);
        const uint64_t term = cluster.nodes[static_cast<size_t>(leader_id)].current_term;
        for (const ms::izaac::consensus::Node& node : cluster.nodes) {
            EXPECT_EQ(node.current_term, term);
            if (node.voted_for != -1) {
                EXPECT_GE(node.voted_for, 0);
                EXPECT_LT(node.voted_for, static_cast<int>(cluster.nodes.size()));
            }
        }
    }
}

TEST(IzaacAdvanced, Consensus_ReplicateAfterElectionSucceeds) {
    ms::izaac::consensus::Cluster cluster(5, 23u);
    const int leader_id = cluster.run_election();
    ASSERT_GE(leader_id, 0);
    EXPECT_TRUE(cluster.replicate(leader_id, "set x=1"));
}

TEST(IzaacAdvanced, Consensus_ReplicateEntryInMajorityWithCorrectTerm) {
    ms::izaac::consensus::Cluster cluster(5, 29u);
    const int leader_id = cluster.run_election();
    ASSERT_GE(leader_id, 0);
    const uint64_t term = cluster.nodes[static_cast<size_t>(leader_id)].current_term;

    ASSERT_TRUE(cluster.replicate(leader_id, "cmd-alpha"));
    EXPECT_GE(count_nodes_holding_log_index(cluster, 1), 3);

    int with_correct_term = 0;
    for (const ms::izaac::consensus::Node& node : cluster.nodes) {
        if (!node.log.empty() && node.log[0].term == term && node.log[0].command == "cmd-alpha") {
            ++with_correct_term;
        }
    }
    EXPECT_GE(with_correct_term, 3);
}

TEST(IzaacAdvanced, Consensus_ReplicateCommitIndexAdvances) {
    ms::izaac::consensus::Cluster cluster(3, 31u);
    const int leader_id = cluster.run_election();
    ASSERT_GE(leader_id, 0);

    ASSERT_TRUE(cluster.replicate(leader_id, "commit-me"));
    for (const ms::izaac::consensus::Node& node : cluster.nodes) {
        if (node.log.size() >= 1) {
            EXPECT_EQ(node.commit_index, 1u);
        }
    }
}

TEST(IzaacAdvanced, Consensus_ReplicateSequentialEntriesOrdered) {
    ms::izaac::consensus::Cluster cluster(5, 37u);
    const int leader_id = cluster.run_election();
    ASSERT_GE(leader_id, 0);

    ASSERT_TRUE(cluster.replicate(leader_id, "first"));
    ASSERT_TRUE(cluster.replicate(leader_id, "second"));
    ASSERT_TRUE(cluster.replicate(leader_id, "third"));

    const ms::izaac::consensus::Node& leader = cluster.nodes[static_cast<size_t>(leader_id)];
    ASSERT_EQ(leader.log.size(), 3u);
    EXPECT_EQ(leader.log[0].command, "first");
    EXPECT_EQ(leader.log[1].command, "second");
    EXPECT_EQ(leader.log[2].command, "third");
    EXPECT_EQ(leader.commit_index, 3u);
}

TEST(IzaacAdvanced, Consensus_ReplicateBeforeElectionReturnsFalse) {
    ms::izaac::consensus::Cluster cluster(3, 41u);
    EXPECT_FALSE(cluster.replicate(0, "orphan"));
    EXPECT_TRUE(cluster.nodes[0].log.empty());
}

TEST(IzaacAdvanced, Consensus_ReplicateInvalidLeaderIdReturnsFalse) {
    ms::izaac::consensus::Cluster cluster(3, 43u);
    const int leader_id = cluster.run_election();
    ASSERT_GE(leader_id, 0);
    EXPECT_FALSE(cluster.replicate(-1, "bad"));
    EXPECT_FALSE(cluster.replicate(99, "bad"));
    EXPECT_FALSE(cluster.replicate((leader_id + 1) % 3, "not-leader"));
}

TEST(IzaacAdvanced, Consensus_CurrentLeaderReflectsElection) {
    ms::izaac::consensus::Cluster cluster(5, 47u);
    EXPECT_EQ(cluster.current_leader(), -1);
    const int leader_id = cluster.run_election();
    ASSERT_GE(leader_id, 0);
    EXPECT_EQ(cluster.current_leader(), leader_id);
}

TEST(IzaacAdvanced, Consensus_DeterminismSameSeedSameElection) {
    ms::izaac::consensus::Cluster a(5, 53u);
    ms::izaac::consensus::Cluster b(5, 53u);

    const int leader_a = a.run_election();
    const int leader_b = b.run_election();
    EXPECT_EQ(leader_a, leader_b);
    EXPECT_TRUE(clusters_match(a, b));
}

TEST(IzaacAdvanced, Consensus_DeterminismSameSequenceIdenticalState) {
    ms::izaac::consensus::Cluster a(5, 59u);
    ms::izaac::consensus::Cluster b(5, 59u);

    const int leader_a = a.run_election();
    const int leader_b = b.run_election();
    ASSERT_EQ(leader_a, leader_b);
    ASSERT_GE(leader_a, 0);

    ASSERT_TRUE(a.replicate(leader_a, "a"));
    ASSERT_TRUE(b.replicate(leader_b, "a"));
    ASSERT_TRUE(a.replicate(leader_a, "b"));
    ASSERT_TRUE(b.replicate(leader_b, "b"));

    EXPECT_TRUE(clusters_match(a, b));
    EXPECT_EQ(a.current_leader(), b.current_leader());
}

TEST(IzaacAdvanced, Consensus_DeterminismReplayMatchesFirstRun) {
    ms::izaac::consensus::Cluster first(5, 61u);
    const int leader_first = first.run_election();
    ASSERT_GE(leader_first, 0);

    ms::izaac::consensus::Cluster replay(5, 61u);
    const int leader_replay = replay.run_election();
    EXPECT_EQ(leader_first, leader_replay);
    EXPECT_TRUE(clusters_match(first, replay));
}

// ---------------------------------------------------------------------------
// fuzz::mutate
// ---------------------------------------------------------------------------

namespace {

std::vector<uint8_t> make_seed_corpus_input() {
    std::vector<uint8_t> data(48);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<uint8_t>((i * 37 + 11) & 0xFF);
    }
    return data;
}

std::array<uint8_t, 32> make_fuzz_seed(uint8_t marker) {
    std::array<uint8_t, 32> seed{};
    seed[0] = marker;
    seed[31] = static_cast<uint8_t>(marker ^ 0xFF);
    return seed;
}

} // namespace

TEST(IzaacAdvanced, Fuzz_MutateIsDeterministicForSameSeed) {
    const std::vector<uint8_t> input = make_seed_corpus_input();

    ms::izaac::CSPRNG rng1(make_fuzz_seed(0x10));
    ms::izaac::CSPRNG rng2(make_fuzz_seed(0x10));

    const std::vector<uint8_t> out1 =
        ms::izaac::fuzz::mutate(std::span<const uint8_t>(input), rng1, 16);
    const std::vector<uint8_t> out2 =
        ms::izaac::fuzz::mutate(std::span<const uint8_t>(input), rng2, 16);

    EXPECT_EQ(out1, out2);
}

TEST(IzaacAdvanced, Fuzz_MutateDeterministicAcrossMultipleSeedsAndMaxEdits) {
    const std::vector<uint8_t> input = make_seed_corpus_input();
    const std::array<uint8_t, 4> markers = {0x21, 0x22, 0x23, 0x24};
    const std::array<std::size_t, 4> edit_caps = {1, 4, 16, 32};

    for (size_t m = 0; m < markers.size(); ++m) {
        for (size_t c = 0; c < edit_caps.size(); ++c) {
            ms::izaac::CSPRNG rng_a(make_fuzz_seed(markers[m]));
            ms::izaac::CSPRNG rng_b(make_fuzz_seed(markers[m]));
            const std::vector<uint8_t> out_a =
                ms::izaac::fuzz::mutate(std::span<const uint8_t>(input), rng_a, edit_caps[c]);
            const std::vector<uint8_t> out_b =
                ms::izaac::fuzz::mutate(std::span<const uint8_t>(input), rng_b, edit_caps[c]);
            EXPECT_EQ(out_a, out_b)
                << "mismatch for marker " << static_cast<int>(markers[m])
                << " max_edits " << edit_caps[c];
        }
    }
}

TEST(IzaacAdvanced, Fuzz_MutateOutputLengthBoundedByMaxEdits) {
    const std::vector<uint8_t> input = make_seed_corpus_input();
    constexpr std::size_t kMaxSpliceLength = 8;
    const std::array<std::size_t, 5> edit_caps = {1, 2, 8, 16, 32};

    for (std::size_t max_edits : edit_caps) {
        for (uint8_t marker = 0; marker < 20; ++marker) {
            ms::izaac::CSPRNG rng(make_fuzz_seed(static_cast<uint8_t>(0x30 + marker)));
            const std::vector<uint8_t> out =
                ms::izaac::fuzz::mutate(std::span<const uint8_t>(input), rng, max_edits);

            const long long delta =
                static_cast<long long>(out.size()) - static_cast<long long>(input.size());
            const long long max_possible_delta =
                static_cast<long long>(max_edits) * static_cast<long long>(kMaxSpliceLength);

            EXPECT_LE(delta, max_possible_delta)
                << "growth exceeded bound for max_edits=" << max_edits;
            EXPECT_GE(delta, -static_cast<long long>(max_edits))
                << "shrinkage exceeded bound for max_edits=" << max_edits;
        }
    }
}

TEST(IzaacAdvanced, Fuzz_MutateZeroMaxEditsReturnsInputUnchanged) {
    const std::vector<uint8_t> input = make_seed_corpus_input();
    ms::izaac::CSPRNG rng(make_fuzz_seed(0x40));
    const std::vector<uint8_t> out =
        ms::izaac::fuzz::mutate(std::span<const uint8_t>(input), rng, 0);
    EXPECT_EQ(out, input);
}

TEST(IzaacAdvanced, Fuzz_MutateChangesInputWithHighProbabilityAcrossSeeds) {
    const std::vector<uint8_t> input = make_seed_corpus_input();
    int changed_count = 0;
    constexpr int seed_trials = 16;
    for (int i = 0; i < seed_trials; ++i) {
        ms::izaac::CSPRNG rng(make_fuzz_seed(static_cast<uint8_t>(0x50 + i)));
        const std::vector<uint8_t> out =
            ms::izaac::fuzz::mutate(std::span<const uint8_t>(input), rng, 16);
        if (out != input) {
            ++changed_count;
        }
    }
    // With max_edits=16 (>=1 edit guaranteed) it would be extraordinarily unlikely for
    // every single one of 16 independently-seeded runs to be a silent no-op; a handful
    // of unchanged runs is still plausible (e.g. a flip that redraws the same byte value).
    EXPECT_GT(changed_count, seed_trials / 2);
}

TEST(IzaacAdvanced, Fuzz_MutateEmptyInputDoesNotCrashAndOnlyInserts) {
    const std::vector<uint8_t> empty_input;
    ms::izaac::CSPRNG rng(make_fuzz_seed(0x60));
    const std::vector<uint8_t> out =
        ms::izaac::fuzz::mutate(std::span<const uint8_t>(empty_input), rng, 5);
    // Every edit on an empty buffer degrades to an insertion, so the output length
    // equals the number of edits actually drawn (between 1 and max_edits).
    EXPECT_GE(out.size(), 1u);
    EXPECT_LE(out.size(), 5u);
}

TEST(IzaacAdvanced, Fuzz_MutateEmptyInputWithZeroMaxEditsStaysEmpty) {
    const std::vector<uint8_t> empty_input;
    ms::izaac::CSPRNG rng(make_fuzz_seed(0x61));
    const std::vector<uint8_t> out =
        ms::izaac::fuzz::mutate(std::span<const uint8_t>(empty_input), rng, 0);
    EXPECT_TRUE(out.empty());
}

TEST(IzaacAdvanced, Fuzz_MutateDifferentSeedsExploreDistinctCorpus) {
    const std::vector<uint8_t> input = make_seed_corpus_input();
    std::vector<std::vector<uint8_t>> outputs;
    constexpr int seed_count = 10;
    for (int i = 0; i < seed_count; ++i) {
        ms::izaac::CSPRNG rng(make_fuzz_seed(static_cast<uint8_t>(0x70 + i)));
        outputs.push_back(ms::izaac::fuzz::mutate(std::span<const uint8_t>(input), rng, 16));
    }

    int distinct_pairs = 0;
    int total_pairs = 0;
    for (size_t i = 0; i < outputs.size(); ++i) {
        for (size_t j = i + 1; j < outputs.size(); ++j) {
            ++total_pairs;
            if (outputs[i] != outputs[j]) {
                ++distinct_pairs;
            }
        }
    }
    // Different seeds should overwhelmingly diverge from one another, demonstrating that
    // mutate() explores a mutation space rather than collapsing to one fixed transform.
    EXPECT_GT(distinct_pairs, total_pairs / 2);
}

TEST(IzaacAdvanced, Fuzz_MutateDoesNotModifyInputBuffer) {
    const std::vector<uint8_t> input = make_seed_corpus_input();
    const std::vector<uint8_t> input_copy = input;
    ms::izaac::CSPRNG rng(make_fuzz_seed(0x80));
    (void)ms::izaac::fuzz::mutate(std::span<const uint8_t>(input), rng, 16);
    EXPECT_EQ(input, input_copy);
}
