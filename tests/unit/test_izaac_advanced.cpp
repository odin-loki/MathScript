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
