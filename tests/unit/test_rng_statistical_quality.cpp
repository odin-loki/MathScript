#include <gtest/gtest.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

#include "ms/frameworks/izaac/izaac.hpp"
#include "ms/stats/stats.hpp"

namespace {

// Theoretical stddev of Uniform(0,1): 1/sqrt(12) ~ 0.2886751345948129.
constexpr double kUniformStddev = 0.2886751345948129;

// Chi-square critical value for df = (bins - 1) = 19 at alpha = 0.01.
constexpr double kChi2CriticalDf19Alpha001 = 36.191;

std::array<uint8_t, 32> make_seed_from_u64(uint64_t value) {
    std::array<uint8_t, 32> seed{};
    std::memcpy(seed.data(), &value, sizeof(value));
    for (size_t i = sizeof(value); i < seed.size(); ++i) {
        seed[i] = static_cast<uint8_t>((value >> (i % 8)) & 0xFF);
    }
    return seed;
}

std::vector<double> draw_uniform_f64(ms::izaac::CSPRNG& rng, size_t count) {
    std::vector<double> samples;
    samples.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        samples.push_back(rng.next_f64());
    }
    return samples;
}

std::vector<double> bin_uniform_samples(std::span<const double> samples, size_t bin_count) {
    std::vector<double> counts(bin_count, 0.0);
    for (double sample : samples) {
        const size_t bin = static_cast<size_t>(sample * static_cast<double>(bin_count));
        const size_t clamped = (bin >= bin_count) ? bin_count - 1 : bin;
        counts[clamped] += 1.0;
    }
    return counts;
}

int popcount64(uint64_t value) {
    int count = 0;
    while (value != 0) {
        count += static_cast<int>(value & 1u);
        value >>= 1;
    }
    return count;
}

} // namespace

// ---------------------------------------------------------------------------
// 1. Uniformity of CSPRNG::next_f64()
// ---------------------------------------------------------------------------

TEST(RngStatisticalQuality, UniformNextF64_ChiSquareGoodnessOfFit) {
    // 50,000 draws binned into 20 equal-width bins on [0, 1).
    // Expected count per bin = 50000 / 20 = 2500 under Uniform(0, 1).
    // Fail to reject uniformity at alpha = 0.01 (df = 19 critical value 36.191).
    constexpr size_t kSampleCount = 50000;
    constexpr size_t kBinCount = 20;
    constexpr double kExpectedPerBin =
        static_cast<double>(kSampleCount) / static_cast<double>(kBinCount);

    ms::izaac::CSPRNG rng(make_seed_from_u64(0x5200600101u));
    const std::vector<double> samples = draw_uniform_f64(rng, kSampleCount);
    const std::vector<double> observed = bin_uniform_samples(samples, kBinCount);
    const std::vector<double> expected(kBinCount, kExpectedPerBin);

    const double chi2 = ms::chi2_gof(observed, expected);
    EXPECT_LT(chi2, kChi2CriticalDf19Alpha001)
        << "chi2=" << chi2 << " rejects uniformity at alpha=0.01 (df=19)";
}

TEST(RngStatisticalQuality, UniformNextF64_MeanNearOneHalf) {
    constexpr size_t kSampleCount = 50000;
    // SE(mean) for Uniform(0,1) is sqrt(1/12)/sqrt(n) ~ 0.00129; 0.02 is generous.
    constexpr double kMeanTolerance = 0.02;

    ms::izaac::CSPRNG rng(make_seed_from_u64(0x5200600102u));
    const std::vector<double> samples = draw_uniform_f64(rng, kSampleCount);
    const double sample_mean = ms::mean(samples);
    EXPECT_NEAR(sample_mean, 0.5, kMeanTolerance)
        << "empirical mean=" << sample_mean;
}

TEST(RngStatisticalQuality, UniformNextF64_StddevNearUniformTheory) {
    constexpr size_t kSampleCount = 50000;
    // Theoretical stddev of Uniform(0,1) is 1/sqrt(12) ~ 0.288675.
    constexpr double kStddevTolerance = 0.02;

    ms::izaac::CSPRNG rng(make_seed_from_u64(0x5200600103u));
    const std::vector<double> samples = draw_uniform_f64(rng, kSampleCount);
    const double sample_stddev = ms::stddev(samples);
    EXPECT_NEAR(sample_stddev, kUniformStddev, kStddevTolerance)
        << "empirical stddev=" << sample_stddev
        << " theoretical=" << kUniformStddev;
}

// ---------------------------------------------------------------------------
// 2. Two-sample distributional consistency
// ---------------------------------------------------------------------------

TEST(RngStatisticalQuality, TwoSample_KsTestIndistinguishableAcrossSeeds) {
    // Independent 20,000-sample batches from differently-seeded CSPRNG instances
    // should look like draws from the same Uniform(0,1) distribution.
    constexpr size_t kSampleCount = 20000;

    ms::izaac::CSPRNG rng_a(make_seed_from_u64(0xA11CE001u));
    ms::izaac::CSPRNG rng_b(make_seed_from_u64(0xBEEFCAFEu));
    const std::vector<double> batch_a = draw_uniform_f64(rng_a, kSampleCount);
    const std::vector<double> batch_b = draw_uniform_f64(rng_b, kSampleCount);

    const ms::KSTestResult ks = ms::ks_test_2sample(batch_a, batch_b);
    EXPECT_GT(ks.p_value, 0.05)
        << "KS D=" << ks.d_stat << " p=" << ks.p_value
        << " (seed-dependent bias would yield a tiny p-value)";
}

TEST(RngStatisticalQuality, TwoSample_MannWhitneyIndistinguishableAcrossSeeds) {
    constexpr size_t kSampleCount = 20000;

    ms::izaac::CSPRNG rng_a(make_seed_from_u64(0xC0FFEE01u));
    ms::izaac::CSPRNG rng_b(make_seed_from_u64(0xDEADBEEFu));
    const std::vector<double> batch_a = draw_uniform_f64(rng_a, kSampleCount);
    const std::vector<double> batch_b = draw_uniform_f64(rng_b, kSampleCount);

    const ms::MannWhitneyResult mw = ms::mann_whitney_u(batch_a, batch_b);
    EXPECT_GT(mw.p_value, 0.05)
        << "Mann-Whitney U=" << mw.u_stat << " p=" << mw.p_value;
}

// ---------------------------------------------------------------------------
// 3. Independence / no short-period serial correlation
// ---------------------------------------------------------------------------

TEST(RngStatisticalQuality, SerialCorrelation_LagOneNearZero) {
    // Under independence, lag-1 Pearson r has approximate SE ~ 1/sqrt(n-3) (~0.007 at n=20000).
    // |r| < 0.05 is ~7 sigma under the null — generous but catches gross short-range correlation
    // from weak state mixing (the pre-fix bug mutated only 8/32 state bytes per step).
    constexpr size_t kSampleCount = 20000;
    constexpr double kMaxAbsCorrelation = 0.05;

    ms::izaac::CSPRNG rng(make_seed_from_u64(0xC0E1E0001u));
    std::vector<double> current;
    std::vector<double> lagged;
    current.reserve(kSampleCount - 1);
    lagged.reserve(kSampleCount - 1);

    double previous = rng.next_f64();
    for (size_t i = 1; i < kSampleCount; ++i) {
        const double value = rng.next_f64();
        current.push_back(value);
        lagged.push_back(previous);
        previous = value;
    }

    const double lag1_corr = ms::correlation(current, lagged);
    EXPECT_LT(std::abs(lag1_corr), kMaxAbsCorrelation)
        << "lag-1 correlation=" << lag1_corr;
}

// ---------------------------------------------------------------------------
// 4. Seed sensitivity / avalanche property
// ---------------------------------------------------------------------------

TEST(RngStatisticalQuality, Avalanche_AdjacentIntegerSeedsDiffer) {
    ms::izaac::CSPRNG rng_a(make_seed_from_u64(100u));
    ms::izaac::CSPRNG rng_b(make_seed_from_u64(101u));

    std::vector<uint64_t> outputs_a;
    std::vector<uint64_t> outputs_b;
    outputs_a.reserve(10);
    outputs_b.reserve(10);
    for (int i = 0; i < 10; ++i) {
        outputs_a.push_back(rng_a.next_u64());
        outputs_b.push_back(rng_b.next_u64());
    }

    EXPECT_NE(outputs_a, outputs_b);
    int identical_positions = 0;
    for (size_t i = 0; i < outputs_a.size(); ++i) {
        if (outputs_a[i] == outputs_b[i]) {
            ++identical_positions;
        }
    }
    EXPECT_LT(identical_positions, 10);
}

TEST(RngStatisticalQuality, Avalanche_NotConstantOffsetBetweenAdjacentSeeds) {
    ms::izaac::CSPRNG rng_a(make_seed_from_u64(100u));
    ms::izaac::CSPRNG rng_b(make_seed_from_u64(101u));

    const uint64_t first_xor = rng_a.next_u64() ^ rng_b.next_u64();
    bool all_same_xor = true;
    for (int i = 1; i < 10; ++i) {
        const uint64_t xor_diff = rng_a.next_u64() ^ rng_b.next_u64();
        if (xor_diff != first_xor) {
            all_same_xor = false;
            break;
        }
    }
    EXPECT_FALSE(all_same_xor) << "outputs differ by a fixed XOR mask across draws";
}

TEST(RngStatisticalQuality, Avalanche_SingleBitSeedFlipDiffuses) {
    std::array<uint8_t, 32> seed_a{};
    seed_a.fill(0x5A);
    std::array<uint8_t, 32> seed_b = seed_a;
    seed_b[7] ^= 0x01; // flip one bit in the 32-byte seed

    ms::izaac::CSPRNG rng_a(seed_a);
    ms::izaac::CSPRNG rng_b(seed_b);

    int total_hamming = 0;
    for (int i = 0; i < 10; ++i) {
        const uint64_t a = rng_a.next_u64();
        const uint64_t b = rng_b.next_u64();
        EXPECT_NE(a, b) << "identical u64 at position " << i;
        total_hamming += popcount64(a ^ b);
    }
    // Avalanche: a 1-bit seed change should flip many output bits on average.
    EXPECT_GT(total_hamming, 64);
}

// ---------------------------------------------------------------------------
// 5. mc::estimate_pi accuracy at scale
// ---------------------------------------------------------------------------

TEST(RngStatisticalQuality, EstimatePi_Seed100WithinTightTolerance) {
    ms::izaac::CSPRNG rng(make_seed_from_u64(100u));
    const double pi_est = ms::izaac::mc::estimate_pi(200000, rng);
    EXPECT_NEAR(pi_est, 3.14159265358979, 0.02) << "pi_est=" << pi_est;
}

TEST(RngStatisticalQuality, EstimatePi_Seed101WithinTightTolerance) {
    ms::izaac::CSPRNG rng(make_seed_from_u64(101u));
    const double pi_est = ms::izaac::mc::estimate_pi(200000, rng);
    EXPECT_NEAR(pi_est, 3.14159265358979, 0.02) << "pi_est=" << pi_est;
}

TEST(RngStatisticalQuality, EstimatePi_Seed202WithinTightTolerance) {
    ms::izaac::CSPRNG rng(make_seed_from_u64(202u));
    const double pi_est = ms::izaac::mc::estimate_pi(200000, rng);
    EXPECT_NEAR(pi_est, 3.14159265358979, 0.02) << "pi_est=" << pi_est;
}

TEST(RngStatisticalQuality, EstimatePi_Seed303WithinTightTolerance) {
    ms::izaac::CSPRNG rng(make_seed_from_u64(303u));
    const double pi_est = ms::izaac::mc::estimate_pi(200000, rng);
    EXPECT_NEAR(pi_est, 3.14159265358979, 0.02) << "pi_est=" << pi_est;
}

TEST(RngStatisticalQuality, EstimatePi_Seed404WithinTightTolerance) {
    ms::izaac::CSPRNG rng(make_seed_from_u64(404u));
    const double pi_est = ms::izaac::mc::estimate_pi(200000, rng);
    EXPECT_NEAR(pi_est, 3.14159265358979, 0.02) << "pi_est=" << pi_est;
}

// ---------------------------------------------------------------------------
// 6. diffpriv::laplace_mechanism / gaussian_mechanism distributional sanity
// ---------------------------------------------------------------------------

TEST(RngStatisticalQuality, DiffPriv_LaplaceMeanNearTrueValue) {
    constexpr double epsilon = 2.0;
    constexpr double sensitivity = 1.0;
    constexpr int kSampleCount = 5000;
    // Laplace noise is centered at 0 when true_value = 0.
    constexpr double kMeanTolerance = 0.15;

    ms::izaac::CSPRNG rng(make_seed_from_u64(0x1AFACE01u));
    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(kSampleCount));
    for (int i = 0; i < kSampleCount; ++i) {
        samples.push_back(
            ms::izaac::diffpriv::laplace_mechanism(0.0, epsilon, sensitivity, rng));
    }

    const double sample_mean = ms::mean(samples);
    EXPECT_NEAR(sample_mean, 0.0, kMeanTolerance) << "empirical mean=" << sample_mean;
}

TEST(RngStatisticalQuality, DiffPriv_LaplaceVarianceNearTheory) {
    constexpr double epsilon = 1.0;
    constexpr double sensitivity = 2.0;
    constexpr int kSampleCount = 5000;
    const double scale = sensitivity / epsilon;
    const double expected_variance = 2.0 * scale * scale;

    ms::izaac::CSPRNG rng(make_seed_from_u64(0x1AFACE02u));
    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(kSampleCount));
    for (int i = 0; i < kSampleCount; ++i) {
        samples.push_back(
            ms::izaac::diffpriv::laplace_mechanism(0.0, epsilon, sensitivity, rng));
    }

    const double sample_variance = ms::var(samples);
    EXPECT_NEAR(sample_variance, expected_variance, expected_variance * 0.35)
        << "empirical var=" << sample_variance << " theory=" << expected_variance;
}

TEST(RngStatisticalQuality, DiffPriv_GaussianMeanNearTrueValue) {
    constexpr double true_value = 0.0;
    constexpr double epsilon = 1.0;
    constexpr double delta = 1e-6;
    constexpr double sensitivity = 1.0;
    constexpr int kSampleCount = 5000;
    constexpr double kMeanTolerance = 0.15;

    ms::izaac::CSPRNG rng(make_seed_from_u64(0x6AA55001u));
    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(kSampleCount));
    for (int i = 0; i < kSampleCount; ++i) {
        samples.push_back(ms::izaac::diffpriv::gaussian_mechanism(
            true_value, epsilon, delta, sensitivity, rng));
    }

    const double sample_mean = ms::mean(samples);
    EXPECT_NEAR(sample_mean, true_value, kMeanTolerance) << "empirical mean=" << sample_mean;
}

TEST(RngStatisticalQuality, DiffPriv_GaussianVarianceNearTheory) {
    constexpr double epsilon = 1.0;
    constexpr double delta = 1e-6;
    constexpr double sensitivity = 1.0;
    constexpr int kSampleCount = 5000;
    const double sigma =
        std::sqrt(2.0 * std::log(1.25 / delta)) * sensitivity / epsilon;
    const double expected_variance = sigma * sigma;

    ms::izaac::CSPRNG rng(make_seed_from_u64(0x6AA55002u));
    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(kSampleCount));
    for (int i = 0; i < kSampleCount; ++i) {
        samples.push_back(
            ms::izaac::diffpriv::gaussian_mechanism(0.0, epsilon, delta, sensitivity, rng));
    }

    const double sample_variance = ms::var(samples);
    EXPECT_NEAR(sample_variance, expected_variance, expected_variance * 0.35)
        << "empirical var=" << sample_variance << " theory=" << expected_variance;
}

// ---------------------------------------------------------------------------
// 7. randn_matrix distributional sanity
// ---------------------------------------------------------------------------

TEST(RngStatisticalQuality, RandnMatrix_MeanNearZero) {
    ms::izaac::seed_session(0xBA4D0001u);
    const ms::Matrix<double> matrix = ms::izaac::randn_matrix(1000, 1000);

    std::vector<double> flat;
    flat.reserve(matrix.rows() * matrix.cols());
    for (size_t r = 0; r < matrix.rows(); ++r) {
        for (size_t c = 0; c < matrix.cols(); ++c) {
            flat.push_back(matrix(r, c));
        }
    }

    const double sample_mean = ms::mean(flat);
    EXPECT_NEAR(sample_mean, 0.0, 0.02) << "empirical mean=" << sample_mean;
    ms::izaac::clear_session();
}

TEST(RngStatisticalQuality, RandnMatrix_StddevNearOne) {
    ms::izaac::seed_session(0xBA4D0002u);
    const ms::Matrix<double> matrix = ms::izaac::randn_matrix(1000, 1000);

    std::vector<double> flat;
    flat.reserve(matrix.rows() * matrix.cols());
    for (size_t r = 0; r < matrix.rows(); ++r) {
        for (size_t c = 0; c < matrix.cols(); ++c) {
            flat.push_back(matrix(r, c));
        }
    }

    const double sample_stddev = ms::stddev(flat);
    EXPECT_NEAR(sample_stddev, 1.0, 0.02) << "empirical stddev=" << sample_stddev;
    ms::izaac::clear_session();
}
