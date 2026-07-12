#define _USE_MATH_DEFINES
#include "ms/info/info.hpp"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <random>
#include <vector>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_E
#define M_E 2.71828182845904523536
#endif

using namespace ms::info;

TEST(InfoEntropy, UniformDistribution) {
    // Uniform over 4 outcomes: H = 2 bits
    std::vector<double> p = {0.25, 0.25, 0.25, 0.25};
    EXPECT_NEAR(entropy(p, 2.0), 2.0, 1e-10);
}

TEST(InfoEntropy, CertainDistribution) {
    std::vector<double> p = {1.0, 0.0, 0.0};
    EXPECT_NEAR(entropy(p, 2.0), 0.0, 1e-10);
}

TEST(InfoEntropy, BinaryEntropy) {
    // H(0.5, 0.5) = 1 bit
    std::vector<double> p = {0.5, 0.5};
    EXPECT_NEAR(entropy(p, 2.0), 1.0, 1e-10);
}

TEST(InfoEntropy, CrossEntropy) {
    std::vector<double> p = {0.5, 0.5};
    std::vector<double> q = {0.5, 0.5};
    EXPECT_NEAR(cross_entropy(p, q, 2.0), 1.0, 1e-10);
}

TEST(InfoDivergence, KLDivergenceSelf) {
    std::vector<double> p = {0.3, 0.4, 0.3};
    EXPECT_NEAR(kl_divergence(p, p, 2.0), 0.0, 1e-10);
}

TEST(InfoDivergence, KLDivergenceAsymmetric) {
    std::vector<double> p = {0.4, 0.6};
    std::vector<double> q = {0.5, 0.5};
    double kl_pq = kl_divergence(p, q, 2.0);
    double kl_qp = kl_divergence(q, p, 2.0);
    EXPECT_GT(kl_pq, 0.0);
    EXPECT_NE(kl_pq, kl_qp);  // asymmetric
}

TEST(InfoDivergence, JSDivergenceSymmetric) {
    std::vector<double> p = {0.7, 0.3};
    std::vector<double> q = {0.3, 0.7};
    EXPECT_NEAR(js_divergence(p, q, 2.0), js_divergence(q, p, 2.0), 1e-10);
    EXPECT_LE(js_divergence(p, q, 2.0), 1.0);
    EXPECT_GE(js_divergence(p, q, 2.0), 0.0);
}

TEST(InfoDivergence, HellingerDistance) {
    std::vector<double> p = {1.0, 0.0};
    std::vector<double> q = {0.0, 1.0};
    EXPECT_NEAR(hellinger_dist(p, q), 1.0, 1e-10);  // max distance
    EXPECT_NEAR(hellinger_dist(p, p), 0.0, 1e-10);
}

TEST(InfoDivergence, TVDistance) {
    std::vector<double> p = {0.5, 0.5};
    std::vector<double> q = {0.5, 0.5};
    EXPECT_NEAR(tv_distance(p, q), 0.0, 1e-10);
    std::vector<double> r = {1.0, 0.0};
    EXPECT_NEAR(tv_distance(p, r), 0.5, 1e-10);
}

TEST(InfoChannel, BSCCapacity) {
    EXPECT_NEAR(channel_capacity_bsc(0.0), 1.0, 1e-10);  // no error = 1 bit
    EXPECT_NEAR(channel_capacity_bsc(0.5), 0.0, 1e-10);  // max error = 0 bits
    EXPECT_GT(channel_capacity_bsc(0.1), 0.5);
}

TEST(InfoChannel, BECCapacity) {
    EXPECT_NEAR(channel_capacity_bec(0.0), 1.0, 1e-10);
    EXPECT_NEAR(channel_capacity_bec(0.5), 0.5, 1e-10);
}

TEST(InfoChannel, ShannonHartley) {
    double C = shannon_hartley(1e6, 1.0);  // 1MHz, SNR=1 => 1Mbps
    EXPECT_NEAR(C, 1e6, 1.0);  // C = B*log2(2) = B
}

TEST(InfoEntropy, Renyi) {
    std::vector<double> p = {0.25, 0.25, 0.25, 0.25};
    // Renyi with alpha=1 should equal Shannon
    EXPECT_NEAR(renyi_entropy(p, 1.0, 2.0), entropy(p, 2.0), 1e-6);
    // Renyi alpha=2 (collision entropy)
    double r2 = renyi_entropy(p, 2.0, 2.0);
    EXPECT_NEAR(r2, 2.0, 1e-10);  // uniform: H_2 = log2(n) = 2
}

TEST(InfoEntropy, Redundancy) {
    std::vector<double> p = {0.25, 0.25, 0.25, 0.25};
    EXPECT_NEAR(redundancy(p), 0.0, 1e-10);  // uniform = max entropy, no redundancy
    std::vector<double> q = {0.9, 0.1};
    EXPECT_GT(redundancy(q), 0.0);
}

TEST(InfoEntropy, Efficiency) {
    std::vector<double> p = {0.5, 0.5};
    EXPECT_NEAR(efficiency(p), 1.0, 1e-10);  // max efficiency

    std::vector<double> q = {0.9, 0.1};
    EXPECT_LT(efficiency(q), 1.0);
    EXPECT_GT(efficiency(q), 0.0);
}

TEST(InfoDifferential, Gaussian) {
    double h = differential_entropy_gaussian(1.0);
    // h = 0.5*ln(2πe) ≈ 1.4189
    EXPECT_NEAR(h, 0.5 * std::log(2.0 * M_PI * M_E), 1e-10);
}

TEST(InfoJointEntropy, Uniform2x2) {
    std::vector<double> joint = {0.25, 0.25, 0.25, 0.25};
    EXPECT_NEAR(joint_entropy(joint, 2, 2, 2.0), 2.0, 1e-10);
}

TEST(InfoConditionalEntropy, Uniform2x2) {
    std::vector<double> joint = {0.25, 0.25, 0.25, 0.25};
    EXPECT_NEAR(conditional_entropy(joint, 2, 2, 2.0), 1.0, 1e-10);
}

TEST(InfoSampleEntropy, MonotoneSeries) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    const double se = sample_entropy(x, 2, 0.5);
    EXPECT_NEAR(se, 0.0, 1e-10);
    EXPECT_TRUE(std::isfinite(se));
}

TEST(InfoMutualInfo, IndependentAndPerfect) {
    // Independent: I(X;Y) = 0
    std::vector<double> indep = {0.25, 0.25, 0.25, 0.25};
    EXPECT_NEAR(mutual_info(indep, 2, 2, 2.0), 0.0, 1e-10);
    // Perfect dependence on diagonal: I(X;Y) = H(X) = 1 bit
    std::vector<double> perfect = {0.5, 0.0, 0.0, 0.5};
    EXPECT_NEAR(mutual_info(perfect, 2, 2, 2.0), 1.0, 1e-10);
}

TEST(InfoKLDivergence, BernoulliExact) {
    std::vector<double> p = {0.8, 0.2};
    std::vector<double> q = {0.5, 0.5};
    double kl = kl_divergence(p, q, 2.0);
    double expected = 0.8 * std::log2(0.8 / 0.5) + 0.2 * std::log2(0.2 / 0.5);
    EXPECT_NEAR(kl, expected, 1e-10);
    EXPECT_GT(kl, 0.0);
}

TEST(InfoJSDivergence, DisjointSupport) {
    std::vector<double> p = {1.0, 0.0};
    std::vector<double> q = {0.0, 1.0};
    EXPECT_NEAR(js_divergence(p, q, 2.0), 1.0, 1e-10);
}

TEST(InfoShannonHartley, HighSNR) {
    double C = shannon_hartley(1e6, 9.0);  // SNR = 10 linear
    EXPECT_NEAR(C, 1e6 * std::log2(10.0), 1.0);
}

TEST(InfoChannelCapacity, LowErrorBSC) {
    double p = 0.01;
    double h = -p * std::log2(p) - (1.0 - p) * std::log2(1.0 - p);
    EXPECT_NEAR(channel_capacity_bsc(p), 1.0 - h, 1e-10);
}

// --- Blahut-Arimoto: general discrete memoryless channel capacity ---

TEST(BlahutArimoto, MatchesBSCClosedForm) {
    for (double p : {0.01, 0.1, 0.2, 0.3, 0.5}) {
        std::vector<std::vector<double>> W = {{1.0 - p, p}, {p, 1.0 - p}};
        double c_ba = blahut_arimoto(W);
        double c_closed = channel_capacity_bsc(p);
        EXPECT_NEAR(c_ba, c_closed, 1e-4) << "p_error=" << p;
    }
}

TEST(BlahutArimoto, MatchesBECClosedForm) {
    for (double eps : {0.0, 0.1, 0.25, 0.5, 0.9}) {
        // Binary erasure channel: 3 outputs {0, 1, erasure}.
        std::vector<std::vector<double>> W = {
            {1.0 - eps, 0.0, eps},
            {0.0, 1.0 - eps, eps},
        };
        double c_ba = blahut_arimoto(W);
        double c_closed = channel_capacity_bec(eps);
        EXPECT_NEAR(c_ba, c_closed, 1e-4) << "epsilon=" << eps;
    }
}

TEST(BlahutArimoto, NoiselessChannelIdentity2x2) {
    std::vector<std::vector<double>> W = {{1.0, 0.0}, {0.0, 1.0}};
    EXPECT_NEAR(blahut_arimoto(W), std::log2(2.0), 1e-6);
}

TEST(BlahutArimoto, NoiselessChannelIdentity3x3) {
    std::vector<std::vector<double>> W = {
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
    };
    EXPECT_NEAR(blahut_arimoto(W), std::log2(3.0), 1e-6);
}

TEST(BlahutArimoto, UselessChannelIdenticalRows) {
    std::vector<std::vector<double>> W = {{0.5, 0.5}, {0.5, 0.5}};
    EXPECT_NEAR(blahut_arimoto(W), 0.0, 1e-6);

    std::vector<std::vector<double>> W3 = {
        {0.2, 0.3, 0.5},
        {0.2, 0.3, 0.5},
        {0.2, 0.3, 0.5},
    };
    EXPECT_NEAR(blahut_arimoto(W3), 0.0, 1e-6);
}

TEST(BlahutArimoto, NonNegativeAndUpperBounded) {
    std::vector<std::vector<std::vector<double>>> matrices = {
        {{0.9, 0.1}, {0.2, 0.8}},
        {{0.7, 0.2, 0.1}, {0.1, 0.2, 0.7}, {0.3, 0.4, 0.3}},
        {{1.0, 0.0, 0.0, 0.0}, {0.25, 0.25, 0.25, 0.25}, {0.0, 0.0, 0.5, 0.5}},
        {{0.6, 0.4}, {0.4, 0.6}, {0.5, 0.5}},
    };
    for (const auto& W : matrices) {
        double c = blahut_arimoto(W);
        double upper_bound = std::log2(static_cast<double>(W[0].size()));
        EXPECT_GE(c, 0.0);
        EXPECT_LE(c, upper_bound + 1e-9);
    }
}

TEST(BlahutArimoto, ReturnsOptimalInputDistribution) {
    // Symmetric BSC-like channel: uniform input is optimal.
    std::vector<std::vector<double>> W = {{0.8, 0.2}, {0.2, 0.8}};
    ChannelCapacityResult res = channel_capacity(W);
    ASSERT_EQ(res.input_distribution.size(), 2u);
    EXPECT_NEAR(res.input_distribution[0], 0.5, 1e-4);
    EXPECT_NEAR(res.input_distribution[1], 0.5, 1e-4);
    EXPECT_NEAR(res.capacity, channel_capacity_bsc(0.2), 1e-4);

    // Asymmetric channel: skewed erasure-like channel where input 1 is
    // "cleaner" should not favor a uniform distribution symmetrically; just
    // sanity check the returned distribution is a valid probability vector.
    std::vector<std::vector<double>> W2 = {{0.9, 0.1}, {0.5, 0.5}};
    ChannelCapacityResult res2 = channel_capacity(W2);
    ASSERT_EQ(res2.input_distribution.size(), 2u);
    double sum = res2.input_distribution[0] + res2.input_distribution[1];
    EXPECT_NEAR(sum, 1.0, 1e-6);
    for (double px : res2.input_distribution) {
        EXPECT_GE(px, 0.0);
        EXPECT_LE(px, 1.0);
    }
    EXPECT_GE(res2.capacity, 0.0);
}

TEST(BlahutArimoto, DegenerateEmptyMatrix) {
    std::vector<std::vector<double>> W;
    EXPECT_NEAR(blahut_arimoto(W), 0.0, 1e-12);
    ChannelCapacityResult res = channel_capacity(W);
    EXPECT_NEAR(res.capacity, 0.0, 1e-12);
    EXPECT_TRUE(res.input_distribution.empty());
}

TEST(BlahutArimoto, DegenerateSingleInputSymbol) {
    std::vector<std::vector<double>> W = {{0.3, 0.7}};
    EXPECT_NEAR(blahut_arimoto(W), 0.0, 1e-12);
    ChannelCapacityResult res = channel_capacity(W);
    EXPECT_NEAR(res.capacity, 0.0, 1e-12);
    ASSERT_EQ(res.input_distribution.size(), 1u);
    EXPECT_NEAR(res.input_distribution[0], 1.0, 1e-12);
}

TEST(BlahutArimoto, DegenerateMismatchedRowLengths) {
    std::vector<std::vector<double>> W = {{0.5, 0.5}, {1.0, 0.0, 0.0}};
    EXPECT_NEAR(blahut_arimoto(W), 0.0, 1e-12);
}

TEST(BlahutArimoto, DegenerateRowsNotSummingToOne) {
    std::vector<std::vector<double>> W = {{0.5, 0.2}, {0.3, 0.3}};
    EXPECT_NEAR(blahut_arimoto(W), 0.0, 1e-12);
}

// --- Permutation entropy (Bandt-Pompe ordinal pattern complexity) ---
//
// Encoding convention used by ms::info::permutation_entropy: for a window
// starting at `start`, pattern[k] is the LOCAL window position (0..order-1)
// holding the k-th smallest value ("position-of-rank" encoding). E.g. for
// window values (5, 2, 8): smallest is position 1, next is position 0,
// largest is position 2, so the pattern is (1, 0, 2).

TEST(PermutationEntropy, MonotonicIncreasingIsZero) {
    // Every window is already sorted ascending -> exactly one ordinal
    // pattern occurs -> zero disorder in the pattern distribution.
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    EXPECT_NEAR(permutation_entropy(x, 3), 0.0, 1e-10);
    EXPECT_NEAR(permutation_entropy(x, 3, 1, false), 0.0, 1e-10);
}

TEST(PermutationEntropy, MonotonicDecreasingIsZero) {
    // Every window is sorted descending -> also exactly one ordinal
    // pattern (just a different one from the ascending case).
    std::vector<double> x = {10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
    EXPECT_NEAR(permutation_entropy(x, 3), 0.0, 1e-10);
    EXPECT_NEAR(permutation_entropy(x, 4, 1, false), 0.0, 1e-10);
}

TEST(PermutationEntropy, HandComputedExactEntropy) {
    // x = {1, 3, 2, 5, 4, 6}, order=3, delay=1 -> 4 windows:
    //   [1,3,2] -> sorted asc: 1(pos0),2(pos2),3(pos1) -> pattern (0,2,1)
    //   [3,2,5] -> sorted asc: 2(pos1),3(pos0),5(pos2) -> pattern (1,0,2)
    //   [2,5,4] -> sorted asc: 2(pos0),4(pos2),5(pos1) -> pattern (0,2,1)
    //   [5,4,6] -> sorted asc: 4(pos1),5(pos0),6(pos2) -> pattern (1,0,2)
    // Frequencies: (0,2,1) x2, (1,0,2) x2 -> probs {0.5, 0.5} -> H = 1 bit.
    std::vector<double> x = {1.0, 3.0, 2.0, 5.0, 4.0, 6.0};
    const double raw = permutation_entropy(x, 3, 1, false);
    EXPECT_NEAR(raw, 1.0, 1e-10);
}

TEST(PermutationEntropy, HandComputedNormalizedEntropy) {
    // Same series as above: normalized = H / log2(3!) = 1 / log2(6).
    std::vector<double> x = {1.0, 3.0, 2.0, 5.0, 4.0, 6.0};
    const double normalized = permutation_entropy(x, 3, 1, true);
    EXPECT_NEAR(normalized, 1.0 / std::log2(6.0), 1e-10);
}

TEST(PermutationEntropy, NormalizeFalseReturnsRawEntropy) {
    std::vector<double> x = {1.0, 3.0, 2.0, 5.0, 4.0, 6.0};
    EXPECT_NEAR(permutation_entropy(x, 3, 1, false), 1.0, 1e-10);
}

TEST(PermutationEntropy, NormalizeRatioMatchesLog2Factorial) {
    // The normalize=true result must equal normalize=false result divided
    // by log2(order!), using the exact same base-2 convention entropy()
    // uses internally.
    std::vector<double> x = {5.0, -4.0, 7.0, -2.0, 9.0, 0.0, 11.0, 2.0,
                              13.0, 4.0, 15.0, 6.0, 17.0, 8.0, 19.0, 10.0};
    const int order = 3;
    const double raw = permutation_entropy(x, order, 1, false);
    const double normalized = permutation_entropy(x, order, 1, true);
    double log2_factorial = 0.0;
    for (int i = 2; i <= order; ++i) log2_factorial += std::log2(static_cast<double>(i));
    EXPECT_NEAR(normalized, raw / log2_factorial, 1e-10);
}

TEST(PermutationEntropy, MaximalDiversityOrder2ApproachesOne) {
    // order=2 has only 2! = 2 possible patterns: ascending or descending.
    // A strictly alternating sequence hits each pattern equally often
    // (4 ascending + 4 descending windows out of 8), giving the exact
    // theoretical maximum entropy for order=2: normalized == 1.0.
    std::vector<double> x = {1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0};
    const double normalized = permutation_entropy(x, 2);
    EXPECT_NEAR(normalized, 1.0, 1e-10);

    // Monotonic series (order=2) has zero diversity by contrast.
    std::vector<double> mono = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    EXPECT_LT(permutation_entropy(mono, 2), normalized);
}

TEST(PermutationEntropy, DelayOneRevealsOscillation) {
    // x[i] = i + 5*(-1)^i: an increasing trend with a superimposed
    // zigzag. With delay=1, consecutive triples alternate between exactly
    // two ordinal patterns, 7 windows each out of 14 -> exact 1 bit raw
    // entropy, normalized = 1/log2(6).
    std::vector<double> x = {5.0, -4.0, 7.0, -2.0, 9.0, 0.0, 11.0, 2.0,
                              13.0, 4.0, 15.0, 6.0, 17.0, 8.0, 19.0, 10.0};
    const double raw = permutation_entropy(x, 3, 1, false);
    EXPECT_NEAR(raw, 1.0, 1e-10);
    const double normalized = permutation_entropy(x, 3, 1, true);
    EXPECT_NEAR(normalized, 1.0 / std::log2(6.0), 1e-10);
}

TEST(PermutationEntropy, DelayTwoLooksMonotonic) {
    // Same zigzag series as above, but sampled with delay=2: every window
    // then only ever sees same-parity elements, whose +5/-5 zigzag offset
    // cancels out, leaving a purely (deceptively) monotonic increasing
    // sub-sample -> exactly one ordinal pattern -> zero entropy, despite
    // the raw series being far from monotonic.
    std::vector<double> x = {5.0, -4.0, 7.0, -2.0, 9.0, 0.0, 11.0, 2.0,
                              13.0, 4.0, 15.0, 6.0, 17.0, 8.0, 19.0, 10.0};
    EXPECT_NEAR(permutation_entropy(x, 3, 2, false), 0.0, 1e-10);
    EXPECT_NEAR(permutation_entropy(x, 3, 2, true), 0.0, 1e-10);
}

TEST(PermutationEntropy, DelayChangesEntropyComparison) {
    // Direct demonstration that changing `delay` alone (same series, same
    // order) genuinely changes which samples are grouped into windows and
    // thus changes the resulting entropy.
    std::vector<double> x = {5.0, -4.0, 7.0, -2.0, 9.0, 0.0, 11.0, 2.0,
                              13.0, 4.0, 15.0, 6.0, 17.0, 8.0, 19.0, 10.0};
    const double h_delay1 = permutation_entropy(x, 3, 1);
    const double h_delay2 = permutation_entropy(x, 3, 2);
    EXPECT_GT(h_delay1, h_delay2);
    EXPECT_NEAR(h_delay2, 0.0, 1e-10);
}

TEST(PermutationEntropy, DegenerateOrderLessThanTwo) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    EXPECT_NEAR(permutation_entropy(x, 0), 0.0, 1e-12);
    EXPECT_NEAR(permutation_entropy(x, 1), 0.0, 1e-12);
}

TEST(PermutationEntropy, DegenerateNegativeOrder) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    EXPECT_NEAR(permutation_entropy(x, -3), 0.0, 1e-12);
}

TEST(PermutationEntropy, DegenerateSeriesTooShortForOrder) {
    // order=5 needs at least 5 samples (delay=1) to form a single window.
    std::vector<double> x = {1.0, 2.0, 3.0};
    EXPECT_NEAR(permutation_entropy(x, 5), 0.0, 1e-12);
}

TEST(PermutationEntropy, DegenerateEmptySeries) {
    std::vector<double> x;
    EXPECT_NEAR(permutation_entropy(x, 3), 0.0, 1e-12);
}

TEST(PermutationEntropy, DegenerateNonPositiveDelay) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    EXPECT_NEAR(permutation_entropy(x, 3, 0), 0.0, 1e-12);
    EXPECT_NEAR(permutation_entropy(x, 3, -1), 0.0, 1e-12);
}

TEST(PermutationEntropy, NoiseRobustnessSmallNoiseStaysLow) {
    // A clean monotonic ramp has zero permutation entropy. Perturbing it
    // with noise much smaller than the step size between samples should
    // not flip most local orderings, so the noisy version's entropy
    // should stay close to (not wildly different from) the clean value.
    std::vector<double> clean(30);
    for (size_t i = 0; i < clean.size(); ++i) clean[i] = static_cast<double>(i);

    // Deterministic small perturbation pattern (amplitude << step size 1.0).
    const double small_noise[6] = {0.02, -0.03, 0.01, -0.01, 0.025, -0.015};
    std::vector<double> noisy = clean;
    for (size_t i = 0; i < noisy.size(); ++i) noisy[i] += small_noise[i % 6];

    const double h_clean = permutation_entropy(clean, 3);
    const double h_noisy = permutation_entropy(noisy, 3);
    EXPECT_NEAR(h_clean, 0.0, 1e-10);
    // Small noise keeps the noisy entropy close to the clean value, and
    // far below the theoretical maximum of 1.0.
    EXPECT_NEAR(h_noisy, h_clean, 0.3);
    EXPECT_LT(h_noisy, 0.5);
}

TEST(PermutationEntropy, NoiseRobustnessLargeNoiseIncreasesEntropy) {
    // By contrast, noise comparable to (or larger than) the step size
    // between samples should genuinely scramble local orderings and push
    // entropy noticeably higher than the small-noise case, illustrating
    // that permutation entropy tracks disorder in ordinal relationships.
    std::vector<double> clean(30);
    for (size_t i = 0; i < clean.size(); ++i) clean[i] = static_cast<double>(i);

    const double small_noise[6] = {0.02, -0.03, 0.01, -0.01, 0.025, -0.015};
    std::vector<double> small_noisy = clean;
    for (size_t i = 0; i < small_noisy.size(); ++i) small_noisy[i] += small_noise[i % 6];

    const double large_noise[6] = {4.0, -5.0, 3.5, -4.5, 5.0, -3.0};
    std::vector<double> large_noisy = clean;
    for (size_t i = 0; i < large_noisy.size(); ++i) large_noisy[i] += large_noise[i % 6];

    const double h_small = permutation_entropy(small_noisy, 3);
    const double h_large = permutation_entropy(large_noisy, 3);
    EXPECT_GT(h_large, h_small);
}

TEST(PermutationEntropy, NormalizedEntropyBoundedZeroOne) {
    // Sanity: for a variety of constructed series and orders, the
    // normalized value must always lie within [0, 1].
    std::vector<std::vector<double>> series = {
        {1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 2.5, 7.0, 1.5},
        {5.0, -4.0, 7.0, -2.0, 9.0, 0.0, 11.0, 2.0, 13.0, 4.0},
        {10.0, 9.0, 8.0, 7.0, 6.0, 5.0},
        {1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0},
    };
    for (const auto& s : series) {
        for (int order = 2; order <= 4; ++order) {
            const double h = permutation_entropy(s, order);
            EXPECT_GE(h, 0.0);
            EXPECT_LE(h, 1.0 + 1e-9);
        }
    }
}

// --- Transfer entropy (directed information flow between time series) ---

namespace {

std::vector<double> deterministic_noise_series(size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::vector<double> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = dist(rng);
    return out;
}

std::vector<double> causal_y_from_x(const std::vector<double>& x, double noise_amp) {
    std::vector<double> y(x.size(), 0.0);
    const double noise[4] = {0.01, -0.02, 0.015, -0.005};
    for (size_t t = 1; t < x.size(); ++t)
        y[t] = x[t - 1] + noise_amp * noise[t % 4];
    y[0] = x[0];
    return y;
}

} // namespace

TEST(TransferEntropy, IndependentSeriesNearZeroBothDirections) {
    const std::vector<double> x = deterministic_noise_series(500, 42);
    const std::vector<double> y = deterministic_noise_series(500, 137);
    const double te_xy = transfer_entropy(x, y);
    const double te_yx = transfer_entropy(y, x);
    EXPECT_GE(te_xy, 0.0);
    EXPECT_GE(te_yx, 0.0);
    // Independent series: directions should be roughly symmetric (no preferred
    // flow) and substantially weaker than a known causal driver.
    EXPECT_NEAR(te_xy, te_yx, 0.15);

    std::vector<double> x_drv(200);
    for (size_t i = 0; i < x_drv.size(); ++i)
        x_drv[i] = static_cast<double>(i % 17) + 0.1 * static_cast<double>(i % 5);
    const std::vector<double> y_drv = causal_y_from_x(x_drv, 0.05);
    const double te_causal = transfer_entropy(x_drv, y_drv);
    EXPECT_GT(te_causal, te_xy + 0.2);
    EXPECT_GT(te_causal, te_yx + 0.2);
}

TEST(TransferEntropy, CausalDriveAsymmetricDirection) {
    std::vector<double> x(200);
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = static_cast<double>(i % 17) + 0.1 * static_cast<double>(i % 5);
    const std::vector<double> y = causal_y_from_x(x, 0.05);

    const double te_xy = transfer_entropy(x, y);
    const double te_yx = transfer_entropy(y, x);
    EXPECT_GE(te_xy, 0.0);
    EXPECT_GE(te_yx, 0.0);
    EXPECT_GT(te_xy, te_yx);
    EXPECT_GT(te_xy, 0.3);
}

TEST(TransferEntropy, NonNegativityAcrossSyntheticPairs) {
    const std::vector<std::vector<double>> series = {
        {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0},
        {5.0, -4.0, 7.0, -2.0, 9.0, 0.0, 11.0, 2.0, 13.0, 4.0},
        deterministic_noise_series(80, 7),
    };
    for (size_t i = 0; i < series.size(); ++i) {
        for (size_t j = 0; j < series.size(); ++j) {
            const size_t n = std::min(series[i].size(), series[j].size());
            std::vector<double> a(series[i].begin(), series[i].begin() + static_cast<std::ptrdiff_t>(n));
            std::vector<double> b(series[j].begin(), series[j].begin() + static_cast<std::ptrdiff_t>(n));
            EXPECT_GE(transfer_entropy(a, b), 0.0)
                << "pair (" << i << "," << j << ")";
        }
    }
}

TEST(TransferEntropy, LaggedCopyStrongerThanShuffled) {
    std::vector<double> x(120);
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = std::sin(static_cast<double>(i) * 0.17) + 0.3 * std::cos(static_cast<double>(i) * 0.05);

    std::vector<double> y_lagged(x.size());
    y_lagged[0] = x[0];
    for (size_t t = 1; t < x.size(); ++t)
        y_lagged[t] = x[t - 1];

    std::vector<double> x_shuffled = x;
    std::mt19937 rng(99);
    std::shuffle(x_shuffled.begin(), x_shuffled.end(), rng);

    const double te_aligned = transfer_entropy(x, y_lagged);
    const double te_misaligned = transfer_entropy(x_shuffled, y_lagged);
    EXPECT_GT(te_aligned, te_misaligned);
    EXPECT_GT(te_aligned, 0.1);
}

TEST(TransferEntropy, BinCountQualitativelyStable) {
    std::vector<double> x(150);
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = static_cast<double>((i * 7) % 23);
    const std::vector<double> y = causal_y_from_x(x, 0.02);

    const double te4 = transfer_entropy(x, y, 4);
    const double te8 = transfer_entropy(x, y, 8);
    const double te16 = transfer_entropy(x, y, 16);
    EXPECT_GT(te4, 0.0);
    EXPECT_GT(te8, 0.0);
    EXPECT_GT(te16, 0.0);
    const double te_max = std::max({te4, te8, te16});
    const double te_min = std::min({te4, te8, te16});
    EXPECT_LT(te_max / te_min, 4.0);
}

TEST(TransferEntropy, VeryShortSeriesNoCrash) {
    const std::vector<double> x = {1.0, 2.0};
    const std::vector<double> y = {2.0, 3.0};
    const double te = transfer_entropy(x, y);
    EXPECT_GE(te, 0.0);
    EXPECT_TRUE(std::isfinite(te));
}

TEST(TransferEntropy, DegenerateMismatchedLength) {
    const std::vector<double> x = {1.0, 2.0, 3.0};
    const std::vector<double> y = {1.0, 2.0};
    EXPECT_NEAR(transfer_entropy(x, y), 0.0, 1e-12);
    EXPECT_NEAR(transfer_entropy(y, x), 0.0, 1e-12);
}

TEST(TransferEntropy, DegenerateTooShortForLag) {
    const std::vector<double> x = {1.0};
    const std::vector<double> y = {2.0};
    EXPECT_NEAR(transfer_entropy(x, y, 8, 1), 0.0, 1e-12);
    EXPECT_NEAR(transfer_entropy(x, y, 8, 2), 0.0, 1e-12);
}
