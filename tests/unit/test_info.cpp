#define _USE_MATH_DEFINES
#include "ms/info/info.hpp"
#include <cmath>
#include <gtest/gtest.h>
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
