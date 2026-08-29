// MathScript: Advanced Probability Distribution Tests (Wave 48)
// Tests for norm, exp, binom, pois, chi2, t, uniform, gamma PDFs/CDFs/PPF

#define _USE_MATH_DEFINES
#include <gtest/gtest.h>
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "ms/prob/prob.hpp"

using namespace ms;

// ---------------------------------------------------------------------------
// Normal distribution
// ---------------------------------------------------------------------------

TEST(ProbAdv2, NormPDF_AtMean_IsMaximum) {
    // PDF is maximised at mean
    double mu = 2.0, sigma = 1.0;
    double at_mean = norm_pdf(mu, mu, sigma);
    double off_mean = norm_pdf(mu + 1.0, mu, sigma);
    EXPECT_GT(at_mean, off_mean);
}

TEST(ProbAdv2, NormPDF_Symmetric) {
    // Normal PDF is symmetric around mean
    double mu = 0.0, sigma = 1.5;
    EXPECT_NEAR(norm_pdf(mu + 1.0, mu, sigma), norm_pdf(mu - 1.0, mu, sigma), 1e-12);
}

TEST(ProbAdv2, NormPDF_NonNegative) {
    for (double x : {-3.0, -1.0, 0.0, 1.0, 3.0})
        EXPECT_GE(norm_pdf(x, 0.0, 1.0), 0.0);
}

TEST(ProbAdv2, NormPDF_StandardNormal_AtZero) {
    // phi(0) = 1/sqrt(2*pi)
    double expected = 1.0 / std::sqrt(2.0 * M_PI);
    EXPECT_NEAR(norm_pdf(0.0, 0.0, 1.0), expected, 1e-12);
}

TEST(ProbAdv2, NormCDF_AtMean_IsHalf) {
    EXPECT_NEAR(norm_cdf(0.0, 0.0, 1.0), 0.5, 1e-10);
    EXPECT_NEAR(norm_cdf(3.0, 3.0, 2.0), 0.5, 1e-10);
}

TEST(ProbAdv2, NormCDF_Monotone) {
    for (double x = -3.0; x < 3.0; x += 0.5)
        EXPECT_LT(norm_cdf(x, 0.0, 1.0), norm_cdf(x + 0.5, 0.0, 1.0));
}

TEST(ProbAdv2, NormCDF_Bounds) {
    EXPECT_NEAR(norm_cdf(-10.0, 0.0, 1.0), 0.0, 1e-5);
    EXPECT_NEAR(norm_cdf(10.0, 0.0, 1.0), 1.0, 1e-5);
}

TEST(ProbAdv2, NormPPF_InverseOfCDF) {
    // PPF(CDF(x)) = x
    for (double x : {-2.0, -1.0, 0.0, 1.0, 2.0}) {
        double p = norm_cdf(x, 0.0, 1.0);
        EXPECT_NEAR(norm_ppf(p, 0.0, 1.0), x, 1e-8)
            << "PPF(CDF(x)) != x at x=" << x;
    }
}

TEST(ProbAdv2, NormPPF_Quantiles) {
    // norm_ppf(0.5) = 0 for standard normal
    EXPECT_NEAR(norm_ppf(0.5, 0.0, 1.0), 0.0, 1e-8);
    // 97.5th percentile ≈ 1.96
    EXPECT_NEAR(norm_ppf(0.975, 0.0, 1.0), 1.96, 0.01);
}

// ---------------------------------------------------------------------------
// Exponential distribution
// ---------------------------------------------------------------------------

TEST(ProbAdv2, ExpPDF_AtZero_IsLambda) {
    double lambda = 2.0;
    EXPECT_NEAR(exp_pdf(0.0, lambda), lambda, 1e-10);
}

TEST(ProbAdv2, ExpPDF_Decreasing) {
    double lambda = 1.5;
    for (double x = 0.0; x < 3.0; x += 0.5)
        EXPECT_GT(exp_pdf(x, lambda), exp_pdf(x + 0.5, lambda));
}

TEST(ProbAdv2, ExpPDF_NonNegative) {
    for (double x : {0.0, 0.5, 1.0, 2.0, 5.0})
        EXPECT_GE(exp_pdf(x, 1.0), 0.0);
}

TEST(ProbAdv2, ExpCDF_AtZero_IsZero) {
    EXPECT_NEAR(exp_cdf(0.0, 1.0), 0.0, 1e-10);
}

TEST(ProbAdv2, ExpCDF_Monotone) {
    for (double x = 0.0; x < 5.0; x += 0.5)
        EXPECT_LT(exp_cdf(x, 1.0), exp_cdf(x + 0.5, 1.0));
}

TEST(ProbAdv2, ExpCDF_Approaches1_LargeX) {
    EXPECT_NEAR(exp_cdf(20.0, 1.0), 1.0, 1e-6);
}

// ---------------------------------------------------------------------------
// Binomial distribution
// ---------------------------------------------------------------------------

TEST(ProbAdv2, BinomPDF_Sums_To_One) {
    int n = 10;
    double p = 0.3;
    double total = 0.0;
    for (int k = 0; k <= n; ++k) total += binom_pdf(k, n, p);
    EXPECT_NEAR(total, 1.0, 1e-10);
}

TEST(ProbAdv2, BinomPDF_NonNegative) {
    for (int k = 0; k <= 10; ++k)
        EXPECT_GE(binom_pdf(k, 10, 0.5), 0.0);
}

TEST(ProbAdv2, BinomPDF_p05_Symmetric) {
    // Symmetric for p=0.5
    int n = 8;
    for (int k = 0; k <= n; ++k)
        EXPECT_NEAR(binom_pdf(k, n, 0.5), binom_pdf(n - k, n, 0.5), 1e-10)
            << "Binom not symmetric at k=" << k;
}

TEST(ProbAdv2, BinomCDF_AtN_IsOne) {
    EXPECT_NEAR(binom_cdf(10, 10, 0.7), 1.0, 1e-10);
}

TEST(ProbAdv2, BinomCDF_Monotone) {
    int n = 10;
    double p = 0.4;
    for (int k = 0; k < n; ++k)
        EXPECT_LE(binom_cdf(k, n, p), binom_cdf(k + 1, n, p));
}

// ---------------------------------------------------------------------------
// Poisson distribution
// ---------------------------------------------------------------------------

TEST(ProbAdv2, PoisPDF_Sums_To_One) {
    double lambda = 3.0;
    double total = 0.0;
    for (int k = 0; k <= 30; ++k) total += pois_pdf(static_cast<double>(k), lambda);
    EXPECT_NEAR(total, 1.0, 1e-6);
}

TEST(ProbAdv2, PoisPDF_AtZero) {
    // P(X=0) = exp(-lambda)
    double lambda = 2.0;
    EXPECT_NEAR(pois_pdf(0.0, lambda), std::exp(-lambda), 1e-12);
}

TEST(ProbAdv2, PoisPDF_NonNegative) {
    for (int k = 0; k <= 10; ++k)
        EXPECT_GE(pois_pdf(static_cast<double>(k), 2.0), 0.0);
}

TEST(ProbAdv2, PoisCDF_Monotone) {
    double lambda = 4.0;
    for (int k = 0; k < 15; ++k)
        EXPECT_LE(pois_cdf(static_cast<double>(k), lambda),
                  pois_cdf(static_cast<double>(k + 1), lambda));
}

// ---------------------------------------------------------------------------
// Chi-squared distribution
// ---------------------------------------------------------------------------

TEST(ProbAdv2, Chi2PDF_NonNegative) {
    for (double x : {0.1, 0.5, 1.0, 2.0, 5.0, 10.0})
        EXPECT_GE(chi2_pdf(x, 2.0), 0.0);
}

TEST(ProbAdv2, Chi2CDF_AtZero_IsZero) {
    EXPECT_NEAR(chi2_cdf(0.0, 2.0), 0.0, 1e-10);
}

TEST(ProbAdv2, Chi2CDF_Monotone) {
    double df = 3.0;
    for (double x = 0.5; x < 10.0; x += 1.0)
        EXPECT_LT(chi2_cdf(x, df), chi2_cdf(x + 1.0, df));
}

TEST(ProbAdv2, Chi2CDF_Approaches1_LargeX) {
    EXPECT_NEAR(chi2_cdf(100.0, 5.0), 1.0, 1e-6);
}

// ---------------------------------------------------------------------------
// Uniform distribution
// ---------------------------------------------------------------------------

TEST(ProbAdv2, UniformPDF_Inside_IsConstant) {
    // PDF = 1/(b-a) inside [a,b]
    double a = 1.0, b = 3.0;
    double pdf_val = 1.0 / (b - a);
    for (double x : {1.5, 2.0, 2.5})
        EXPECT_NEAR(uniform_pdf(x, a, b), pdf_val, 1e-10);
}

TEST(ProbAdv2, UniformPDF_Outside_IsZero) {
    EXPECT_NEAR(uniform_pdf(0.5, 1.0, 3.0), 0.0, 1e-10);
    EXPECT_NEAR(uniform_pdf(3.5, 1.0, 3.0), 0.0, 1e-10);
}

TEST(ProbAdv2, UniformCDF_AtA_IsZero) {
    EXPECT_NEAR(uniform_cdf(1.0, 1.0, 3.0), 0.0, 1e-10);
}

TEST(ProbAdv2, UniformCDF_AtB_IsOne) {
    EXPECT_NEAR(uniform_cdf(3.0, 1.0, 3.0), 1.0, 1e-10);
}

TEST(ProbAdv2, UniformCDF_Midpoint_IsHalf) {
    double a = 0.0, b = 4.0;
    EXPECT_NEAR(uniform_cdf(2.0, a, b), 0.5, 1e-10);
}

// ---------------------------------------------------------------------------
// Student's t-distribution
// ---------------------------------------------------------------------------

TEST(ProbAdv2, TPDF_Symmetric) {
    // t-pdf is symmetric
    double df = 5.0;
    for (double x : {0.5, 1.0, 2.0})
        EXPECT_NEAR(t_pdf(x, df), t_pdf(-x, df), 1e-12);
}

TEST(ProbAdv2, TPDF_NonNegative) {
    for (double x : {-3.0, -1.0, 0.0, 1.0, 3.0})
        EXPECT_GE(t_pdf(x, 5.0), 0.0);
}

TEST(ProbAdv2, TCDF_AtZero_IsHalf) {
    EXPECT_NEAR(t_cdf(0.0, 5.0), 0.5, 1e-10);
}

TEST(ProbAdv2, TCDF_Monotone) {
    double df = 10.0;
    for (double x = -3.0; x < 3.0; x += 0.5)
        EXPECT_LT(t_cdf(x, df), t_cdf(x + 0.5, df));
}
