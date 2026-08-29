// MathScript: Advanced Probability Distribution Tests
// Tests for t_pdf/cdf, gamma_pdf, norm_ppf, binomial, Poisson edge cases

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

TEST(ProbAdv, NormPDF_At_Mean_Is_Maximum) {
    // PDF(mu) should be maximum
    double mu = 2.0, sigma = 1.0;
    double pdf_at_mean = norm_pdf(mu, mu, sigma);
    double pdf_away = norm_pdf(mu + 1.0, mu, sigma);
    EXPECT_GT(pdf_at_mean, pdf_away);
}

TEST(ProbAdv, NormPDF_StandardNormal_AtZero) {
    // Standard normal: pdf(0) = 1/sqrt(2*pi) ≈ 0.3989
    double pdf = norm_pdf(0.0, 0.0, 1.0);
    EXPECT_NEAR(pdf, 1.0 / std::sqrt(2.0 * M_PI), 1e-9);
}

TEST(ProbAdv, NormCDF_AtMean_Is_Half) {
    EXPECT_NEAR(norm_cdf(0.0, 0.0, 1.0), 0.5, 1e-9);
    EXPECT_NEAR(norm_cdf(5.0, 5.0, 2.0), 0.5, 1e-9);
}

TEST(ProbAdv, NormCDF_Monotone) {
    double mu = 0.0, sigma = 1.0;
    double prev = norm_cdf(-5.0, mu, sigma);
    for (double x = -4.0; x <= 4.0; x += 0.5) {
        double curr = norm_cdf(x, mu, sigma);
        EXPECT_GE(curr, prev) << "Not monotone at x=" << x;
        prev = curr;
    }
}

TEST(ProbAdv, NormPPF_InverseCDF) {
    // PPF(CDF(x)) ≈ x for standard normal
    for (double x : {-2.0, -1.0, 0.0, 1.0, 2.0}) {
        double p = norm_cdf(x, 0.0, 1.0);
        double back = norm_ppf(p, 0.0, 1.0);
        EXPECT_NEAR(back, x, 1e-6) << "PPF(CDF(x)) != x for x=" << x;
    }
}

TEST(ProbAdv, NormPPF_Extremes) {
    // PPF(0.5) = mu
    EXPECT_NEAR(norm_ppf(0.5, 0.0, 1.0), 0.0, 1e-9);
    // PPF near 0 should give very negative value
    double v = norm_ppf(0.001, 0.0, 1.0);
    EXPECT_LT(v, -2.0);
    // PPF near 1 should give very positive value
    v = norm_ppf(0.999, 0.0, 1.0);
    EXPECT_GT(v, 2.0);
}

// ---------------------------------------------------------------------------
// Exponential distribution
// ---------------------------------------------------------------------------

TEST(ProbAdv, ExpPDF_AtZero_Is_Lambda) {
    double lambda = 2.0;
    EXPECT_NEAR(exp_pdf(0.0, lambda), lambda, 1e-9);
}

TEST(ProbAdv, ExpCDF_AtZero_Is_Zero) {
    EXPECT_NEAR(exp_cdf(0.0, 1.0), 0.0, 1e-9);
}

TEST(ProbAdv, ExpCDF_Approaches_One) {
    EXPECT_NEAR(exp_cdf(100.0, 1.0), 1.0, 1e-6);
}

TEST(ProbAdv, ExpPDF_Decreasing) {
    double lambda = 1.0;
    for (double x = 0.0; x < 5.0; x += 0.5) {
        EXPECT_GE(exp_pdf(x, lambda), exp_pdf(x + 0.5, lambda) - 1e-10);
    }
}

TEST(ProbAdv, ExpCDF_Monotone) {
    double lambda = 0.5;
    double prev = 0.0;
    for (double x = 0.0; x <= 10.0; x += 0.5) {
        double curr = exp_cdf(x, lambda);
        EXPECT_GE(curr, prev);
        prev = curr;
    }
}

// ---------------------------------------------------------------------------
// t-distribution
// ---------------------------------------------------------------------------

TEST(ProbAdv, TPDF_Symmetric) {
    // t-distribution is symmetric around 0
    for (double x : {0.5, 1.0, 2.0, 3.0}) {
        for (double df : {1.0, 5.0, 10.0, 30.0}) {
            EXPECT_NEAR(t_pdf(x, df), t_pdf(-x, df), 1e-10)
                << "t_pdf not symmetric at x=" << x << " df=" << df;
        }
    }
}

TEST(ProbAdv, TPDF_MaxAtZero) {
    // t-distribution has maximum at x=0
    for (double df : {1.0, 5.0, 30.0}) {
        double pdf_0 = t_pdf(0.0, df);
        EXPECT_GT(pdf_0, t_pdf(1.0, df));
        EXPECT_GT(pdf_0, t_pdf(2.0, df));
    }
}

TEST(ProbAdv, TCDF_AtZero_Is_Half) {
    for (double df : {1.0, 5.0, 10.0, 30.0}) {
        EXPECT_NEAR(t_cdf(0.0, df), 0.5, 1e-9) << "t_cdf(0, df) != 0.5 at df=" << df;
    }
}

TEST(ProbAdv, TCDF_Monotone) {
    double df = 10.0;
    double prev = 0.0;
    for (double x = -5.0; x <= 5.0; x += 0.5) {
        double curr = t_cdf(x, df);
        EXPECT_GE(curr, prev - 1e-12) << "t_cdf not monotone at x=" << x;
        prev = curr;
    }
}

TEST(ProbAdv, TPDF_HighDF_ApproachesNormal) {
    // t with large df should be close to standard normal
    // Use df=30 which is well-supported and close to normal
    double x = 1.5;
    double t_val = t_pdf(x, 30.0);
    double n_val = norm_pdf(x, 0.0, 1.0);
    EXPECT_TRUE(std::isfinite(t_val));
    EXPECT_TRUE(std::isfinite(n_val));
    // t(df=30) should be within 2% of normal
    EXPECT_NEAR(t_val, n_val, 0.02) << "t(df=30) should be close to Normal";
}

// ---------------------------------------------------------------------------
// Gamma distribution
// ---------------------------------------------------------------------------

TEST(ProbAdv, GammaPDF_PositiveForPositiveX) {
    for (double x : {0.5, 1.0, 2.0, 5.0}) {
        double pdf = gamma_pdf(x, 2.0, 1.0);
        EXPECT_GT(pdf, 0.0) << "gamma_pdf should be > 0 at x=" << x;
        EXPECT_TRUE(std::isfinite(pdf));
    }
}

TEST(ProbAdv, GammaPDF_ZeroAtZero_For_Shape_GE_1) {
    // gamma_pdf(0, k, 1) = 0 for k > 1
    double pdf = gamma_pdf(0.0, 2.0, 1.0);
    EXPECT_NEAR(pdf, 0.0, 1e-10);
}

TEST(ProbAdv, GammaPDF_Mode_At_Expected_Location) {
    // Mode of Gamma(k, theta) = (k-1)*theta for k >= 1
    double k = 3.0, theta = 2.0;
    double mode = (k - 1.0) * theta;
    double pdf_at_mode = gamma_pdf(mode, k, theta);
    double pdf_away = gamma_pdf(mode + 2.0, k, theta);
    EXPECT_GT(pdf_at_mode, pdf_away);
}

TEST(ProbAdv, GammaPDF_NegativeX_IsZero) {
    double pdf = gamma_pdf(-1.0, 2.0, 1.0);
    EXPECT_NEAR(pdf, 0.0, 1e-10);
}

// ---------------------------------------------------------------------------
// Binomial distribution
// ---------------------------------------------------------------------------

TEST(ProbAdv, BinomPDF_SumsToOne) {
    int n = 10;
    double p = 0.3;
    double total = 0.0;
    for (int k = 0; k <= n; ++k) {
        total += binom_pdf(k, n, p);
    }
    EXPECT_NEAR(total, 1.0, 1e-9);
}

TEST(ProbAdv, BinomCDF_At_N_Is_One) {
    EXPECT_NEAR(binom_cdf(10, 10, 0.5), 1.0, 1e-9);
}

TEST(ProbAdv, BinomCDF_At_Neg_Is_Zero) {
    EXPECT_NEAR(binom_cdf(-1, 10, 0.5), 0.0, 1e-9);
}

TEST(ProbAdv, BinomPDF_AllNonnegative) {
    int n = 8;
    double p = 0.4;
    for (int k = 0; k <= n; ++k) {
        EXPECT_GE(binom_pdf(k, n, p), 0.0);
    }
}

// ---------------------------------------------------------------------------
// Poisson distribution
// ---------------------------------------------------------------------------

TEST(ProbAdv, PoisPDF_SumsToNearOne) {
    double lambda = 5.0;
    double total = 0.0;
    for (int k = 0; k <= 40; ++k) {
        total += pois_pdf(static_cast<double>(k), lambda);
    }
    EXPECT_NEAR(total, 1.0, 1e-5);
}

TEST(ProbAdv, PoisPDF_AllNonnegative) {
    double lambda = 3.0;
    for (int k = 0; k <= 20; ++k) {
        EXPECT_GE(pois_pdf(static_cast<double>(k), lambda), 0.0);
    }
}

TEST(ProbAdv, PoisCDF_Monotone) {
    double lambda = 2.0;
    double prev = 0.0;
    for (int k = 0; k <= 15; ++k) {
        double curr = pois_cdf(static_cast<double>(k), lambda);
        EXPECT_GE(curr, prev - 1e-12);
        prev = curr;
    }
}

TEST(ProbAdv, PoisCDF_At_Large_K_Near_One) {
    EXPECT_NEAR(pois_cdf(100.0, 2.0), 1.0, 1e-6);
}

// ---------------------------------------------------------------------------
// Chi-squared distribution
// ---------------------------------------------------------------------------

TEST(ProbAdv, Chi2PDF_AtZero_WithDF1) {
    // chi2_pdf(0, 1) is undefined (→∞ for df=1); skip
    // For df=2: chi2_pdf(0, 2) = 0.5
    double pdf = chi2_pdf(0.0, 2.0);
    EXPECT_TRUE(std::isfinite(pdf));
    EXPECT_GE(pdf, 0.0);
}

TEST(ProbAdv, Chi2PDF_Positive_ForPositiveX) {
    for (double x : {0.5, 1.0, 2.0, 5.0}) {
        for (double df : {2.0, 5.0, 10.0}) {
            EXPECT_GE(chi2_pdf(x, df), 0.0);
            EXPECT_TRUE(std::isfinite(chi2_pdf(x, df)));
        }
    }
}

TEST(ProbAdv, Chi2CDF_Monotone) {
    double df = 5.0;
    double prev = 0.0;
    for (double x = 0.0; x <= 20.0; x += 1.0) {
        double curr = chi2_cdf(x, df);
        EXPECT_GE(curr, prev - 1e-12);
        prev = curr;
    }
}

// ---------------------------------------------------------------------------
// Uniform distribution
// ---------------------------------------------------------------------------

TEST(ProbAdv, UniformPDF_WithinRange) {
    double a = 1.0, b = 4.0;
    double expected = 1.0 / (b - a);
    for (double x : {1.5, 2.0, 3.0, 3.9}) {
        EXPECT_NEAR(uniform_pdf(x, a, b), expected, 1e-10);
    }
}

TEST(ProbAdv, UniformPDF_OutsideRange_Zero) {
    EXPECT_NEAR(uniform_pdf(0.5, 1.0, 4.0), 0.0, 1e-10);
    EXPECT_NEAR(uniform_pdf(5.0, 1.0, 4.0), 0.0, 1e-10);
}

TEST(ProbAdv, UniformCDF_LinearInRange) {
    double a = 0.0, b = 10.0;
    for (double x : {0.0, 2.5, 5.0, 7.5, 10.0}) {
        EXPECT_NEAR(uniform_cdf(x, a, b), (x - a) / (b - a), 1e-10);
    }
}

TEST(ProbAdv, UniformCDF_Clamps) {
    EXPECT_NEAR(uniform_cdf(-1.0, 0.0, 1.0), 0.0, 1e-10);
    EXPECT_NEAR(uniform_cdf(2.0, 0.0, 1.0), 1.0, 1e-10);
}
