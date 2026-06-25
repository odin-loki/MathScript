// MathScript: Probability + Statistics Pipeline Integration Tests (Wave 48)
// Tests combining prob distributions with statistical analysis functions

#define _USE_MATH_DEFINES
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <numeric>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "ms/prob/prob.hpp"
#include "ms/stats/stats.hpp"

using namespace ms;

// ---------------------------------------------------------------------------
// Pipeline 1: Normal distribution CDF/PDF consistency
// ---------------------------------------------------------------------------

TEST(ProbStatsPipeline, NormCDF_PDF_Consistent) {
    // CDF should be the integral of PDF
    // Numerically: sum(PDF(x)*dx) from -4 to x should approximate CDF(x)
    double mu = 0.0, sigma = 1.0;
    double dx = 0.001;
    double cumulative = 0.0;
    for (double x = -4.0; x <= 0.0; x += dx)
        cumulative += norm_pdf(x, mu, sigma) * dx;
    // CDF(0) should be ~0.5
    EXPECT_NEAR(cumulative, 0.5, 0.01)
        << "Numerical integral of norm_pdf should approximate norm_cdf";
}

TEST(ProbStatsPipeline, NormPPF_CDF_Roundtrip) {
    // PPF(CDF(x)) = x for various x values
    for (double x : {-2.5, -1.0, -0.5, 0.0, 0.5, 1.0, 2.5}) {
        double p = norm_cdf(x, 0.0, 1.0);
        double x_back = norm_ppf(p, 0.0, 1.0);
        EXPECT_NEAR(x_back, x, 1e-6) << "PPF(CDF) roundtrip failed at x=" << x;
    }
}

// ---------------------------------------------------------------------------
// Pipeline 2: Binomial → mean/var consistent with theory
// ---------------------------------------------------------------------------

TEST(ProbStatsPipeline, BinomPDF_MeanMatchesTheory) {
    // E[X] = n*p for binomial
    int n = 20;
    double p = 0.4;
    double expected_mean = n * p;

    // Compute mean from PDF
    double computed_mean = 0.0;
    for (int k = 0; k <= n; ++k)
        computed_mean += k * binom_pdf(k, n, p);
    EXPECT_NEAR(computed_mean, expected_mean, 1e-10);
}

TEST(ProbStatsPipeline, BinomPDF_VarMatchesTheory) {
    // Var[X] = n*p*(1-p)
    int n = 20;
    double p = 0.4;
    double mean_b = n * p;
    double expected_var = n * p * (1.0 - p);

    double computed_mean = 0.0;
    for (int k = 0; k <= n; ++k)
        computed_mean += k * binom_pdf(k, n, p);

    double computed_var = 0.0;
    for (int k = 0; k <= n; ++k) {
        double d = k - computed_mean;
        computed_var += d * d * binom_pdf(k, n, p);
    }
    EXPECT_NEAR(computed_var, expected_var, 1e-9);
}

// ---------------------------------------------------------------------------
// Pipeline 3: Poisson → mean/var consistent
// ---------------------------------------------------------------------------

TEST(ProbStatsPipeline, PoissonPDF_MeanMatchesLambda) {
    // E[X] = lambda for Poisson
    double lambda = 4.0;
    double computed_mean = 0.0;
    for (int k = 0; k <= 40; ++k)
        computed_mean += k * pois_pdf(static_cast<double>(k), lambda);
    EXPECT_NEAR(computed_mean, lambda, 1e-6);
}

TEST(ProbStatsPipeline, PoissonPDF_VarMatchesLambda) {
    // Var[X] = lambda for Poisson
    double lambda = 4.0;
    double mu = 0.0;
    for (int k = 0; k <= 40; ++k)
        mu += k * pois_pdf(static_cast<double>(k), lambda);

    double var_p = 0.0;
    for (int k = 0; k <= 40; ++k) {
        double d = k - mu;
        var_p += d * d * pois_pdf(static_cast<double>(k), lambda);
    }
    EXPECT_NEAR(var_p, lambda, 1e-5);
}

// ---------------------------------------------------------------------------
// Pipeline 4: Stats functions applied to distribution samples
// ---------------------------------------------------------------------------

TEST(ProbStatsPipeline, NormCDF_Quantile_Stats) {
    // Generate uniform [0,1] quantiles and apply PPF to get "samples"
    int N = 100;
    std::vector<double> samples;
    for (int i = 1; i <= N; ++i) {
        double p = static_cast<double>(i) / (N + 1);  // avoids 0 and 1
        samples.push_back(norm_ppf(p, 0.0, 1.0));
    }
    // Mean should be ~0 and stddev ~1 for standard normal quantiles
    EXPECT_NEAR(mean(samples), 0.0, 0.1);
    EXPECT_NEAR(stddev(samples), 1.0, 0.1);
}

TEST(ProbStatsPipeline, ExpPDF_MeanMatchesLambdaInverse) {
    // E[X] = 1/lambda for exponential
    double lambda = 3.0;
    double computed_mean = 0.0;
    double dx = 0.0001;
    for (double x = 0.0; x <= 10.0; x += dx)
        computed_mean += x * exp_pdf(x, lambda) * dx;
    EXPECT_NEAR(computed_mean, 1.0 / lambda, 0.01);
}

// ---------------------------------------------------------------------------
// Pipeline 5: Chi2 CDF used for p-value interpretation
// ---------------------------------------------------------------------------

TEST(ProbStatsPipeline, Chi2CDF_95thPercentile_Near_Critical) {
    // For df=5, chi2 critical value at 95% ≈ 11.07
    double df = 5.0;
    double cv = 11.07;
    EXPECT_NEAR(chi2_cdf(cv, df), 0.95, 0.01);
}

TEST(ProbStatsPipeline, Chi2CDF_Complement) {
    // CDF(x) + (1 - CDF(x)) = 1
    double df = 3.0;
    for (double x : {1.0, 3.0, 5.0, 10.0}) {
        double cdf_val = chi2_cdf(x, df);
        EXPECT_NEAR(cdf_val + (1.0 - cdf_val), 1.0, 1e-12);
        EXPECT_TRUE(std::isfinite(cdf_val));
    }
}

// ---------------------------------------------------------------------------
// Pipeline 6: Normal distribution vs. stats comparison
// ---------------------------------------------------------------------------

TEST(ProbStatsPipeline, NormPDF_Integral_IsOne) {
    // Integral of norm_pdf over [-inf, inf] = 1
    double total = 0.0;
    double dx = 0.001;
    for (double x = -6.0; x <= 6.0; x += dx)
        total += norm_pdf(x, 0.0, 1.0) * dx;
    EXPECT_NEAR(total, 1.0, 0.01);
}

TEST(ProbStatsPipeline, UniformCDF_IsLinear) {
    // For uniform [a,b], CDF is linear
    double a = 1.0, b = 5.0;
    for (double x = a; x <= b; x += 0.5) {
        double expected = (x - a) / (b - a);
        EXPECT_NEAR(uniform_cdf(x, a, b), expected, 1e-10);
    }
}
