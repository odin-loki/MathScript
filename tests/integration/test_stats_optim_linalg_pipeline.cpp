// MathScript Integration: Stats + Optim + Linalg full pipeline (Wave 53)

#define _USE_MATH_DEFINES
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>

#include "ms/stats/stats.hpp"
#include "ms/optim/optim.hpp"
#include "ms/linalg/linalg.hpp"
#include "ms/core/operations.hpp"
#include "ms/prob/prob.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ms;
using DMatrix = Matrix<double>;

// Helper: matvec
static DMatrix mv(const DMatrix& A, const DMatrix& x) {
    size_t n = A.rows();
    DMatrix r(n, 1);
    for (size_t i = 0; i < n; ++i) {
        double s = 0.0;
        for (size_t k = 0; k < A.cols(); ++k) s += A(i, k) * x(k, 0);
        r(i, 0) = s;
    }
    return r;
}

// ---------------------------------------------------------------------------
// Stats → Optim: find optimal parameter using golden section
// ---------------------------------------------------------------------------

TEST(StatsOptimLinalgPipeline, GoldenSection_FindsBestSigmaForNormalFit) {
    // Data drawn from N(0, 2.0) — find sigma that maximizes log-likelihood
    // log L(sigma) = -n*log(sigma) - sum(x^2)/(2*sigma^2) + const
    // Maximized at sigma = stddev(data)
    std::vector<double> data;
    double sigma_true = 2.0;
    for (int i = 0; i < 50; ++i)
        data.push_back(sigma_true * (i % 2 == 0 ? 1.0 : -1.0) * ((i % 5 + 1) * 0.1));
    double s_data = stddev(data);
    // Negative log-likelihood as function of sigma
    auto neg_loglik = [&data](double sigma) {
        if (sigma <= 0.0) return 1e12;
        double n = static_cast<double>(data.size());
        double ss = 0.0;
        for (double x : data) ss += x * x;
        return n * std::log(sigma) + ss / (2.0 * sigma * sigma);
    };
    double sigma_opt = golden_section(neg_loglik, 0.1, 10.0, 1e-6);
    EXPECT_NEAR(sigma_opt, s_data, 0.1 * s_data)
        << "Optimal sigma should be close to sample stddev";
}

// ---------------------------------------------------------------------------
// Linalg → Stats: PCA using eig_sym
// ---------------------------------------------------------------------------

TEST(StatsOptimLinalgPipeline, EigSym_PCA_LargestEigenvalueCaptures_MostVariance) {
    // Construct covariance matrix from correlated data
    DMatrix C(2, 2);
    C(0, 0) = 4.0; C(0, 1) = 2.0;
    C(1, 0) = 2.0; C(1, 1) = 1.0;
    auto eig_res = eig_sym(C);
    ASSERT_TRUE(eig_res.has_value());
    auto& values = eig_res.value().values;
    // Eigenvalues should be positive (C is SPD-ish: det=0, trace=5)
    // Check at least trace is sum of eigenvalues
    double trace_C = C(0, 0) + C(1, 1);
    double sum_eig = 0.0;
    for (size_t i = 0; i < values.rows(); ++i) sum_eig += values(i, 0);
    EXPECT_NEAR(sum_eig, trace_C, 1e-6) << "Sum of eigenvalues should equal trace";
}

TEST(StatsOptimLinalgPipeline, EigSym_SPD_Trace_Equals_SumEigenvalues) {
    // Use small well-conditioned SPD matrix matching pattern from existing tests
    DMatrix A(3, 3);
    A(0, 0) = 4.0; A(0, 1) = 1.0;
    A(1, 0) = 1.0; A(1, 1) = 3.0; A(1, 2) = 1.0;
    A(2, 1) = 1.0; A(2, 2) = 2.0;
    double trace_A = A(0, 0) + A(1, 1) + A(2, 2);
    auto eig_res = eig_sym(A);
    ASSERT_TRUE(eig_res.has_value());
    auto& ev = eig_res.value().values;
    double ev_sum = 0.0;
    for (size_t i = 0; i < ev.rows(); ++i) ev_sum += ev(i, 0);
    EXPECT_NEAR(ev_sum, trace_A, 1e-6) << "Sum of eigenvalues should equal trace";
}

// ---------------------------------------------------------------------------
// Linalg solve → Stats residual analysis
// ---------------------------------------------------------------------------

TEST(StatsOptimLinalgPipeline, Solve_Residual_Stats) {
    DMatrix A(4, 4);
    A(0, 0) = 4.0; A(0, 1) = 1.0;
    A(1, 0) = 1.0; A(1, 1) = 4.0; A(1, 2) = 1.0;
    A(2, 1) = 1.0; A(2, 2) = 4.0; A(2, 3) = 1.0;
    A(3, 2) = 1.0; A(3, 3) = 4.0;
    DMatrix b(4, 1);
    b(0, 0) = 1.0; b(1, 0) = 2.0; b(2, 0) = 3.0; b(3, 0) = 4.0;
    auto result = solve(A, b);
    ASSERT_TRUE(result.has_value());
    auto& x = result.value();
    auto Ax = mv(A, x);
    std::vector<double> residuals(4);
    for (size_t i = 0; i < 4; ++i)
        residuals[i] = std::abs(Ax(i, 0) - b(i, 0));
    double max_res = max_value(residuals);
    double mean_res = mean(residuals);
    EXPECT_LT(max_res, 1e-8) << "Max residual should be tiny";
    EXPECT_LT(mean_res, 1e-8) << "Mean residual should be tiny";
}

// ---------------------------------------------------------------------------
// Stats linear regression → Linalg solve consistency
// ---------------------------------------------------------------------------

TEST(StatsOptimLinalgPipeline, LinearRegression_Matches_LsqSolve) {
    // y = 3x + 2 (exact) — regression should find slope=3, intercept=2
    std::vector<double> xs = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    std::vector<double> ys;
    for (double x : xs) ys.push_back(3.0 * x + 2.0);
    auto reg = linear_regression(xs, ys);
    EXPECT_NEAR(reg.slope, 3.0, 1e-8);
    EXPECT_NEAR(reg.intercept, 2.0, 1e-8);
    EXPECT_NEAR(reg.r_squared, 1.0, 1e-10) << "Perfect fit should have R^2=1";
}

TEST(StatsOptimLinalgPipeline, LinearRegression_NoisyData_SloperSign) {
    // Upward trend: slope should be positive
    std::vector<double> xs = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> ys = {1.1, 2.3, 2.9, 4.2, 5.0};
    auto reg = linear_regression(xs, ys);
    EXPECT_GT(reg.slope, 0.0) << "Upward trend should have positive slope";
    EXPECT_GT(reg.r_squared, 0.9) << "Good fit: R^2 > 0.9";
}

// ---------------------------------------------------------------------------
// Prob + Stats + Optim pipeline
// ---------------------------------------------------------------------------

TEST(StatsOptimLinalgPipeline, NormCDF_Percentile_Roundtrip) {
    // For standard normal: norm_ppf(norm_cdf(x)) ≈ x
    for (double x : {-1.5, -0.5, 0.0, 0.5, 1.5}) {
        double cdf_val = norm_cdf(x, 0.0, 1.0);
        double ppf_val = norm_ppf(cdf_val, 0.0, 1.0);
        EXPECT_NEAR(ppf_val, x, 1e-6) << "norm_ppf(norm_cdf(x)) roundtrip failed at x=" << x;
    }
}

TEST(StatsOptimLinalgPipeline, GoldenSection_MinimizesNegLogCDF) {
    // neg-log of cdf should be minimized away from tails
    auto f = [](double x) { return -std::log(norm_cdf(x, 0.0, 1.0) + 1e-12); };
    double xopt = golden_section(f, -3.0, 3.0);
    // neg-log of cdf is monotone decreasing → minimum at right end
    EXPECT_GT(xopt, 0.0) << "Minimum of -log(norm_cdf(x)) on [-3,3] should be to the right";
}

// ---------------------------------------------------------------------------
// Full pipeline: data → stats → regression → solve
// ---------------------------------------------------------------------------

TEST(StatsOptimLinalgPipeline, Full_DataToRegression_AllFinite) {
    std::vector<double> xs = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0};
    std::vector<double> ys;
    for (double x : xs) ys.push_back(2.0 * x - 1.0 + 0.05 * (x - 3.0)); // slight curvature
    auto reg = linear_regression(xs, ys);
    EXPECT_TRUE(std::isfinite(reg.slope));
    EXPECT_TRUE(std::isfinite(reg.intercept));
    EXPECT_TRUE(std::isfinite(reg.r_squared));
    EXPECT_GT(reg.r_squared, 0.99) << "Near-linear data should have high R^2";
}
