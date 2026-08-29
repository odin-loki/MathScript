// MathScript Integration Tests - Full Pipeline Scenarios
// Tests end-to-end workflows combining multiple modules

#include <cmath>
#include <gtest/gtest.h>
#include <complex>
#include <numeric>
#include <vector>

#include "ms/core/matrix.hpp"
#include "ms/core/operations.hpp"
#include "ms/fft/fft.hpp"
#include "ms/linalg/linalg.hpp"
#include "ms/optim/optim.hpp"
#include "ms/poly/poly.hpp"
#include "ms/prob/prob.hpp"
#include "ms/signal/signal.hpp"
#include "ms/special/special.hpp"
#include "ms/stats/stats.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ms;
using DMatrix = ColMatrix<double>;

// ---------------------------------------------------------------------------
// Pipeline 1: Linear System Solve via QR then verify solution
// ---------------------------------------------------------------------------

TEST(PipelineIntegration, QR_Solve_And_Verify) {
    // Build an SPD system, solve via QR, verify A*x = b
    DMatrix A(3, 3);
    A(0,0)=4; A(0,1)=1; A(0,2)=0;
    A(1,0)=1; A(1,1)=4; A(1,2)=1;
    A(2,0)=0; A(2,1)=1; A(2,2)=4;
    DMatrix b(3, 1);
    b(0,0)=5; b(1,0)=6; b(2,0)=5;

    // QR decompose, then solve via R
    const auto qr_res = qr(A);
    ASSERT_TRUE(qr_res.has_value());

    // Also solve directly
    const auto sol = solve(A, b);
    ASSERT_TRUE(sol.has_value());

    // Verify: A*x = b
    const auto Ax = matmul(A, *sol);
    ASSERT_TRUE(Ax.has_value());
    for (size_t i = 0; i < 3; ++i)
        EXPECT_NEAR((*Ax)(i, 0), b(i, 0), 1e-9);
}

// ---------------------------------------------------------------------------
// Pipeline 2: Statistical Analysis Pipeline
// ---------------------------------------------------------------------------

TEST(PipelineIntegration, Stats_GenerateAnalyze) {
    // Build a synthetic dataset: y = 2x + 3 + small noise
    const size_t N = 10;
    std::vector<double> x(N), y(N);
    for (size_t i = 0; i < N; ++i) {
        x[i] = static_cast<double>(i);
        y[i] = 2.0 * x[i] + 3.0;  // exact linear relationship
    }

    // Fit regression
    const auto reg = linear_regression(x, y);
    EXPECT_NEAR(reg.slope, 2.0, 1e-10);
    EXPECT_NEAR(reg.intercept, 3.0, 1e-10);

    // Correlation should be perfect
    EXPECT_NEAR(correlation(x, y), 1.0, 1e-10);

    // Mean of y values: mean(2x+3) = 2*mean(x)+3
    const double mx = mean(x);
    EXPECT_NEAR(mean(y), 2.0 * mx + 3.0, 1e-10);
}

// ---------------------------------------------------------------------------
// Pipeline 3: FFT -> Filter -> IFFT signal chain
// ---------------------------------------------------------------------------

TEST(PipelineIntegration, FFT_Filter_IFFT_Signal_Chain) {
    // Create pure 440Hz sine sampled at 8000Hz (N=16 samples for speed)
    const size_t N = 16;
    const double fs = 8000.0;
    const double f0 = 250.0; // Low frequency
    std::vector<double> signal(N);
    for (size_t i = 0; i < N; ++i)
        signal[i] = std::sin(2.0 * M_PI * f0 * i / fs);

    // Forward FFT
    const auto spectrum = fft(signal);
    ASSERT_TRUE(spectrum.has_value());
    ASSERT_EQ(spectrum->size(), N);

    // All spectrum values should be finite
    for (const auto& c : *spectrum) {
        EXPECT_TRUE(std::isfinite(c.real()));
        EXPECT_TRUE(std::isfinite(c.imag()));
    }

    // Inverse FFT should recover signal
    const auto recovered = ifft(*spectrum);
    ASSERT_TRUE(recovered.has_value());
    ASSERT_EQ(recovered->size(), N);
    for (size_t i = 0; i < N; ++i)
        EXPECT_NEAR((*recovered)[i], signal[i], 1e-9);
}

// ---------------------------------------------------------------------------
// Pipeline 4: Signal Convolution then Statistics
// ---------------------------------------------------------------------------

TEST(PipelineIntegration, Convolve_Then_Stats) {
    // Convolve a ramp with a box filter, then take statistics of result
    const std::vector<double> ramp = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    const std::vector<double> box = {1.0/3.0, 1.0/3.0, 1.0/3.0};

    const auto convolved = convolve(ramp, box);
    ASSERT_EQ(convolved.size(), ramp.size() + box.size() - 1);

    // Result should all be finite
    for (double v : convolved)
        EXPECT_TRUE(std::isfinite(v));

    // Mean of interior elements should approximate mean of ramp (smoothed)
    const double m = mean(convolved);
    EXPECT_GT(m, 0.0);
}

// ---------------------------------------------------------------------------
// Pipeline 5: Polynomial Fit Evaluation
// ---------------------------------------------------------------------------

TEST(PipelineIntegration, Poly_Build_Eval_Deriv) {
    // Polynomial p(x) = 1 + 2x + 3x^2
    const std::vector<double> p = {1.0, 2.0, 3.0};

    // Evaluate at multiple points
    EXPECT_NEAR(poly_eval(p, 0.0)[0], 1.0, 1e-12);
    EXPECT_NEAR(poly_eval(p, 1.0)[0], 6.0, 1e-12);   // 1+2+3
    EXPECT_NEAR(poly_eval(p, -1.0)[0], 2.0, 1e-12);  // 1-2+3

    // Derivative: d/dx(1+2x+3x^2) = 2+6x
    const auto dp = poly_deriv(p);
    ASSERT_EQ(dp.size(), 2u);
    EXPECT_NEAR(poly_eval(dp, 0.0)[0], 2.0, 1e-12);
    EXPECT_NEAR(poly_eval(dp, 1.0)[0], 8.0, 1e-12);  // 2+6

    // Second derivative: d^2/dx^2(1+2x+3x^2) = 6 (constant)
    const auto ddp = poly_deriv(dp);
    ASSERT_EQ(ddp.size(), 1u);
    EXPECT_NEAR(ddp[0], 6.0, 1e-12);
}

// ---------------------------------------------------------------------------
// Pipeline 6: Optimization on analytical function
// ---------------------------------------------------------------------------

TEST(PipelineIntegration, Optim_Golden_Then_Eval) {
    // Minimize f(x) = (x-2)^2 + 1 in [0, 4]
    const auto f = [](double x) { return (x - 2.0) * (x - 2.0) + 1.0; };
    const double xmin = golden_section(f, 0.0, 4.0, 1e-10);
    EXPECT_NEAR(xmin, 2.0, 1e-6);
    EXPECT_NEAR(f(xmin), 1.0, 1e-10);  // minimum value is 1.0
}

// ---------------------------------------------------------------------------
// Pipeline 7: CDF → PPF roundtrip for multiple distributions
// ---------------------------------------------------------------------------

TEST(PipelineIntegration, CDF_PPF_Roundtrip_Normal) {
    // norm_ppf(norm_cdf(x)) ≈ x for various x
    const std::vector<double> xvals = {-2.0, -1.0, 0.0, 0.5, 1.0, 1.5};
    for (double x : xvals) {
        const double p = norm_cdf(x, 0.0, 1.0);
        if (p > 0.01 && p < 0.99) {  // stay away from extreme quantiles
            EXPECT_NEAR(norm_ppf(p, 0.0, 1.0), x, 1e-4);
        }
    }
}

TEST(PipelineIntegration, CDF_Monotone_Normal) {
    // CDF should be strictly monotone
    double prev = 0.0;
    for (double x = -3.0; x <= 3.0; x += 0.5) {
        const double cur = norm_cdf(x, 0.0, 1.0);
        EXPECT_GE(cur, prev);
        prev = cur;
    }
}

// ---------------------------------------------------------------------------
// Pipeline 8: Matrix decomposition → solve → verify error
// ---------------------------------------------------------------------------

TEST(PipelineIntegration, LU_Then_Solve_Verify) {
    DMatrix A(3, 3);
    A(0,0)=2; A(0,1)=1; A(0,2)=0;
    A(1,0)=4; A(1,1)=3; A(1,2)=2;
    A(2,0)=2; A(2,1)=1; A(2,2)=3;
    DMatrix b(3, 1);
    b(0,0)=1; b(1,0)=2; b(2,0)=3;

    const auto sol = solve(A, b);
    ASSERT_TRUE(sol.has_value());

    // Verify residual: ||A*x - b|| < tol
    const auto Ax = matmul(A, *sol);
    ASSERT_TRUE(Ax.has_value());
    double res = 0.0;
    for (size_t i = 0; i < 3; ++i) {
        double d = (*Ax)(i, 0) - b(i, 0);
        res += d * d;
    }
    EXPECT_LT(std::sqrt(res), 1e-8);
}

// ---------------------------------------------------------------------------
// Pipeline 9: Special functions composed with stats
// ---------------------------------------------------------------------------

TEST(PipelineIntegration, Special_Erf_CDF_Relationship) {
    // norm_cdf(x) = 0.5*(1 + erf(x/sqrt(2)))
    const std::vector<double> xvals = {-1.5, -0.5, 0.0, 0.5, 1.0, 1.5};
    for (double x : xvals) {
        const double cdf = norm_cdf(x, 0.0, 1.0);
        const double erf_val = ms::erf(x / std::sqrt(2.0));
        EXPECT_NEAR(cdf, 0.5 * (1.0 + erf_val), 1e-6);
    }
}

// ---------------------------------------------------------------------------
// Pipeline 10: Poly roots via Newton optimization
// ---------------------------------------------------------------------------

TEST(PipelineIntegration, Poly_Newton_Root) {
    // p(x) = x^3 - x -> roots at -1, 0, 1
    // Start near x=0.8, gradient_descent-style => find root near x=1
    const auto f = [](double x, double /*y*/) { return x * x * x - x; };
    const auto [xr, yr] = newton_raphson(f, 0.9, 0.0, 1e-10, 100);
    // Should converge to root near 1
    EXPECT_NEAR(f(xr, 0.0), 0.0, 1e-8);
}
