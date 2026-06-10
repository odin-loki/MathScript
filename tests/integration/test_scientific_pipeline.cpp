#include <cmath>
#include <gtest/gtest.h>
#include <numeric>
#include <vector>

#include "ms/core/operations.hpp"
#include "ms/fft/fft.hpp"
#include "ms/linalg/linalg.hpp"
#include "ms/ode/ode.hpp"
#include "ms/signal/signal.hpp"
#include "ms/special/special.hpp"
#include "ms/stats/stats.hpp"

using namespace ms;
using DMatrix = ColMatrix<double>;

// ---------------------------------------------------------------------------
// ODE pipeline: solve exponential decay, verify accuracy
// ---------------------------------------------------------------------------
TEST(ScientificPipelineTest, ode_euler_exponential_decay) {
    const double lambda = 1.0;
    auto f = [&](double /*t*/, double y) { return -lambda * y; };
    const auto result = ode_euler(f, 0.0, 1.0, 5.0, 500);
    ASSERT_FALSE(result.t.empty());
    ASSERT_FALSE(result.y.empty());
    const double y_end = result.y.back();
    const double y_exact = std::exp(-lambda * 5.0);
    EXPECT_NEAR(y_end, y_exact, 0.05);  // Euler with 500 steps: ~1% accuracy
}

TEST(ScientificPipelineTest, ode_rk4_exponential_decay) {
    const double lambda = 1.0;
    auto f = [&](double /*t*/, double y) { return -lambda * y; };
    const auto result = ode_rk4(f, 0.0, 1.0, 5.0, 100);
    ASSERT_FALSE(result.y.empty());
    const double y_end = result.y.back();
    const double y_exact = std::exp(-lambda * 5.0);
    EXPECT_NEAR(y_end, y_exact, 1e-5);  // RK4 is 4th order accurate
}

// ---------------------------------------------------------------------------
// Signal processing pipeline: tone -> FFT -> verify dominant frequency
// ---------------------------------------------------------------------------
TEST(ScientificPipelineTest, fft_detects_dominant_frequency) {
    constexpr size_t N = 128;
    constexpr double fs = 128.0;  // Hz
    constexpr double f0 = 10.0;   // signal frequency

    std::vector<double> x(N);
    for (size_t i = 0; i < N; ++i) {
        x[i] = std::cos(2.0 * 3.14159265358979323846 * f0 * i / fs);
    }

    const auto X_result = fft(x);
    ASSERT_TRUE(X_result.has_value());
    const auto& X = *X_result;
    ASSERT_EQ(X.size(), N);

    // Find peak magnitude (ignore DC)
    double peak_mag = 0.0;
    size_t peak_idx = 0;
    for (size_t i = 1; i < N / 2; ++i) {
        const double mag = std::abs(X[i]);
        if (mag > peak_mag) { peak_mag = mag; peak_idx = i; }
    }
    // Peak should be at bin 10 (f0 = 10 Hz, fs = 128 Hz, N = 128)
    EXPECT_EQ(peak_idx, 10u);
}

// ---------------------------------------------------------------------------
// Linear algebra pipeline: full solve and verify residual
// ---------------------------------------------------------------------------
TEST(ScientificPipelineTest, linalg_solve_and_residual) {
    // Hilbert-like 3x3 system: use simple system A x = b
    DMatrix A(3, 3);
    A(0,0) = 6; A(0,1) = 2; A(0,2) = 1;
    A(1,0) = 2; A(1,1) = 5; A(1,2) = 3;
    A(2,0) = 1; A(2,1) = 3; A(2,2) = 7;
    DMatrix b(3, 1);
    b(0,0) = 9; b(1,0) = 10; b(2,0) = 11;

    const auto result = solve(A, b);
    ASSERT_TRUE(result.has_value());

    // Compute residual: r = A*x - b
    const auto& x = *result;
    for (size_t i = 0; i < 3; ++i) {
        double ax_i = 0.0;
        for (size_t j = 0; j < 3; ++j) ax_i += A(i,j) * x(j,0);
        EXPECT_NEAR(ax_i, b(i,0), 1e-9);
    }
}

// ---------------------------------------------------------------------------
// Statistical pipeline: generate data, compute stats, verify
// ---------------------------------------------------------------------------
TEST(ScientificPipelineTest, stats_pipeline_on_linear_sequence) {
    const size_t n = 100;
    std::vector<double> data(n);
    std::iota(data.begin(), data.end(), 1.0);  // [1, 2, ..., 100]

    const double m = mean(data);
    const double s = stddev(data);
    const double med = median(data);

    EXPECT_NEAR(m, 50.5, 1e-10);
    EXPECT_NEAR(med, 50.5, 1e-10);
    EXPECT_GT(s, 28.0);
    EXPECT_LT(s, 30.0);  // stddev of 1..100 ≈ 28.87
}

// ---------------------------------------------------------------------------
// Special functions pipeline: verify erf properties
// ---------------------------------------------------------------------------
TEST(ScientificPipelineTest, erf_gaussian_cdf_relation) {
    // Normal CDF: Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
    for (double x : {-2.0, -1.0, 0.0, 1.0, 2.0}) {
        const double phi_x = 0.5 * (1.0 + ms::erf(x / std::sqrt(2.0)));
        EXPECT_TRUE(std::isfinite(phi_x));
        EXPECT_GE(phi_x, 0.0);
        EXPECT_LE(phi_x, 1.0);
    }
    // erf(0) = 0
    EXPECT_NEAR(ms::erf(0.0), 0.0, 1e-12);
    // erf(-x) = -erf(x)
    EXPECT_NEAR(ms::erf(-1.0), -ms::erf(1.0), 1e-12);
}
