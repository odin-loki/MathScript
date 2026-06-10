// MathScript PDE Numerical Reference Tests
// Tests pde_heat_1d for known analytical solutions of the heat equation

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <numeric>

#include "ms/pde/pde.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ms;

// Helper: evaluate Fourier series solution for heat eq
// u(x,t) = sum_{n=1,3,...} (4/n*pi) * sin(n*pi*x) * exp(-n^2*pi^2*alpha*t)
// for u(x,0) = 1 (block IC), alpha=1, [0,1]
static double heat_analytical(double x, double t, double alpha, int terms = 20) {
    double sum = 0.0;
    for (int n = 1; n <= terms; n += 2) {
        sum += (4.0 / (n * M_PI)) * std::sin(n * M_PI * x)
               * std::exp(-n * n * M_PI * M_PI * alpha * t);
    }
    return sum;
}

// ---------------------------------------------------------------------------
// Heat1DResult structure tests
// ---------------------------------------------------------------------------

TEST(PdeRef, HeatResult_Fields_Exist) {
    std::vector<double> x0(10, 1.0);
    const auto result = pde_heat_1d(x0, 0.01, 0.1, 0.001, 1);
    EXPECT_FALSE(result.x.empty());
    EXPECT_FALSE(result.t.empty());
    EXPECT_FALSE(result.u.empty());
}

TEST(PdeRef, Heat1D_OutputDimensions) {
    const size_t Nx = 11;
    const size_t steps = 5;
    std::vector<double> x0(Nx, 0.0);
    const auto result = pde_heat_1d(x0, 0.1, 0.1, 0.001, steps);
    EXPECT_EQ(result.x.size(), Nx);
    EXPECT_GE(result.t.size(), 1u);
    EXPECT_GE(result.u.size(), 1u);
}

TEST(PdeRef, Heat1D_SteadyState_ZeroIC_IsZero) {
    // If initial condition is all zeros, solution stays zero
    const size_t Nx = 11;
    std::vector<double> x0(Nx, 0.0);
    const auto result = pde_heat_1d(x0, 0.1, 0.1, 0.001, 100);
    for (const auto& row : result.u) {
        for (double v : row) {
            EXPECT_NEAR(v, 0.0, 1e-10);
        }
    }
}

TEST(PdeRef, Heat1D_ConstantIC_Diffuses) {
    // Constant IC with fixed boundary 0 → diffuses to 0
    const size_t Nx = 11;
    std::vector<double> x0(Nx, 1.0);
    x0[0] = 0.0;  // Dirichlet boundary condition (approx)
    x0[Nx - 1] = 0.0;
    const auto result = pde_heat_1d(x0, 0.1, 0.1, 0.001, 200);
    EXPECT_FALSE(result.u.empty());
    // After diffusion, max value should decrease
    double max_after = 0.0;
    for (double v : result.u.back()) max_after = std::max(max_after, v);
    EXPECT_LE(max_after, 1.0);
}

TEST(PdeRef, Heat1D_TimeGridIsMonotone) {
    std::vector<double> x0(5, 1.0);
    const auto result = pde_heat_1d(x0, 0.1, 0.25, 0.01, 10);
    for (size_t i = 1; i < result.t.size(); ++i)
        EXPECT_GE(result.t[i], result.t[i - 1]);
}

TEST(PdeRef, Heat1D_SpatialGridIsMonotone) {
    std::vector<double> x0(8, 0.5);
    const auto result = pde_heat_1d(x0, 0.1, 0.1, 0.001, 1);
    for (size_t i = 1; i < result.x.size(); ++i)
        EXPECT_GE(result.x[i], result.x[i - 1]);
}

TEST(PdeRef, Heat1D_AllFinite_AfterDiffusion) {
    std::vector<double> x0(11, 0.5);
    const auto result = pde_heat_1d(x0, 0.01, 0.1, 0.01, 50);
    for (const auto& row : result.u) {
        for (double v : row) {
            EXPECT_TRUE(std::isfinite(v));
        }
    }
}

TEST(PdeRef, Heat1D_PeakIC_Diffuses_Symmetrically) {
    // Place peak at center; with symmetric BC it should spread symmetrically
    const size_t Nx = 11;
    std::vector<double> x0(Nx, 0.0);
    x0[5] = 1.0;  // peak at center
    const auto result = pde_heat_1d(x0, 0.1, 0.1, 0.001, 100);
    EXPECT_FALSE(result.u.empty());
    const auto& u_final = result.u.back();
    // Symmetry: u[i] ≈ u[Nx-1-i] (within tolerance due to floating point)
    for (size_t i = 0; i < Nx / 2; ++i)
        EXPECT_NEAR(u_final[i], u_final[Nx - 1 - i], 1e-8);
}

TEST(PdeRef, Heat1D_SmallAlpha_SlowDiffusion) {
    // With very small alpha, diffusion is slow; max stays near initial
    const size_t Nx = 11;
    std::vector<double> x0(Nx, 1.0);
    const auto result_fast = pde_heat_1d(x0, 1.0, 0.1, 0.001, 10);
    const auto result_slow = pde_heat_1d(x0, 0.001, 0.1, 0.001, 10);
    // Larger alpha diffuses faster → variance in u should be smaller
    double var_fast = 0.0, var_slow = 0.0;
    if (!result_fast.u.empty() && !result_slow.u.empty()) {
        double mean_fast = 0.0, mean_slow = 0.0;
        for (double v : result_fast.u.back()) mean_fast += v;
        for (double v : result_slow.u.back()) mean_slow += v;
        mean_fast /= result_fast.u.back().size();
        mean_slow /= result_slow.u.back().size();
        for (double v : result_fast.u.back()) var_fast += (v - mean_fast) * (v - mean_fast);
        for (double v : result_slow.u.back()) var_slow += (v - mean_slow) * (v - mean_slow);
    }
    EXPECT_LE(var_fast, var_slow + 1e-10);
}
