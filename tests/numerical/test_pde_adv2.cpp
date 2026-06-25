// MathScript Advanced PDE Tests (Wave 51)
// Additional heat-equation tests: energy, symmetry, stability, grid refinement.

#define _USE_MATH_DEFINES
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>

#include "ms/pde/pde.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ms;

// Helper: sum of all values in the final time slice
static double sum_final(const Heat1DResult& r) {
    if (r.u.empty()) return 0.0;
    double s = 0.0;
    for (double v : r.u.back()) s += v;
    return s;
}

// Helper: max value in the final time slice
static double max_final(const Heat1DResult& r) {
    if (r.u.empty()) return 0.0;
    double m = r.u.back()[0];
    for (double v : r.u.back()) m = std::max(m, v);
    return m;
}

// ---------------------------------------------------------------------------
// Energy dissipation
// ---------------------------------------------------------------------------

TEST(PdeAdv2, HeatEq_Energy_Decreases_FixedBoundary) {
    // With constant-zero boundary conditions at endpoints, total energy (sum) should decay
    const size_t Nx = 11;
    std::vector<double> x0(Nx, 1.0);
    x0[0] = 0.0;
    x0[Nx - 1] = 0.0;
    // alpha*dt/dx^2 = 0.1*0.001/0.01 = 0.01 < 0.5 (stable)
    auto result = pde_heat_1d(x0, 0.1, 0.1, 0.001, 500);
    ASSERT_FALSE(result.u.empty());
    double sum_init = 0.0;
    for (double v : result.u.front()) sum_init += v;
    double sum_end  = sum_final(result);
    // Energy decreases because boundaries are zero (Dirichlet BCs)
    EXPECT_LT(sum_end, sum_init)
        << "Energy should dissipate with zero Dirichlet BCs";
}

TEST(PdeAdv2, HeatEq_MaxValue_DecreasesOverTime) {
    const size_t Nx = 13;
    std::vector<double> x0(Nx, 0.0);
    x0[Nx / 2] = 5.0; // Spike at center
    auto result = pde_heat_1d(x0, 0.1, 0.1, 0.001, 200);
    ASSERT_FALSE(result.u.empty());
    double max_init = 0.0;
    for (double v : result.u.front()) max_init = std::max(max_init, v);
    EXPECT_LT(max_final(result), max_init)
        << "Max value should decrease as spike diffuses";
}

// ---------------------------------------------------------------------------
// Symmetry preservation
// ---------------------------------------------------------------------------

TEST(PdeAdv2, HeatEq_SymmetricIC_StaysSymmetric) {
    // Symmetric IC about center → solution stays symmetric
    const size_t Nx = 11;
    std::vector<double> x0(Nx, 0.0);
    // Symmetric: x0[i] = x0[Nx-1-i]
    for (size_t i = 0; i < Nx; ++i)
        x0[i] = std::sin(M_PI * static_cast<double>(i) / (Nx - 1));
    auto result = pde_heat_1d(x0, 0.05, 0.1, 0.0005, 100);
    ASSERT_FALSE(result.u.empty());
    const auto& u_last = result.u.back();
    for (size_t i = 0; i < Nx / 2; ++i)
        EXPECT_NEAR(u_last[i], u_last[Nx - 1 - i], 1e-7)
            << "Symmetry broken at i=" << i;
}

// ---------------------------------------------------------------------------
// Stability: CFL condition alpha*dt/dx^2 <= 0.5
// ---------------------------------------------------------------------------

TEST(PdeAdv2, HeatEq_Stable_CFL_AllFinite) {
    // Stable: alpha=0.1, dt=0.001, dx=0.1 → r=0.01 < 0.5
    const size_t Nx = 11;
    std::vector<double> x0(Nx, 1.0);
    auto result = pde_heat_1d(x0, 0.1, 0.1, 0.001, 100);
    for (const auto& row : result.u)
        for (double v : row)
            EXPECT_TRUE(std::isfinite(v)) << "NaN/inf in stable heat equation";
}

TEST(PdeAdv2, HeatEq_VerySmallDt_FiniteOutput) {
    // Very small dt, very few steps
    const size_t Nx = 7;
    std::vector<double> x0(Nx, 0.5);
    auto result = pde_heat_1d(x0, 0.01, 0.1, 0.0001, 20);
    ASSERT_FALSE(result.u.empty());
    for (const auto& row : result.u)
        for (double v : row)
            EXPECT_TRUE(std::isfinite(v));
}

// ---------------------------------------------------------------------------
// Diffusivity comparison: larger alpha = faster diffusion
// ---------------------------------------------------------------------------

TEST(PdeAdv2, HeatEq_LargerAlpha_FasterDiffusion) {
    const size_t Nx = 11;
    std::vector<double> x0(Nx, 0.0);
    x0[Nx / 2] = 1.0;
    // Same stable time step for both
    auto slow = pde_heat_1d(x0, 0.01, 0.1, 0.001, 50);
    auto fast  = pde_heat_1d(x0, 0.1,  0.1, 0.001, 50);
    ASSERT_FALSE(slow.u.empty());
    ASSERT_FALSE(fast.u.empty());
    // Larger alpha → smaller max value (more diffused) in final step
    EXPECT_LT(max_final(fast), max_final(slow))
        << "Larger alpha should diffuse faster";
}

// ---------------------------------------------------------------------------
// Zero initial condition stays zero
// ---------------------------------------------------------------------------

TEST(PdeAdv2, HeatEq_ZeroIC_RemainsZero_ManySteps) {
    const size_t Nx = 9;
    std::vector<double> x0(Nx, 0.0);
    auto result = pde_heat_1d(x0, 0.1, 0.1, 0.001, 500);
    for (const auto& row : result.u)
        for (double v : row)
            EXPECT_NEAR(v, 0.0, 1e-12);
}

// ---------------------------------------------------------------------------
// Spatial grid properties
// ---------------------------------------------------------------------------

TEST(PdeAdv2, HeatEq_SpatialGrid_CorrectSize) {
    const size_t Nx = 15;
    std::vector<double> x0(Nx, 0.5);
    auto result = pde_heat_1d(x0, 0.1, 0.1, 0.001, 10);
    EXPECT_EQ(result.x.size(), Nx);
}

TEST(PdeAdv2, HeatEq_SpatialGrid_UniformSpacing) {
    const size_t Nx = 9;
    double dx = 0.1;
    std::vector<double> x0(Nx, 0.5);
    auto result = pde_heat_1d(x0, 0.1, dx, 0.001, 1);
    ASSERT_GE(result.x.size(), 2u);
    for (size_t i = 1; i < result.x.size(); ++i)
        EXPECT_NEAR(result.x[i] - result.x[i - 1], dx, 1e-10);
}

// ---------------------------------------------------------------------------
// Time grid properties
// ---------------------------------------------------------------------------

TEST(PdeAdv2, HeatEq_TimeGrid_Monotone) {
    std::vector<double> x0(7, 1.0);
    auto result = pde_heat_1d(x0, 0.1, 0.1, 0.01, 30);
    for (size_t i = 1; i < result.t.size(); ++i)
        EXPECT_GE(result.t[i], result.t[i - 1]);
}

TEST(PdeAdv2, HeatEq_TimeGrid_NonEmpty) {
    std::vector<double> x0(5, 1.0);
    auto result = pde_heat_1d(x0, 0.1, 0.1, 0.01, 5);
    EXPECT_FALSE(result.t.empty());
    EXPECT_FALSE(result.u.empty());
}

// ---------------------------------------------------------------------------
// Values remain bounded (no blow-up in stable regime)
// ---------------------------------------------------------------------------

TEST(PdeAdv2, HeatEq_ValuesRemainBounded) {
    const size_t Nx = 11;
    std::vector<double> x0(Nx, 2.0);
    // Max of IC is 2.0; with Neumann-like boundaries values shouldn't exceed IC
    auto result = pde_heat_1d(x0, 0.05, 0.1, 0.001, 200);
    for (const auto& row : result.u)
        for (double v : row)
            EXPECT_LE(v, 2.0 + 1e-9) << "Heat equation solution should not exceed initial max";
}
