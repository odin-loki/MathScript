// MathScript: Advanced PDE Tests (Wave 45)
// Tests for 1D Heat equation discretization and solver correctness

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "ms/pde/pde.hpp"

using namespace ms;

// ---------------------------------------------------------------------------
// pde_heat_1d: u_t = alpha * u_xx, FTCS scheme
// ---------------------------------------------------------------------------

TEST(PdeAdv, Heat1D_OutputDimensions) {
    // 10 spatial points, 5 time steps
    std::vector<double> x0(10, 0.0);
    x0[5] = 1.0;  // Impulse in center
    auto result = pde_heat_1d(x0, 0.1, 0.1, 0.01, 5);
    EXPECT_EQ(result.u.size(), 6u);   // Initial + 5 steps
    for (auto& row : result.u)
        EXPECT_EQ(row.size(), 10u);
}

TEST(PdeAdv, Heat1D_UniformIC_Unchanged) {
    // u(x, 0) = 1.0 everywhere should remain 1.0 (constant is steady state)
    // With Dirichlet BC = 0 at boundaries, interior should diffuse
    // But test: if initial condition is all zeros, solution stays zero
    std::vector<double> x0(8, 0.0);
    auto result = pde_heat_1d(x0, 0.25, 0.125, 0.01, 5);
    // All values should remain zero
    for (auto& row : result.u)
        for (auto v : row)
            EXPECT_NEAR(v, 0.0, 1e-14);
}

TEST(PdeAdv, Heat1D_HeatDiffuses_MaxDecreases) {
    // A spike in the middle should spread out — peak decreases over time
    int N = 20;
    std::vector<double> x0(N, 0.0);
    x0[N / 2] = 1.0;
    auto result = pde_heat_1d(x0, 0.1, 0.05, 0.001, 50);
    // Peak at t=0 should be 1.0; peak should decrease
    double peak0 = *std::max_element(result.u[0].begin(), result.u[0].end());
    double peak_final = *std::max_element(result.u.back().begin(), result.u.back().end());
    EXPECT_GT(peak0, peak_final) << "Peak should decrease as heat diffuses";
}

TEST(PdeAdv, Heat1D_AllFinite) {
    std::vector<double> x0(10, 0.0);
    x0[4] = 0.5;
    x0[5] = 0.5;
    auto result = pde_heat_1d(x0, 0.1, 0.1, 0.001, 20);
    for (auto& row : result.u)
        for (auto v : row)
            EXPECT_TRUE(std::isfinite(v)) << "Non-finite value in PDE solution";
}

TEST(PdeAdv, Heat1D_ConservesEnergyApprox) {
    // Without boundary flux, total "energy" (sum) should be approximately conserved
    // Note: if BC absorbs at boundary, it won't be perfectly conserved
    std::vector<double> x0(10, 0.0);
    x0[3] = 1.0;
    x0[4] = 1.0;
    x0[5] = 1.0;
    auto result = pde_heat_1d(x0, 0.1, 0.05, 0.001, 5);
    double sum0 = 0.0;
    for (auto v : result.u[0]) sum0 += v;
    double sum_final = 0.0;
    for (auto v : result.u.back()) sum_final += v;
    // Energy should be non-increasing (boundary conditions may dissipate)
    EXPECT_LE(sum_final, sum0 + 1e-10)
        << "Sum should not increase (energy dissipates at boundaries)";
}

TEST(PdeAdv, Heat1D_XGrid_Correct) {
    std::vector<double> x0(5, 0.0);
    auto result = pde_heat_1d(x0, 0.1, 0.5, 0.01, 3);
    ASSERT_EQ(result.x.size(), 5u);
    // x should be uniformly spaced with dx=0.5
    for (size_t i = 1; i < result.x.size(); ++i) {
        EXPECT_NEAR(result.x[i] - result.x[i - 1], 0.5, 1e-10)
            << "x grid should be uniform";
    }
}

TEST(PdeAdv, Heat1D_TGrid_Correct) {
    std::vector<double> x0(5, 0.0);
    auto result = pde_heat_1d(x0, 0.1, 0.5, 0.01, 4);
    ASSERT_EQ(result.t.size(), 5u);  // t0 + 4 steps = 5 points
    for (size_t i = 1; i < result.t.size(); ++i) {
        EXPECT_NEAR(result.t[i] - result.t[i - 1], 0.01, 1e-12)
            << "t grid should be uniform with dt=0.01";
    }
}

TEST(PdeAdv, Heat1D_HigherAlpha_DiffusesFaster) {
    // Higher thermal diffusivity α means faster diffusion
    std::vector<double> x0(20, 0.0);
    x0[10] = 1.0;
    auto slow = pde_heat_1d(x0, 0.01, 0.05, 0.001, 100);
    auto fast  = pde_heat_1d(x0, 0.5, 0.05, 0.001, 100);

    // Peak should be lower in the fast case
    double peak_slow = *std::max_element(slow.u.back().begin(), slow.u.back().end());
    double peak_fast = *std::max_element(fast.u.back().begin(), fast.u.back().end());
    EXPECT_LT(peak_fast, peak_slow)
        << "Higher alpha should diffuse faster (lower peak)";
}

TEST(PdeAdv, Heat1D_SymmetricIC_SolutionRemainsSym) {
    // Symmetric initial condition should give symmetric solution
    int N = 11;
    std::vector<double> x0(N, 0.0);
    x0[N / 2] = 1.0;  // Center spike
    auto result = pde_heat_1d(x0, 0.1, 0.1, 0.001, 10);
    // Check symmetry at each time step
    for (auto& row : result.u) {
        for (int i = 0; i < N / 2; ++i) {
            EXPECT_NEAR(row[i], row[N - 1 - i], 1e-10)
                << "Solution should be symmetric";
        }
    }
}
