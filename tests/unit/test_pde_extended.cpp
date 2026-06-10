#include <gtest/gtest.h>
#include <cmath>
#include <numeric>
#include <vector>

#include "ms/pde/pde.hpp"

using namespace ms;

TEST(PdeExtTest, heat_1d_zero_initial_stays_zero) {
    // All-zero IC stays zero
    const std::vector<double> x0(5, 0.0);
    const auto result = pde_heat_1d(x0, 0.1, 0.1, 0.01, 10);
    ASSERT_FALSE(result.u.empty());
    for (const auto& row : result.u) {
        for (double v : row) {
            EXPECT_NEAR(v, 0.0, 1e-12);
        }
    }
}

TEST(PdeExtTest, heat_1d_output_structure) {
    const size_t nx = 9;
    std::vector<double> x0(nx, 0.0);
    x0[4] = 1.0;  // delta function at center

    const auto result = pde_heat_1d(x0, 0.1, 0.1, 0.01, 20);

    EXPECT_EQ(result.x.size(), nx);
    EXPECT_FALSE(result.t.empty());
    EXPECT_FALSE(result.u.empty());
    // Final time step should have finite values
    for (double v : result.u.back()) {
        EXPECT_TRUE(std::isfinite(v));
    }
}

TEST(PdeExtTest, heat_1d_monotone_spread) {
    // Delta IC: initial max at center, later timestep should have smaller max
    const size_t nx = 9;
    std::vector<double> x0(nx, 0.0);
    x0[4] = 10.0;

    const auto result = pde_heat_1d(x0, 0.1, 0.1, 0.001, 50);
    ASSERT_FALSE(result.u.empty());

    const auto& last = result.u.back();
    const double max_last = *std::max_element(last.begin(), last.end());
    EXPECT_LT(max_last, 10.0);  // peak should have decreased
    EXPECT_GT(max_last, 0.0);   // but not gone to zero entirely
}

TEST(PdeExtTest, heat_1d_constant_ic_stays_constant) {
    // Constant IC: each row should remain the same (heat equation with uniform IC)
    const size_t nx = 5;
    const std::vector<double> x0(nx, 2.0);
    const auto result = pde_heat_1d(x0, 0.01, 0.1, 0.001, 10);
    ASSERT_FALSE(result.u.empty());
    for (const auto& row : result.u) {
        for (double v : row) {
            EXPECT_NEAR(v, 2.0, 1e-6);
        }
    }
}

TEST(PdeExtTest, heat_1d_time_vector_monotone) {
    const std::vector<double> x0(5, 1.0);
    const auto result = pde_heat_1d(x0, 0.1, 0.1, 0.01, 5);
    ASSERT_GE(result.t.size(), 2u);
    for (size_t i = 1; i < result.t.size(); ++i) {
        EXPECT_GT(result.t[i], result.t[i-1]);
    }
}
