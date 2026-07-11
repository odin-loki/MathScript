#include <gtest/gtest.h>
#include <cmath>
#include <numeric>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "ms/pde/pde.hpp"

using namespace ms;

namespace {

std::vector<std::vector<double>> make_grid(std::size_t ny, std::size_t nx, double fill = 0.0) {
    return std::vector<std::vector<double>>(ny, std::vector<double>(nx, fill));
}

double grid_max(const std::vector<std::vector<double>>& g) {
    double m = 0.0;
    for (const auto& row : g) {
        for (double v : row) {
            m = std::max(m, v);
        }
    }
    return m;
}

} // namespace

// ---------------------------------------------------------------------------
// pde_heat_2d
// ---------------------------------------------------------------------------

TEST(PdeExtTest, heat_2d_spike_smooths) {
    auto u0 = make_grid(7, 7);
    u0[3][3] = 10.0;
    const auto result = pde_heat_2d(u0, 0.1, 0.1, 0.1, 0.001, 50);
    ASSERT_FALSE(result.u.empty());
    EXPECT_LT(grid_max(result.u.back()), 10.0);
    EXPECT_GT(grid_max(result.u.back()), 0.0);
}

TEST(PdeExtTest, heat_2d_dirichlet_boundaries_zero) {
    auto u0 = make_grid(5, 5);
    u0[2][2] = 1.0;
    const auto result = pde_heat_2d(u0, 0.05, 0.1, 0.1, 0.001, 20);
    ASSERT_FALSE(result.u.empty());
    const auto& last = result.u.back();
    for (std::size_t i = 0; i < last[0].size(); ++i) {
        EXPECT_NEAR(last[0][i], 0.0, 1e-12);
        EXPECT_NEAR(last.back()[i], 0.0, 1e-12);
    }
    for (const auto& row : last) {
        EXPECT_NEAR(row.front(), 0.0, 1e-12);
        EXPECT_NEAR(row.back(), 0.0, 1e-12);
    }
}

TEST(PdeExtTest, heat_2d_stability_rejection) {
    auto u0 = make_grid(5, 5, 1.0);
    const auto result = pde_heat_2d(u0, 1.0, 0.1, 0.1, 0.1, 5);
    EXPECT_TRUE(result.u.empty());
    EXPECT_TRUE(result.t.empty());
}

TEST(PdeExtTest, heat_2d_too_small_grid) {
    auto u0 = make_grid(2, 2, 1.0);
    const auto result = pde_heat_2d(u0, 0.1, 0.1, 0.1, 0.001, 5);
    EXPECT_TRUE(result.u.empty());
}

TEST(PdeExtTest, heat_2d_zero_steps) {
    auto u0 = make_grid(5, 5, 1.0);
    const auto result = pde_heat_2d(u0, 0.1, 0.1, 0.1, 0.001, 0);
    EXPECT_TRUE(result.u.empty());
}

// ---------------------------------------------------------------------------
// pde_wave_1d
// ---------------------------------------------------------------------------

TEST(PdeExtTest, wave_1d_energy_roughly_preserved) {
    const std::size_t n = 21;
    const double dx = 0.05;
    const double c = 1.0;
    const double dt = 0.01;
    std::vector<double> u0(n, 0.0);
    std::vector<double> v0(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        u0[i] = std::sin(M_PI * static_cast<double>(i) / static_cast<double>(n - 1));
    }
    const auto result = pde_wave_1d(u0, v0, c, dx, dt, 5);
    ASSERT_GE(result.u.size(), 2u);
    auto l2 = [](const std::vector<double>& u) {
        double s = 0.0;
        for (double v : u) {
            s += v * v;
        }
        return std::sqrt(s);
    };
    const double norm0 = l2(result.u.front());
    const double norm1 = l2(result.u.back());
    EXPECT_NEAR(norm0, norm1, 0.15 * norm0);
}

TEST(PdeExtTest, wave_1d_cfl_rejection) {
    const std::vector<double> u0(7, 0.0);
    const std::vector<double> v0(7, 0.0);
    std::vector<double> u0_mut = u0;
    u0_mut[3] = 1.0;
    const auto result = pde_wave_1d(u0_mut, v0, 2.0, 0.1, 0.1, 5);
    EXPECT_TRUE(result.u.empty());
}

TEST(PdeExtTest, wave_1d_fixed_boundaries) {
    std::vector<double> u0(9, 0.0);
    std::vector<double> v0(9, 0.0);
    u0[4] = 1.0;
    const auto result = pde_wave_1d(u0, v0, 1.0, 0.1, 0.05, 10);
    ASSERT_FALSE(result.u.empty());
    for (const auto& snap : result.u) {
        EXPECT_NEAR(snap.front(), 0.0, 1e-12);
        EXPECT_NEAR(snap.back(), 0.0, 1e-12);
    }
}

TEST(PdeExtTest, wave_1d_finite_output) {
    std::vector<double> u0(11, 0.0);
    std::vector<double> v0(11, 0.0);
    u0[5] = 0.5;
    v0[5] = 0.1;
    const auto result = pde_wave_1d(u0, v0, 1.0, 0.1, 0.05, 8);
    ASSERT_EQ(result.u.size(), 9u);
    for (const auto& snap : result.u) {
        for (double v : snap) {
            EXPECT_TRUE(std::isfinite(v));
        }
    }
}

TEST(PdeExtTest, wave_1d_zero_steps) {
    const std::vector<double> u0(5, 1.0);
    const std::vector<double> v0(5, 0.0);
    const auto result = pde_wave_1d(u0, v0, 1.0, 0.1, 0.01, 0);
    EXPECT_TRUE(result.u.empty());
}

// ---------------------------------------------------------------------------
// pde_advection_1d
// ---------------------------------------------------------------------------

TEST(PdeExtTest, advection_1d_shifts_right) {
    const std::size_t n = 20;
    std::vector<double> u0(n, 0.0);
    u0[3] = 1.0;
    const double v = 1.0;
    const double dx = 0.1;
    const double dt = 0.05;
    const std::size_t steps = 4;
    const auto result = pde_advection_1d(u0, v, dx, dt, steps);
    ASSERT_FALSE(result.u.empty());
    const std::size_t expected_shift = static_cast<std::size_t>(std::lround(v * dt * steps / dx));
    const std::size_t peak_idx =
        static_cast<std::size_t>(std::distance(result.u.back().begin(),
            std::max_element(result.u.back().begin(), result.u.back().end())));
    EXPECT_EQ(peak_idx, (3u + expected_shift) % n);
}

TEST(PdeExtTest, advection_1d_periodic_wrap) {
    const std::size_t n = 10;
    std::vector<double> u0(n, 0.0);
    u0[0] = 1.0;
    const double v = -1.0;
    const double dx = 1.0;
    const double dt = 1.0;
    const auto result = pde_advection_1d(u0, v, dx, dt, 1);
    ASSERT_FALSE(result.u.empty());
    const std::size_t peak_idx =
        static_cast<std::size_t>(std::distance(result.u.back().begin(),
            std::max_element(result.u.back().begin(), result.u.back().end())));
    EXPECT_EQ(peak_idx, 9u);
}

TEST(PdeExtTest, advection_1d_cfl_rejection) {
    const std::vector<double> u0(8, 0.0);
    const auto result = pde_advection_1d(u0, 2.0, 0.1, 0.1, 5);
    EXPECT_TRUE(result.u.empty());
}

TEST(PdeExtTest, advection_1d_zero_velocity_unchanged) {
    const std::vector<double> u0 = {0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0};
    const auto result = pde_advection_1d(u0, 0.0, 0.1, 0.01, 10);
    ASSERT_FALSE(result.u.empty());
    for (const auto& snap : result.u) {
        for (std::size_t i = 0; i < u0.size(); ++i) {
            EXPECT_NEAR(snap[i], u0[i], 1e-12);
        }
    }
}

TEST(PdeExtTest, advection_1d_zero_steps) {
    const std::vector<double> u0(5, 1.0);
    const auto result = pde_advection_1d(u0, 1.0, 0.1, 0.01, 0);
    EXPECT_TRUE(result.u.empty());
}

// ---------------------------------------------------------------------------
// pde_poisson_2d
// ---------------------------------------------------------------------------

TEST(PdeExtTest, poisson_2d_manufactured_solution) {
    const std::size_t n = 11;
    const double dx = 1.0 / static_cast<double>(n - 1);
    const double dy = dx;
    auto f = make_grid(n, n);
    for (std::size_t j = 1; j + 1 < n; ++j) {
        for (std::size_t i = 1; i + 1 < n; ++i) {
            const double x = static_cast<double>(i) * dx;
            const double y = static_cast<double>(j) * dy;
            f[j][i] = -2.0 * M_PI * M_PI * std::sin(M_PI * x) * std::sin(M_PI * y);
        }
    }
    const auto result = pde_poisson_2d(f, dx, dy, 5000, 1e-6);
    ASSERT_FALSE(result.u.empty());
    EXPECT_TRUE(result.converged);
    EXPECT_GT(result.iterations, 0u);
    double max_err = 0.0;
    for (std::size_t j = 1; j + 1 < n; ++j) {
        for (std::size_t i = 1; i + 1 < n; ++i) {
            const double x = static_cast<double>(i) * dx;
            const double y = static_cast<double>(j) * dy;
            const double exact = std::sin(M_PI * x) * std::sin(M_PI * y);
            max_err = std::max(max_err, std::abs(result.u[j][i] - exact));
        }
    }
    EXPECT_LT(max_err, 0.05);
}

TEST(PdeExtTest, poisson_2d_zero_source_is_zero) {
    auto f = make_grid(7, 7, 0.0);
    const auto result = pde_poisson_2d(f, 0.1, 0.1, 100, 1e-8);
    ASSERT_FALSE(result.u.empty());
    EXPECT_TRUE(result.converged);
    for (const auto& row : result.u) {
        for (double v : row) {
            EXPECT_NEAR(v, 0.0, 1e-6);
        }
    }
}

TEST(PdeExtTest, poisson_2d_converged_flag) {
    auto f = make_grid(9, 9);
    f[4][4] = -1.0;
    const auto result = pde_poisson_2d(f, 0.1, 0.1, 2000, 1e-4);
    ASSERT_FALSE(result.u.empty());
    EXPECT_TRUE(result.converged);
    EXPECT_LE(result.iterations, 2000u);
}

TEST(PdeExtTest, poisson_2d_too_small_grid) {
    auto f = make_grid(2, 2, 1.0);
    const auto result = pde_poisson_2d(f, 0.1, 0.1, 10, 1e-6);
    EXPECT_TRUE(result.u.empty());
    EXPECT_FALSE(result.converged);
}

TEST(PdeExtTest, poisson_2d_max_iterations_used) {
    auto f = make_grid(7, 7);
    f[3][3] = -10.0;
    const auto result = pde_poisson_2d(f, 0.1, 0.1, 3, 1e-12);
    ASSERT_FALSE(result.u.empty());
    EXPECT_EQ(result.iterations, 3u);
}

// ---------------------------------------------------------------------------
// pde_burgers_1d
// ---------------------------------------------------------------------------

TEST(PdeExtTest, burgers_1d_boundaries_fixed) {
    std::vector<double> u0 = {0.0, 0.5, 1.0, 0.5, 0.0};
    const auto result = pde_burgers_1d(u0, 0.01, 0.1, 0.001, 20);
    ASSERT_FALSE(result.u.empty());
    for (const auto& snap : result.u) {
        EXPECT_NEAR(snap.front(), 0.0, 1e-12);
        EXPECT_NEAR(snap.back(), 0.0, 1e-12);
    }
}

TEST(PdeExtTest, burgers_1d_finite_values) {
    std::vector<double> u0(15, 0.0);
    u0[7] = 1.0;
    const auto result = pde_burgers_1d(u0, 0.05, 0.1, 0.001, 50);
    ASSERT_FALSE(result.u.empty());
    for (const auto& snap : result.u) {
        for (double v : snap) {
            EXPECT_TRUE(std::isfinite(v));
        }
    }
}

TEST(PdeExtTest, burgers_1d_viscosity_smooths) {
    std::vector<double> u0(11, 0.0);
    u0[5] = 2.0;
    const auto result = pde_burgers_1d(u0, 0.2, 0.1, 0.001, 100);
    ASSERT_FALSE(result.u.empty());
    const double peak0 = *std::max_element(result.u.front().begin(), result.u.front().end());
    const double peak1 = *std::max_element(result.u.back().begin(), result.u.back().end());
    EXPECT_LT(peak1, peak0);
}

TEST(PdeExtTest, burgers_1d_too_small_grid) {
    const std::vector<double> u0 = {0.0, 1.0};
    const auto result = pde_burgers_1d(u0, 0.01, 0.1, 0.001, 5);
    EXPECT_TRUE(result.u.empty());
}

TEST(PdeExtTest, burgers_1d_zero_steps) {
    const std::vector<double> u0(5, 1.0);
    const auto result = pde_burgers_1d(u0, 0.01, 0.1, 0.001, 0);
    EXPECT_TRUE(result.u.empty());
}
