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

double grid_max_abs(const std::vector<std::vector<double>>& g) {
    double m = 0.0;
    for (const auto& row : g) {
        for (double v : row) {
            m = std::max(m, std::abs(v));
        }
    }
    return m;
}

std::vector<double> sin_initial_1d(std::size_t n) {
    std::vector<double> u0(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        u0[i] = std::sin(M_PI * static_cast<double>(i) / static_cast<double>(n - 1));
    }
    return u0;
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
// pde_advection_1d_lax_wendroff
// ---------------------------------------------------------------------------

TEST(PdeExtTest, lax_wendroff_gaussian_pulse_shifts_to_expected_location) {
    const std::size_t n = 200;
    const double dx = 1.0;
    const double v = 1.0;
    const double dt = 0.5;
    const std::size_t steps = 40;
    const double x0 = 50.0;
    const double sigma = 5.0;

    std::vector<double> u0(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(i) * dx;
        u0[i] = std::exp(-(x - x0) * (x - x0) / (2.0 * sigma * sigma));
    }

    const auto result = pde_advection_1d_lax_wendroff(u0, v, dx, dt, steps);
    ASSERT_FALSE(result.u.empty());

    const auto& last = result.u.back();
    const std::size_t peak_idx = static_cast<std::size_t>(
        std::distance(last.begin(), std::max_element(last.begin(), last.end())));

    const double expected_x = x0 + v * dt * static_cast<double>(steps);
    const double expected_idx = expected_x / dx;
    EXPECT_NEAR(static_cast<double>(peak_idx), expected_idx, 2.0);
}

TEST(PdeExtTest, lax_wendroff_more_accurate_than_upwind_for_smooth_profile) {
    const std::size_t n = 200;
    const double dx = 1.0;
    const double v = 1.0;
    const double dt = 0.5;
    const std::size_t steps = 40;
    const double x0 = 50.0;
    const double sigma = 8.0;

    std::vector<double> u0(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(i) * dx;
        u0[i] = std::exp(-(x - x0) * (x - x0) / (2.0 * sigma * sigma));
    }

    const auto lw_result = pde_advection_1d_lax_wendroff(u0, v, dx, dt, steps);
    const auto up_result = pde_advection_1d(u0, v, dx, dt, steps);
    ASSERT_FALSE(lw_result.u.empty());
    ASSERT_FALSE(up_result.u.empty());

    // Exact solution for u_t + v u_x = 0 is a rigid shift: u(x, t) = u0(x - v*t).
    // The pulse stays far from the periodic boundary for the whole run, so wraparound
    // is negligible and this direct comparison is valid.
    const double shift = v * dt * static_cast<double>(steps);
    double lw_sq_err = 0.0;
    double up_sq_err = 0.0;
    const auto& lw_last = lw_result.u.back();
    const auto& up_last = up_result.u.back();
    for (std::size_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(i) * dx;
        const double exact_x = x - shift;
        const double exact = std::exp(-(exact_x - x0) * (exact_x - x0) / (2.0 * sigma * sigma));
        lw_sq_err += (lw_last[i] - exact) * (lw_last[i] - exact);
        up_sq_err += (up_last[i] - exact) * (up_last[i] - exact);
    }
    const double lw_rmse = std::sqrt(lw_sq_err / static_cast<double>(n));
    const double up_rmse = std::sqrt(up_sq_err / static_cast<double>(n));

    EXPECT_GT(up_rmse, 0.0);
    EXPECT_LT(lw_rmse, up_rmse);
    EXPECT_LT(lw_rmse, 0.5 * up_rmse);
}

TEST(PdeExtTest, lax_wendroff_bounded_over_many_periodic_steps) {
    const std::size_t n = 64;
    const double dx = 1.0;
    const double v = 1.0;
    const double dt = 0.5;
    const std::size_t steps = 500;
    ASSERT_LE(std::abs(v) * dt / dx, 1.0);

    std::vector<double> u0(n);
    for (std::size_t i = 0; i < n; ++i) {
        u0[i] = std::sin(2.0 * M_PI * static_cast<double>(i) / static_cast<double>(n));
    }

    const auto result = pde_advection_1d_lax_wendroff(u0, v, dx, dt, steps);
    ASSERT_FALSE(result.u.empty());

    double max_abs = 0.0;
    for (const auto& snap : result.u) {
        for (double val : snap) {
            ASSERT_TRUE(std::isfinite(val));
            max_abs = std::max(max_abs, std::abs(val));
        }
    }
    EXPECT_LT(max_abs, 2.0);
}

TEST(PdeExtTest, lax_wendroff_zero_velocity_unchanged) {
    const std::vector<double> u0 = {0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0};
    const auto result = pde_advection_1d_lax_wendroff(u0, 0.0, 0.1, 0.01, 10);
    ASSERT_FALSE(result.u.empty());
    for (const auto& snap : result.u) {
        for (std::size_t i = 0; i < u0.size(); ++i) {
            EXPECT_NEAR(snap[i], u0[i], 1e-12);
        }
    }
}

TEST(PdeExtTest, lax_wendroff_cfl_rejection) {
    const std::vector<double> u0(8, 0.0);
    const auto result = pde_advection_1d_lax_wendroff(u0, 2.0, 0.1, 0.1, 5);
    EXPECT_TRUE(result.u.empty());
}

TEST(PdeExtTest, lax_wendroff_zero_steps) {
    const std::vector<double> u0(5, 1.0);
    const auto result = pde_advection_1d_lax_wendroff(u0, 1.0, 0.1, 0.01, 0);
    EXPECT_TRUE(result.u.empty());
}

TEST(PdeExtTest, lax_wendroff_too_short_input_handled_gracefully) {
    const std::vector<double> two_points = {1.0, 2.0};
    const auto result_two = pde_advection_1d_lax_wendroff(two_points, 1.0, 0.1, 0.01, 5);
    EXPECT_TRUE(result_two.u.empty());

    const std::vector<double> single_point = {1.0};
    const auto result_single = pde_advection_1d_lax_wendroff(single_point, 1.0, 0.1, 0.01, 5);
    EXPECT_TRUE(result_single.u.empty());

    const std::vector<double> empty_input;
    const auto result_empty = pde_advection_1d_lax_wendroff(empty_input, 1.0, 0.1, 0.01, 5);
    EXPECT_TRUE(result_empty.u.empty());
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
// pde_helmholtz_2d
// ---------------------------------------------------------------------------

TEST(PdeExtTest, helmholtz_2d_manufactured_solution) {
    const std::size_t n = 21;
    const double dx = 1.0 / static_cast<double>(n - 1);
    const double dy = dx;
    const double k = 1.0;
    const double coeff = k * k - 2.0 * M_PI * M_PI;
    auto f = make_grid(n, n);
    for (std::size_t j = 1; j + 1 < n; ++j) {
        for (std::size_t i = 1; i + 1 < n; ++i) {
            const double x = static_cast<double>(i) * dx;
            const double y = static_cast<double>(j) * dy;
            f[j][i] = coeff * std::sin(M_PI * x) * std::sin(M_PI * y);
        }
    }
    const auto result = pde_helmholtz_2d(f, k, dx, dy);
    ASSERT_FALSE(result.u.empty());
    double max_err = 0.0;
    for (std::size_t j = 1; j + 1 < n; ++j) {
        for (std::size_t i = 1; i + 1 < n; ++i) {
            const double x = static_cast<double>(i) * dx;
            const double y = static_cast<double>(j) * dy;
            const double exact = std::sin(M_PI * x) * std::sin(M_PI * y);
            max_err = std::max(max_err, std::abs(result.u[j][i] - exact));
        }
    }
    EXPECT_LT(max_err, 0.01);
}

TEST(PdeExtTest, helmholtz_2d_convergence_with_refinement) {
    const double k = 1.0;
    const double coeff = k * k - 2.0 * M_PI * M_PI;
    double prev_err = 1.0;
    for (const std::size_t n : {11u, 21u, 41u}) {
        const double dx = 1.0 / static_cast<double>(n - 1);
        const double dy = dx;
        auto f = make_grid(n, n);
        for (std::size_t j = 1; j + 1 < n; ++j) {
            for (std::size_t i = 1; i + 1 < n; ++i) {
                const double x = static_cast<double>(i) * dx;
                const double y = static_cast<double>(j) * dy;
                f[j][i] = coeff * std::sin(M_PI * x) * std::sin(M_PI * y);
            }
        }
        const auto result = pde_helmholtz_2d(f, k, dx, dy);
        ASSERT_FALSE(result.u.empty());
        double max_err = 0.0;
        for (std::size_t j = 1; j + 1 < n; ++j) {
            for (std::size_t i = 1; i + 1 < n; ++i) {
                const double x = static_cast<double>(i) * dx;
                const double y = static_cast<double>(j) * dy;
                const double exact = std::sin(M_PI * x) * std::sin(M_PI * y);
                max_err = std::max(max_err, std::abs(result.u[j][i] - exact));
            }
        }
        EXPECT_LT(max_err, prev_err);
        prev_err = max_err;
    }
}

TEST(PdeExtTest, helmholtz_2d_k_zero_matches_poisson) {
    const std::size_t n = 15;
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
    const auto helm = pde_helmholtz_2d(f, 0.0, dx, dy);
    const auto pois = pde_poisson_2d(f, dx, dy, 10000, 1e-8);
    ASSERT_FALSE(helm.u.empty());
    ASSERT_FALSE(pois.u.empty());
    ASSERT_TRUE(pois.converged);
    for (std::size_t j = 1; j + 1 < n; ++j) {
        for (std::size_t i = 1; i + 1 < n; ++i) {
            EXPECT_NEAR(helm.u[j][i], pois.u[j][i], 0.02);
        }
    }
}

TEST(PdeExtTest, helmholtz_2d_zero_forcing_zero_solution) {
    auto f = make_grid(9, 9, 0.0);
    const auto result = pde_helmholtz_2d(f, 2.0, 0.1, 0.1);
    ASSERT_FALSE(result.u.empty());
    for (const auto& row : result.u) {
        for (double v : row) {
            EXPECT_NEAR(v, 0.0, 1e-10);
        }
    }
}

TEST(PdeExtTest, helmholtz_2d_nonzero_boundary_values) {
    const std::size_t ny = 9;
    const std::size_t nx = 11;
    const double dx = 0.1;
    const double dy = 0.15;
    auto f = make_grid(ny, nx, 0.0);
    auto g = make_grid(ny, nx, 0.0);
    for (std::size_t i = 0; i < nx; ++i) {
        g[0][i] = std::sin(M_PI * static_cast<double>(i) / static_cast<double>(nx - 1));
        g[ny - 1][i] = -0.5 * g[0][i];
    }
    for (std::size_t j = 0; j < ny; ++j) {
        g[j][0] = static_cast<double>(j) * 0.1;
        g[j][nx - 1] = -g[j][0];
    }
    const auto result = pde_helmholtz_2d(f, 1.5, dx, dy, g);
    ASSERT_FALSE(result.u.empty());
    for (std::size_t i = 0; i < nx; ++i) {
        EXPECT_NEAR(result.u[0][i], g[0][i], 1e-12);
        EXPECT_NEAR(result.u[ny - 1][i], g[ny - 1][i], 1e-12);
    }
    for (std::size_t j = 0; j < ny; ++j) {
        EXPECT_NEAR(result.u[j][0], g[j][0], 1e-12);
        EXPECT_NEAR(result.u[j][nx - 1], g[j][nx - 1], 1e-12);
    }
    for (const auto& row : result.u) {
        for (double v : row) {
            EXPECT_TRUE(std::isfinite(v));
        }
    }
}

TEST(PdeExtTest, helmholtz_2d_non_square_grid) {
    const std::size_t ny = 7;
    const std::size_t nx = 13;
    const double dx = 0.2;
    const double dy = 0.1;
    auto f = make_grid(ny, nx, 1.0);
    const auto result = pde_helmholtz_2d(f, 0.5, dx, dy);
    ASSERT_FALSE(result.u.empty());
    EXPECT_EQ(result.u.size(), ny);
    EXPECT_EQ(result.u[0].size(), nx);
    for (const auto& row : result.u) {
        for (double v : row) {
            EXPECT_TRUE(std::isfinite(v));
            EXPECT_LT(std::abs(v), 1e6);
        }
    }
}

TEST(PdeExtTest, helmholtz_2d_too_small_grid) {
    auto f = make_grid(2, 2, 1.0);
    const auto result = pde_helmholtz_2d(f, 1.0, 0.1, 0.1);
    EXPECT_TRUE(result.u.empty());
}

TEST(PdeExtTest, helmholtz_2d_invalid_parameters) {
    auto f = make_grid(7, 7, 1.0);
    EXPECT_TRUE(pde_helmholtz_2d(f, 1.0, 0.0, 0.1).u.empty());
    EXPECT_TRUE(pde_helmholtz_2d(f, 1.0, 0.1, 0.0).u.empty());
    auto bad_g = make_grid(5, 7, 0.0);
    EXPECT_TRUE(pde_helmholtz_2d(f, 1.0, 0.1, 0.1, bad_g).u.empty());
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

// ---------------------------------------------------------------------------
// pde_heat_1d_cn
// ---------------------------------------------------------------------------

TEST(PdeExtTest, heat_1d_cn_matches_explicit_when_stable) {
    const std::size_t n = 21;
    const double dx = 0.05;
    const double alpha = 0.1;
    const double dt = 0.005;
    const std::size_t steps = 20;
    const auto x0 = sin_initial_1d(n);

    const auto explicit_result = pde_heat_1d(x0, alpha, dx, dt, steps);
    const auto cn_result = pde_heat_1d_cn(x0, alpha, dx, dt, steps);
    ASSERT_FALSE(explicit_result.u.empty());
    ASSERT_FALSE(cn_result.u.empty());
    ASSERT_EQ(explicit_result.u.size(), cn_result.u.size());

    for (std::size_t k = 0; k < explicit_result.u.size(); ++k) {
        for (std::size_t i = 0; i < n; ++i) {
            EXPECT_NEAR(cn_result.u[k][i], explicit_result.u[k][i], 0.05);
        }
    }
}

TEST(PdeExtTest, heat_1d_cn_stable_for_large_dt) {
    const std::size_t n = 21;
    const double dx = 0.05;
    const double alpha = 0.1;
    const double dt = 0.1;
    const auto x0 = sin_initial_1d(n);

    const double r = alpha * dt / (dx * dx);
    ASSERT_GT(r, 0.5);

    const auto explicit_result = pde_heat_1d(x0, alpha, dx, dt, 10);
    EXPECT_TRUE(explicit_result.u.empty());

    const auto cn_result = pde_heat_1d_cn(x0, alpha, dx, dt, 10);
    ASSERT_FALSE(cn_result.u.empty());
    for (const auto& snap : cn_result.u) {
        for (double v : snap) {
            EXPECT_TRUE(std::isfinite(v));
            EXPECT_LT(std::abs(v), 2.0);
        }
    }
}

TEST(PdeExtTest, heat_1d_cn_dirichlet_boundaries_zero) {
    const std::vector<double> x0 = {0.0, 1.0, 0.8, 0.5, 0.8, 1.0, 0.0};
    const auto result = pde_heat_1d_cn(x0, 0.1, 0.1, 0.01, 15);
    ASSERT_FALSE(result.u.empty());
    for (const auto& snap : result.u) {
        EXPECT_NEAR(snap.front(), 0.0, 1e-12);
        EXPECT_NEAR(snap.back(), 0.0, 1e-12);
    }
}

TEST(PdeExtTest, heat_1d_cn_diffuses_spike) {
    std::vector<double> x0(11, 0.0);
    x0[5] = 5.0;
    const auto result = pde_heat_1d_cn(x0, 0.2, 0.1, 0.01, 30);
    ASSERT_FALSE(result.u.empty());
    EXPECT_LT(*std::max_element(result.u.back().begin(), result.u.back().end()), 5.0);
    EXPECT_GT(*std::max_element(result.u.back().begin(), result.u.back().end()), 0.0);
}

TEST(PdeExtTest, heat_1d_cn_too_small_grid) {
    const std::vector<double> x0 = {0.0, 1.0};
    const auto result = pde_heat_1d_cn(x0, 0.1, 0.1, 0.01, 5);
    EXPECT_TRUE(result.u.empty());
}

TEST(PdeExtTest, heat_1d_cn_zero_steps) {
    const std::vector<double> x0(5, 0.0);
    const auto result = pde_heat_1d_cn(x0, 0.1, 0.1, 0.01, 0);
    EXPECT_TRUE(result.u.empty());
}

TEST(PdeExtTest, heat_1d_cn_invalid_parameters) {
    std::vector<double> x0(5, 0.0);
    x0[2] = 1.0;
    EXPECT_TRUE(pde_heat_1d_cn(x0, 0.0, 0.1, 0.01, 5).u.empty());
    EXPECT_TRUE(pde_heat_1d_cn(x0, 0.1, 0.0, 0.01, 5).u.empty());
    EXPECT_TRUE(pde_heat_1d_cn(x0, 0.1, 0.1, 0.0, 5).u.empty());
}

// ---------------------------------------------------------------------------
// pde_poisson_1d
// ---------------------------------------------------------------------------

TEST(PdeExtTest, poisson_1d_manufactured_solution) {
    const std::size_t n = 51;
    const double dx = 1.0 / static_cast<double>(n - 1);
    const double L = static_cast<double>(n - 1) * dx;
    const double coeff = -(M_PI / L) * (M_PI / L);

    std::vector<double> f(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(i) * dx;
        f[i] = coeff * std::sin(M_PI * x / L);
    }

    const auto result = pde_poisson_1d(f, dx, 0.0, 0.0);
    ASSERT_EQ(result.u.size(), n);
    double max_err = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(i) * dx;
        const double exact = std::sin(M_PI * x / L);
        max_err = std::max(max_err, std::abs(result.u[i] - exact));
    }
    EXPECT_LT(max_err, 0.002);
}

TEST(PdeExtTest, poisson_1d_boundary_values) {
    const std::size_t n = 11;
    const double dx = 0.1;
    std::vector<double> f(n, 0.0);
    const auto result = pde_poisson_1d(f, dx, 2.0, -1.0);
    ASSERT_EQ(result.u.size(), n);
    EXPECT_NEAR(result.u.front(), 2.0, 1e-12);
    EXPECT_NEAR(result.u.back(), -1.0, 1e-12);
}

TEST(PdeExtTest, poisson_1d_linear_with_zero_source) {
    const std::size_t n = 6;
    const double dx = 0.2;
    std::vector<double> f(n, 0.0);
    const double ua = 1.0;
    const double ub = 3.0;
    const auto result = pde_poisson_1d(f, dx, ua, ub);
    ASSERT_EQ(result.u.size(), n);
    for (std::size_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(i) * dx;
        const double exact = ua + (ub - ua) * x / (static_cast<double>(n - 1) * dx);
        EXPECT_NEAR(result.u[i], exact, 1e-10);
    }
}

TEST(PdeExtTest, poisson_1d_too_small_grid) {
    const std::vector<double> f = {0.0, 1.0};
    const auto result = pde_poisson_1d(f, 0.1, 0.0, 0.0);
    EXPECT_TRUE(result.u.empty());
}

TEST(PdeExtTest, poisson_1d_invalid_dx) {
    const std::vector<double> f(5, 0.0);
    const auto result = pde_poisson_1d(f, 0.0, 0.0, 0.0);
    EXPECT_TRUE(result.u.empty());
}

// ---------------------------------------------------------------------------
// pde_wave_2d
// ---------------------------------------------------------------------------

TEST(PdeExtTest, wave_2d_cfl_rejection) {
    auto u0 = make_grid(7, 7, 0.0);
    auto v0 = make_grid(7, 7, 0.0);
    u0[3][3] = 1.0;
    const auto result = pde_wave_2d(u0, v0, 2.0, 0.1, 0.1, 0.2, 5);
    EXPECT_TRUE(result.u.empty());
    EXPECT_TRUE(result.t.empty());
}

TEST(PdeExtTest, wave_2d_symmetry_preserved) {
    const std::size_t n = 9;
    auto u0 = make_grid(n, n, 0.0);
    auto v0 = make_grid(n, n, 0.0);
    u0[n / 2][n / 2] = 1.0;

    const auto result = pde_wave_2d(u0, v0, 1.0, 0.1, 0.1, 0.02, 8);
    ASSERT_FALSE(result.u.empty());
    for (const auto& snap : result.u) {
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t i = 0; i < n; ++i) {
                EXPECT_NEAR(snap[j][i], snap[j][n - 1 - i], 1e-10);
                EXPECT_NEAR(snap[j][i], snap[n - 1 - j][i], 1e-10);
            }
        }
    }
}

TEST(PdeExtTest, wave_2d_dirichlet_boundaries_zero) {
    auto u0 = make_grid(7, 7, 0.0);
    auto v0 = make_grid(7, 7, 0.0);
    u0[3][3] = 1.0;
    const auto result = pde_wave_2d(u0, v0, 1.0, 0.1, 0.1, 0.02, 10);
    ASSERT_FALSE(result.u.empty());
    for (const auto& snap : result.u) {
        for (std::size_t i = 0; i < snap[0].size(); ++i) {
            EXPECT_NEAR(snap[0][i], 0.0, 1e-12);
            EXPECT_NEAR(snap.back()[i], 0.0, 1e-12);
        }
        for (const auto& row : snap) {
            EXPECT_NEAR(row.front(), 0.0, 1e-12);
            EXPECT_NEAR(row.back(), 0.0, 1e-12);
        }
    }
}

TEST(PdeExtTest, wave_2d_finite_output) {
    auto u0 = make_grid(11, 11, 0.0);
    auto v0 = make_grid(11, 11, 0.0);
    u0[5][5] = 0.5;
    v0[5][5] = 0.1;
    const auto result = pde_wave_2d(u0, v0, 1.0, 0.1, 0.1, 0.01, 12);
    ASSERT_EQ(result.u.size(), 13u);
    for (const auto& snap : result.u) {
        for (const auto& row : snap) {
            for (double v : row) {
                EXPECT_TRUE(std::isfinite(v));
            }
        }
    }
}

TEST(PdeExtTest, wave_2d_energy_roughly_preserved) {
    const std::size_t n = 11;
    auto u0 = make_grid(n, n, 0.0);
    auto v0 = make_grid(n, n, 0.0);
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < n; ++i) {
            const double x = static_cast<double>(i) / static_cast<double>(n - 1);
            const double y = static_cast<double>(j) / static_cast<double>(n - 1);
            u0[j][i] = std::sin(M_PI * x) * std::sin(M_PI * y);
        }
    }
    const auto result = pde_wave_2d(u0, v0, 1.0, 0.1, 0.1, 0.005, 6);
    ASSERT_GE(result.u.size(), 2u);
    const double norm0 = grid_max_abs(result.u.front());
    const double norm1 = grid_max_abs(result.u.back());
    EXPECT_NEAR(norm0, norm1, 0.2 * norm0);
}

TEST(PdeExtTest, wave_2d_too_small_grid) {
    auto u0 = make_grid(2, 2, 1.0);
    auto v0 = make_grid(2, 2, 0.0);
    const auto result = pde_wave_2d(u0, v0, 1.0, 0.1, 0.1, 0.01, 5);
    EXPECT_TRUE(result.u.empty());
}

TEST(PdeExtTest, wave_2d_zero_steps) {
    auto u0 = make_grid(5, 5, 0.0);
    auto v0 = make_grid(5, 5, 0.0);
    const auto result = pde_wave_2d(u0, v0, 1.0, 0.1, 0.1, 0.01, 0);
    EXPECT_TRUE(result.u.empty());
}

TEST(PdeExtTest, wave_2d_mismatched_shapes) {
    auto u0 = make_grid(5, 5, 0.0);
    auto v0 = make_grid(5, 7, 0.0);
    const auto result = pde_wave_2d(u0, v0, 1.0, 0.1, 0.1, 0.01, 5);
    EXPECT_TRUE(result.u.empty());
}

// ---------------------------------------------------------------------------
// pde_reaction_diffusion_1d
// ---------------------------------------------------------------------------

double logistic_solution(double c, double r, double t) {
    return c / (c + (1.0 - c) * std::exp(-r * t));
}

TEST(PdeExtTest, reaction_diffusion_1d_stability_rejection) {
    const std::vector<double> u0(11, 0.5);
    const double D = 1.0;
    const double dx = 0.1;
    const double dt = 0.1;
    ASSERT_GT(D * dt / (dx * dx), 0.5);
    const auto result = pde_reaction_diffusion_1d(u0, D, 1.0, dx, dt, 5);
    EXPECT_TRUE(result.u.empty());
    EXPECT_TRUE(result.t.empty());
}

TEST(PdeExtTest, reaction_diffusion_1d_uniform_logistic_evolution) {
    const std::size_t n = 21;
    const double c = 0.4;
    const double r = 2.0;
    const double D = 0.05;
    const double dx = 0.1;
    const double dt = 0.01;
    const std::size_t steps = 30;
    const std::vector<double> u0(n, c);

    const auto result = pde_reaction_diffusion_1d(u0, D, r, dx, dt, steps);
    ASSERT_FALSE(result.u.empty());
    ASSERT_EQ(result.u.size(), steps + 1);

    for (std::size_t k = 0; k < result.u.size(); ++k) {
        const double expected = logistic_solution(c, r, result.t[k]);
        for (std::size_t i = 0; i < n; ++i) {
            EXPECT_NEAR(result.u[k][i], expected, 0.02);
        }
    }
}

TEST(PdeExtTest, reaction_diffusion_1d_spatial_uniformity_preserved) {
    const std::size_t n = 15;
    const double c = 0.6;
    const std::vector<double> u0(n, c);
    const auto result = pde_reaction_diffusion_1d(u0, 0.1, 1.5, 0.1, 0.005, 20);
    ASSERT_FALSE(result.u.empty());
    for (const auto& snap : result.u) {
        for (std::size_t i = 1; i < n; ++i) {
            EXPECT_NEAR(snap[i], snap[0], 1e-12);
        }
    }
}

TEST(PdeExtTest, reaction_diffusion_1d_neumann_boundaries_finite) {
    const std::size_t n = 17;
    std::vector<double> u0(n);
    for (std::size_t i = 0; i < n; ++i) {
        u0[i] = 0.3 + 0.4 * std::exp(-0.5 * static_cast<double>((i - n / 2) * (i - n / 2)));
    }
    const auto result = pde_reaction_diffusion_1d(u0, 0.2, 1.0, 0.1, 0.002, 50);
    ASSERT_FALSE(result.u.empty());
    for (const auto& snap : result.u) {
        for (double v : snap) {
            EXPECT_TRUE(std::isfinite(v));
            EXPECT_GE(v, -0.1);
            EXPECT_LE(v, 1.2);
        }
        for (std::size_t i = 0; i < n; ++i) {
            EXPECT_NEAR(snap[i], snap[n - 1 - i], 0.05);
        }
    }
}

TEST(PdeExtTest, reaction_diffusion_1d_neumann_conserves_mass_when_r_zero) {
    const std::size_t n = 21;
    std::vector<double> u0(n);
    for (std::size_t i = 0; i < n; ++i) {
        u0[i] = 0.5 + 0.3 * std::sin(M_PI * static_cast<double>(i) / static_cast<double>(n - 1));
    }
    const double mass0 = std::accumulate(u0.begin(), u0.end(), 0.0);
    const auto result = pde_reaction_diffusion_1d(u0, 0.1, 0.0, 0.1, 0.01, 25);
    ASSERT_FALSE(result.u.empty());
    const double mass1 = std::accumulate(result.u.back().begin(), result.u.back().end(), 0.0);
    EXPECT_NEAR(mass0, mass1, 0.05 * std::abs(mass0));
}

TEST(PdeExtTest, reaction_diffusion_1d_finite_stable_output) {
    const std::size_t n = 11;
    std::vector<double> u0(n, 0.0);
    u0[n / 2] = 0.9;
    const auto result = pde_reaction_diffusion_1d(u0, 0.1, 2.0, 0.1, 0.004, 40);
    ASSERT_FALSE(result.u.empty());
    for (const auto& snap : result.u) {
        for (double v : snap) {
            EXPECT_TRUE(std::isfinite(v));
            EXPECT_GE(v, -0.1);
            EXPECT_LE(v, 1.1);
        }
    }
}

TEST(PdeExtTest, reaction_diffusion_1d_logistic_multiple_times) {
    const std::size_t n = 9;
    const double c = 0.25;
    const double r = 1.0;
    const double dx = 0.2;
    const double dt = 0.02;
    const std::vector<double> u0(n, c);
    const auto result = pde_reaction_diffusion_1d(u0, 0.01, r, dx, dt, 10);
    ASSERT_FALSE(result.u.empty());
    for (std::size_t k : {0u, 3u, 7u, 10u}) {
        const double expected = logistic_solution(c, r, result.t[k]);
        for (std::size_t i = 0; i < n; ++i) {
            EXPECT_NEAR(result.u[k][i], expected, 0.03);
        }
    }
}

TEST(PdeExtTest, reaction_diffusion_1d_too_small_grid) {
    const std::vector<double> u0 = {0.5, 0.6};
    const auto result = pde_reaction_diffusion_1d(u0, 0.1, 1.0, 0.1, 0.01, 5);
    EXPECT_TRUE(result.u.empty());
}

TEST(PdeExtTest, reaction_diffusion_1d_zero_steps) {
    const std::vector<double> u0(5, 0.5);
    const auto result = pde_reaction_diffusion_1d(u0, 0.1, 1.0, 0.1, 0.01, 0);
    EXPECT_TRUE(result.u.empty());
}

TEST(PdeExtTest, reaction_diffusion_1d_invalid_parameters) {
    const std::vector<double> u0(7, 0.5);
    EXPECT_TRUE(pde_reaction_diffusion_1d(u0, 0.1, 1.0, 0.0, 0.01, 5).u.empty());
    EXPECT_TRUE(pde_reaction_diffusion_1d(u0, 0.1, 1.0, 0.1, 0.0, 5).u.empty());
    EXPECT_TRUE(pde_reaction_diffusion_1d(u0, -0.1, 1.0, 0.1, 0.01, 5).u.empty());
}

// ---------------------------------------------------------------------------
// pde_heat_2d_cn_adi
// ---------------------------------------------------------------------------

TEST(PdeExtTest, heat_2d_cn_adi_stable_large_dt) {
    const std::size_t n = 9;
    const double dx = 0.1;
    const double dy = 0.1;
    const double alpha = 0.1;
    const double dt = 0.5;
    auto u0 = make_grid(n, n, 0.0);
    u0[n / 2][n / 2] = 1.0;

    const double stability = alpha * dt * (1.0 / (dx * dx) + 1.0 / (dy * dy));
    ASSERT_GT(stability, 0.5);

    const auto explicit_result = pde_heat_2d(u0, alpha, dx, dy, dt, 10);
    EXPECT_TRUE(explicit_result.u.empty());

    const auto adi_result = pde_heat_2d_cn_adi(u0, alpha, dx, dy, dt, 10);
    ASSERT_FALSE(adi_result.u.empty());
    for (const auto& snap : adi_result.u) {
        for (const auto& row : snap) {
            for (double v : row) {
                EXPECT_TRUE(std::isfinite(v));
                EXPECT_LT(std::abs(v), 2.0);
            }
        }
    }
}

TEST(PdeExtTest, heat_2d_cn_adi_dirichlet_boundaries_zero) {
    auto u0 = make_grid(7, 7, 0.0);
    u0[3][3] = 2.0;
    const auto result = pde_heat_2d_cn_adi(u0, 0.1, 0.1, 0.1, 0.05, 15);
    ASSERT_FALSE(result.u.empty());
    for (const auto& snap : result.u) {
        for (std::size_t i = 0; i < snap[0].size(); ++i) {
            EXPECT_NEAR(snap[0][i], 0.0, 1e-12);
            EXPECT_NEAR(snap.back()[i], 0.0, 1e-12);
        }
        for (const auto& row : snap) {
            EXPECT_NEAR(row.front(), 0.0, 1e-12);
            EXPECT_NEAR(row.back(), 0.0, 1e-12);
        }
    }
}

TEST(PdeExtTest, heat_2d_cn_adi_symmetry_preserved) {
    const std::size_t n = 9;
    auto u0 = make_grid(n, n, 0.0);
    u0[n / 2][n / 2] = 1.0;
    const auto result = pde_heat_2d_cn_adi(u0, 0.1, 0.1, 0.1, 0.02, 12);
    ASSERT_FALSE(result.u.empty());
    for (const auto& snap : result.u) {
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t i = 0; i < n; ++i) {
                EXPECT_NEAR(snap[j][i], snap[j][n - 1 - i], 1e-10);
                EXPECT_NEAR(snap[j][i], snap[n - 1 - j][i], 1e-10);
            }
        }
    }
}

TEST(PdeExtTest, heat_2d_cn_adi_matches_explicit_when_stable) {
    const std::size_t n = 7;
    const double dx = 0.1;
    const double dy = 0.1;
    const double alpha = 0.1;
    const double dt = 0.001;
    const std::size_t steps = 20;
    auto u0 = make_grid(n, n, 0.0);
    u0[n / 2][n / 2] = 1.0;

    const double stability = alpha * dt * (1.0 / (dx * dx) + 1.0 / (dy * dy));
    ASSERT_LE(stability, 0.5);

    const auto explicit_result = pde_heat_2d(u0, alpha, dx, dy, dt, steps);
    const auto adi_result = pde_heat_2d_cn_adi(u0, alpha, dx, dy, dt, steps);
    ASSERT_FALSE(explicit_result.u.empty());
    ASSERT_FALSE(adi_result.u.empty());
    ASSERT_EQ(explicit_result.u.size(), adi_result.u.size());

    for (std::size_t k = 0; k < explicit_result.u.size(); ++k) {
        for (std::size_t j = 1; j + 1 < n; ++j) {
            for (std::size_t i = 1; i + 1 < n; ++i) {
                EXPECT_NEAR(adi_result.u[k][j][i], explicit_result.u[k][j][i], 0.08);
            }
        }
    }
}

TEST(PdeExtTest, heat_2d_cn_adi_finite_output) {
    auto u0 = make_grid(11, 11, 0.0);
    u0[5][5] = 0.8;
    const auto result = pde_heat_2d_cn_adi(u0, 0.2, 0.1, 0.1, 0.1, 8);
    ASSERT_EQ(result.u.size(), 9u);
    for (const auto& snap : result.u) {
        for (const auto& row : snap) {
            for (double v : row) {
                EXPECT_TRUE(std::isfinite(v));
            }
        }
    }
}

TEST(PdeExtTest, heat_2d_cn_adi_diffuses_spike) {
    auto u0 = make_grid(7, 7, 0.0);
    u0[3][3] = 5.0;
    const auto result = pde_heat_2d_cn_adi(u0, 0.2, 0.1, 0.1, 0.01, 30);
    ASSERT_FALSE(result.u.empty());
    EXPECT_LT(grid_max(result.u.back()), 5.0);
    EXPECT_GT(grid_max(result.u.back()), 0.0);
}

TEST(PdeExtTest, heat_2d_cn_adi_too_small_grid) {
    auto u0 = make_grid(2, 2, 1.0);
    const auto result = pde_heat_2d_cn_adi(u0, 0.1, 0.1, 0.1, 0.01, 5);
    EXPECT_TRUE(result.u.empty());
}

TEST(PdeExtTest, heat_2d_cn_adi_zero_steps) {
    auto u0 = make_grid(5, 5, 1.0);
    const auto result = pde_heat_2d_cn_adi(u0, 0.1, 0.1, 0.1, 0.01, 0);
    EXPECT_TRUE(result.u.empty());
}

TEST(PdeExtTest, heat_2d_cn_adi_invalid_parameters) {
    auto u0 = make_grid(5, 5, 0.0);
    u0[2][2] = 1.0;
    EXPECT_TRUE(pde_heat_2d_cn_adi(u0, 0.0, 0.1, 0.1, 0.01, 5).u.empty());
    EXPECT_TRUE(pde_heat_2d_cn_adi(u0, 0.1, 0.0, 0.1, 0.01, 5).u.empty());
    EXPECT_TRUE(pde_heat_2d_cn_adi(u0, 0.1, 0.1, 0.0, 0.01, 5).u.empty());
    EXPECT_TRUE(pde_heat_2d_cn_adi(u0, 0.1, 0.1, 0.1, 0.0, 5).u.empty());
}
