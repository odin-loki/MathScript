// MathScript ODE + PDE + Stats Integration Test Suite

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <numeric>
#include <vector>

#include "ms/ode/ode.hpp"
#include "ms/pde/pde.hpp"
#include "ms/stats/stats.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ms;

// ---------------------------------------------------------------------------
// Test 1: RK4 exponential decay trajectory mean is bounded in (0, 1)
// ---------------------------------------------------------------------------
TEST(OdePdePipeline, rk4_decay_trajectory_mean_in_range) {
    auto f = [](double /*t*/, double y) { return -y; };
    const auto result = ode_rk4(f, 0.0, 1.0, 5.0, 200);
    ASSERT_FALSE(result.y.empty());
    const double m = mean(result.y);
    // y = exp(-t) on [0,5]: all values in (exp(-5), 1], so mean strictly in (0,1)
    EXPECT_GT(m, 0.0);
    EXPECT_LT(m, 1.0);
}

// ---------------------------------------------------------------------------
// Test 2: RK4 is more accurate than Euler at the same step count
// ---------------------------------------------------------------------------
TEST(OdePdePipeline, rk4_more_accurate_than_euler_same_steps) {
    auto f = [](double /*t*/, double y) { return -y; };
    const auto euler = ode_euler(f, 0.0, 1.0, 2.0, 100);
    const auto rk4   = ode_rk4(f, 0.0, 1.0, 2.0, 100);
    const double exact      = std::exp(-2.0);
    const double euler_err  = std::abs(euler.y.back() - exact);
    const double rk4_err    = std::abs(rk4.y.back()   - exact);
    EXPECT_LT(rk4_err, euler_err);
}

// ---------------------------------------------------------------------------
// Test 3: RK4 logistic growth mean lies strictly between initial value and K
// ---------------------------------------------------------------------------
TEST(OdePdePipeline, rk4_population_growth_mean_between_bounds) {
    const double r = 0.5, K = 10.0, y0_val = 1.0;
    auto f = [r, K](double /*t*/, double y) { return r * y * (1.0 - y / K); };
    const auto result = ode_rk4(f, 0.0, y0_val, 10.0, 500);
    ASSERT_FALSE(result.y.empty());
    const double m = mean(result.y);
    EXPECT_GT(m, y0_val);  // trajectory is monotone increasing
    EXPECT_LT(m, K);        // mean below carrying capacity
}

// ---------------------------------------------------------------------------
// Test 4: RK4 of cosine-driven ODE matches analytical solution y = sin(t)
// ---------------------------------------------------------------------------
TEST(OdePdePipeline, rk4_cosine_driven_vs_analytical) {
    // dy/dt = cos(t), y(0) = 0  =>  y(t) = sin(t)
    auto f = [](double t, double /*y*/) { return std::cos(t); };
    const auto result = ode_rk4(f, 0.0, 0.0, M_PI, 200);
    ASSERT_FALSE(result.y.empty());
    // y(pi) = sin(pi) = 0
    EXPECT_NEAR(result.y.back(), std::sin(M_PI), 1e-6);
    // Check midpoint y(pi/2) ≈ 1
    const size_t mid = result.y.size() / 2;
    EXPECT_NEAR(result.y[mid], std::sin(result.t[mid]), 1e-5);
}

// ---------------------------------------------------------------------------
// Test 5: Euler integral of decay trajectory ≈ 1 - exp(-T)
// ---------------------------------------------------------------------------
TEST(OdePdePipeline, euler_decay_integral_conservation) {
    // integral_0^T exp(-t) dt = 1 - exp(-T)
    const double T = 3.0;
    const size_t steps = 3000;
    auto f = [](double /*t*/, double y) { return -y; };
    const auto result = ode_euler(f, 0.0, 1.0, T, steps);
    const double dt = T / static_cast<double>(steps);
    double integral = 0.0;
    for (const double v : result.y) integral += v * dt;
    EXPECT_NEAR(integral, 1.0 - std::exp(-T), 0.01);
}

// ---------------------------------------------------------------------------
// Test 6: RK4 decay trajectory stddev is positive and less than 1
// ---------------------------------------------------------------------------
TEST(OdePdePipeline, rk4_decay_trajectory_stddev_positive) {
    auto f = [](double /*t*/, double y) { return -y; };
    const auto result = ode_rk4(f, 0.0, 1.0, 5.0, 100);
    ASSERT_GT(result.y.size(), 1u);
    const double s = stddev(result.y);
    EXPECT_GT(s, 0.0);
    EXPECT_LT(s, 1.0);  // all values in [0,1] so stddev < 1
}

// ---------------------------------------------------------------------------
// Test 7: PDE heat - peak temperature decreases as heat diffuses
// ---------------------------------------------------------------------------
TEST(OdePdePipeline, pde_heat_1d_peak_decreases_over_time) {
    // Rectangle bump in middle; Dirichlet endpoints fixed at 0
    const size_t nx = 20;
    std::vector<double> x0(nx, 0.0);
    for (size_t i = nx / 3; i < 2 * nx / 3; ++i) x0[i] = 1.0;
    const auto result = pde_heat_1d(x0, 0.01, 0.1, 0.01, 500);
    ASSERT_FALSE(result.u.empty());
    const auto& final_state = result.u.back();
    const double init_max  = *std::max_element(x0.begin(), x0.end());
    const double final_max = *std::max_element(final_state.begin(), final_state.end());
    EXPECT_LT(final_max, init_max);
}

// ---------------------------------------------------------------------------
// Test 8: PDE heat - spatial variance decreases as distribution flattens
// ---------------------------------------------------------------------------
TEST(OdePdePipeline, pde_heat_spatial_variance_decreases) {
    const size_t nx = 30;
    std::vector<double> x0(nx, 0.0);
    for (size_t i = nx / 3; i < 2 * nx / 3; ++i) x0[i] = 1.0;
    const auto result = pde_heat_1d(x0, 0.02, 0.05, 0.005, 200);
    ASSERT_GE(result.u.size(), 2u);
    const double var_initial = var(result.u.front());
    const double var_final   = var(result.u.back());
    EXPECT_LT(var_final, var_initial);
}

// ---------------------------------------------------------------------------
// Test 9: RK4 logistic growth final value matches analytical formula
// ---------------------------------------------------------------------------
TEST(OdePdePipeline, rk4_logistic_matches_analytical) {
    const double r = 1.0, K = 10.0, y0_val = 1.0;
    auto f = [r, K](double /*t*/, double y) { return r * y * (1.0 - y / K); };
    const auto result = ode_rk4(f, 0.0, y0_val, 5.0, 500);
    const double t_end      = result.t.back();
    const double y_analytic = K * y0_val / (y0_val + (K - y0_val) * std::exp(-r * t_end));
    EXPECT_NEAR(result.y.back(), y_analytic, 1e-4);
}

// ---------------------------------------------------------------------------
// Test 10: Log-linear regression on RK4 decay gives slope ≈ -lambda
// ---------------------------------------------------------------------------
TEST(OdePdePipeline, rk4_decay_log_regression_slope) {
    const double lambda = 2.0;
    auto f = [lambda](double /*t*/, double y) { return -lambda * y; };
    const auto result = ode_rk4(f, 0.0, 1.0, 3.0, 300);
    std::vector<double> log_y;
    log_y.reserve(result.y.size());
    std::vector<double> t_sub;
    t_sub.reserve(result.t.size());
    for (size_t i = 0; i < result.y.size(); ++i) {
        if (result.y[i] > 0.0) {
            log_y.push_back(std::log(result.y[i]));
            t_sub.push_back(result.t[i]);
        }
    }
    ASSERT_FALSE(log_y.empty());
    const auto reg = linear_regression(t_sub, log_y);
    EXPECT_NEAR(reg.slope, -lambda, 0.05);
    EXPECT_GT(reg.r_squared, 0.99);
}

// ---------------------------------------------------------------------------
// Test 11: Euler convergence - finer step size reduces error
// ---------------------------------------------------------------------------
TEST(OdePdePipeline, euler_convergence_with_step_count) {
    auto f = [](double /*t*/, double y) { return -y; };
    const double exact       = std::exp(-1.0);
    const auto coarse        = ode_euler(f, 0.0, 1.0, 1.0, 100);
    const auto fine          = ode_euler(f, 0.0, 1.0, 1.0, 1000);
    const double err_coarse  = std::abs(coarse.y.back() - exact);
    const double err_fine    = std::abs(fine.y.back()   - exact);
    EXPECT_LT(err_fine, err_coarse);
}

// ---------------------------------------------------------------------------
// Test 12: PDE heat - all temperatures remain finite and non-negative
// ---------------------------------------------------------------------------
TEST(OdePdePipeline, pde_heat_temperatures_finite_and_nonnegative) {
    const size_t nx = 25;
    std::vector<double> x0(nx, 0.0);
    for (size_t i = 5; i < 20; ++i) x0[i] = 2.0;
    const auto result = pde_heat_1d(x0, 0.01, 0.1, 0.01, 100);
    ASSERT_FALSE(result.u.empty());
    for (const auto& row : result.u) {
        ASSERT_FALSE(row.empty());
        const double m = mean(row);
        EXPECT_TRUE(std::isfinite(m));
        for (const double v : row) EXPECT_GE(v, -1e-10);
    }
}

// ---------------------------------------------------------------------------
// Test 13: Euler and RK4 trajectories are highly correlated
// ---------------------------------------------------------------------------
TEST(OdePdePipeline, euler_rk4_trajectory_correlation_high) {
    auto f = [](double /*t*/, double y) { return -0.5 * y; };
    const auto euler = ode_euler(f, 0.0, 1.0, 4.0, 200);
    const auto rk4   = ode_rk4(f, 0.0, 1.0, 4.0, 200);
    ASSERT_EQ(euler.y.size(), rk4.y.size());
    const double corr = correlation(euler.y, rk4.y);
    EXPECT_GT(corr, 0.999);
}
