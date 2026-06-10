#include <gtest/gtest.h>

#include "ms/ode/ode.hpp"
#include "ms/optim/optim.hpp"
#include <cmath>

using namespace ms;

TEST(OdeMoreTest, euler_exponential) {
    const auto result = ode_euler([](double, double y) { return y; }, 0.0, 1.0, 1.0, 200);
    ASSERT_FALSE(result.t.empty());
    EXPECT_NEAR(result.y.back(), std::exp(1.0), 1e-2);
}

TEST(OdeMoreTest, zero_steps_returns_empty) {
    const auto euler = ode_euler([](double, double y) { return y; }, 0.0, 1.0, 1.0, 0);
    const auto rk4 = ode_rk4([](double, double y) { return y; }, 0.0, 1.0, 1.0, 0);
    EXPECT_TRUE(euler.t.empty());
    EXPECT_TRUE(rk4.t.empty());
}

TEST(OptimMoreTest, gradient_descent_quadratic) {
    const auto f = [](double x, double y) { return x * x + y * y; };
    const auto result = gradient_descent(f, 3.0, -2.0, 0.1, 500);
    EXPECT_NEAR(result.x, 0.0, 1e-3);
    EXPECT_NEAR(result.y, 0.0, 1e-3);
    EXPECT_TRUE(result.converged);
}

TEST(OptimMoreTest, newton_raphson_sqrt2) {
    const auto f = [](double x, double) { return x * x - 2.0; };
    const auto nr = newton_raphson(f, 1.0, 0.0, 1e-10, 50);
    EXPECT_NEAR(nr.first, std::sqrt(2.0), 1e-6);
    EXPECT_NEAR(f(nr.first, nr.second), 0.0, 1e-6);
}

TEST(OptimMoreTest, broyden_sqrt2) {
    const auto f = [](double x, double) { return x * x - 2.0; };
    const auto bro = broyden(f, 1.5, 0.0);
    EXPECT_NEAR(bro.first, std::sqrt(2.0), 1e-4);
    EXPECT_NEAR(f(bro.first, bro.second), 0.0, 1e-4);
}

TEST(OptimMoreTest, minimize_with_constraints_box) {
    const auto f = [](double x, double y) { return (x - 0.25) * (x - 0.25) + (y - 0.75) * (y - 0.75); };
    const auto result = minimize_with_constraints(f, 2.0, -1.0, {0.0, 1.0});
    EXPECT_NEAR(std::get<0>(result), 0.25, 1e-2);
    EXPECT_NEAR(std::get<1>(result), 0.75, 1e-2);
    EXPECT_NEAR(std::get<2>(result), f(0.25, 0.75), 1e-2);
}

TEST(OptimMoreTest, simplex_solver_n2) {
    const auto result = simplex_solver({1.0, 1.0}, {{1.0, 0.0}, {0.0, 1.0}}, {1.0, 1.0});
    const auto& x = std::get<0>(result);
    ASSERT_EQ(x.size(), 2u);
    EXPECT_NEAR(x[0], 0.0, 1e-2);
    EXPECT_NEAR(x[1], 0.0, 1e-2);
    EXPECT_NEAR(std::get<1>(result), 0.0, 1e-2);
}

TEST(OptimMoreTest, simplex_solver_n_gt_2_empty) {
    const auto result = simplex_solver({1.0, 1.0, 1.0}, {{1.0, 0.0, 0.0}}, {1.0});
    EXPECT_TRUE(std::get<0>(result).empty());
}

TEST(OptimMoreTest, golden_section_else_branch) {
    const auto f = [](double x) { return (x + 2.0) * (x + 2.0); };
    const double xmin = golden_section(f, -5.0, 3.0);
    EXPECT_NEAR(xmin, -2.0, 1e-3);
}

TEST(OptimTest, gradient_descent_non_convergent) {
    const auto f = [](double x, double y) { return x * x + y * y; };
    const auto result = gradient_descent(f, 3.0, -2.0, 100.0, 1);
    EXPECT_FALSE(result.converged);
    EXPECT_GT(std::abs(result.x) + std::abs(result.y), 1.0);
}

TEST(OptimTest, newton_raphson_no_root) {
    const auto f = [](double x, double) { return x * x + 1.0; };
    const auto nr = newton_raphson(f, 1.0, 0.0, 1e-10, 50);
    EXPECT_GT(std::abs(f(nr.first, nr.second)), 1e-6);
}

TEST(OptimTest, broyden_max_iter) {
    const auto f = [](double x, double) { return x * x + 1.0; };
    const auto bro = broyden(f, 1.0, 0.0);
    EXPECT_GT(std::abs(f(bro.first, bro.second)), 1e-6);
}

TEST(OptimTest, golden_section_narrow_interval) {
    const auto f = [](double x) { return (x - 3.0) * (x - 3.0); };
    const double xmin = golden_section(f, 2.0, 2.0);
    EXPECT_NEAR(xmin, 2.0, 1e-12);
    EXPECT_NEAR(f(xmin), 1.0, 1e-6);
}

TEST(OptimTest, minimize_constraints_infeasible) {
    const auto f = [](double x, double y) { return x * x + y * y; };
    const auto result = minimize_with_constraints(f, 0.5, 0.5, {1.0, 0.0, 0.0, 1.0});
    const double x = std::get<0>(result);
    const double y = std::get<1>(result);
    EXPECT_TRUE(std::isfinite(x));
    EXPECT_TRUE(std::isfinite(y));
    EXPECT_TRUE(std::isfinite(std::get<2>(result)));
}

TEST(OptimTest, simplex_solver_single_variable) {
    const auto unsupported = simplex_solver({1.0}, {}, {});
    EXPECT_TRUE(std::get<0>(unsupported).empty());

    const auto result = simplex_solver({1.0, 0.0}, {{1.0, 0.0}, {0.0, 1.0}}, {10.0, 10.0});
    const auto& x = std::get<0>(result);
    ASSERT_EQ(x.size(), 2u);
    EXPECT_NEAR(x[0], 0.0, 1e-2);
}
