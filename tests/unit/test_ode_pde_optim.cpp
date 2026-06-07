#include <gtest/gtest.h>
#include "ms/ode/ode.hpp"
#include "ms/pde/pde.hpp"
#include "ms/optim/optim.hpp"
#include <cmath>

using namespace ms;

TEST(OdeTest, rk4_exponential) {
    auto result = ode_rk4([](double, double y) { return y; }, 0.0, 1.0, 1.0, 100);
    EXPECT_NEAR(result.y.back(), std::exp(1.0), 1e-3);
}

TEST(PdeTest, heat_diffusion) {
    std::vector<double> x0(11, 0.0);
    x0[5] = 1.0;
    auto result = pde_heat_1d(x0, 0.1, 0.1, 0.01, 10);
    EXPECT_EQ(result.u.size(), 11);
    EXPECT_LT(result.u.back()[5], result.u.front()[5]);
}

TEST(OptimExtTest, golden_section) {
    auto f = [](double x) { return (x - 2.0) * (x - 2.0); };
    const double xmin = golden_section(f, -5.0, 5.0);
    EXPECT_NEAR(xmin, 2.0, 1e-4);
}

TEST(OptimExtTest, newton_1d) {
    auto f = [](double x) { return (x - 3.0) * (x - 3.0); };
    const double xmin = newton_1d(f, 0.0);
    EXPECT_NEAR(xmin, 3.0, 1e-4);
}
