// MathScript ODE Numerical Reference Tests
// Verifies ODE solvers against known analytical solutions

#define _USE_MATH_DEFINES
#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#include "ms/ode/ode.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ms;

// ---------------------------------------------------------------------------
// Euler method: exponential decay
// dy/dt = -y, y(0) = 1 => y(t) = e^{-t}
// ---------------------------------------------------------------------------

TEST(OdeNumerical, Euler_ExponentialDecay_EndValue) {
    const auto f = [](double /*t*/, double y) { return -y; };
    const auto result = ode_euler(f, 0.0, 1.0, 1.0, 10000);
    ASSERT_FALSE(result.y.empty());
    EXPECT_NEAR(result.y.back(), std::exp(-1.0), 0.001);  // Euler has first-order error
}

TEST(OdeNumerical, Euler_ExponentialGrowth_EndValue) {
    // dy/dt = y, y(0) = 1 => y(t) = e^t
    const auto f = [](double /*t*/, double y) { return y; };
    const auto result = ode_euler(f, 0.0, 1.0, 1.0, 10000);
    ASSERT_FALSE(result.y.empty());
    EXPECT_NEAR(result.y.back(), std::exp(1.0), 0.001);
}

TEST(OdeNumerical, Euler_ConstantRHS) {
    // dy/dt = 2, y(0) = 0 => y(t) = 2t; at t=1 => y=2
    const auto f = [](double /*t*/, double /*y*/) { return 2.0; };
    const auto result = ode_euler(f, 0.0, 0.0, 1.0, 1000);
    ASSERT_FALSE(result.y.empty());
    EXPECT_NEAR(result.y.back(), 2.0, 1e-10);  // Exact for constant RHS
}

TEST(OdeNumerical, Euler_LinearRHS) {
    // dy/dt = t, y(0) = 0 => y(t) = t^2/2; at t=2 => y=2
    const auto f = [](double t, double /*y*/) { return t; };
    const auto result = ode_euler(f, 0.0, 0.0, 2.0, 100000);
    ASSERT_FALSE(result.y.empty());
    EXPECT_NEAR(result.y.back(), 2.0, 0.01);
}

TEST(OdeNumerical, Euler_OutputLength) {
    const auto f = [](double /*t*/, double y) { return -y; };
    const size_t steps = 100;
    const auto result = ode_euler(f, 0.0, 1.0, 1.0, steps);
    EXPECT_EQ(result.t.size(), steps + 1);
    EXPECT_EQ(result.y.size(), steps + 1);
}

TEST(OdeNumerical, Euler_TimeVector_Monotone) {
    const auto f = [](double /*t*/, double y) { return -y; };
    const auto result = ode_euler(f, 0.0, 1.0, 2.0, 50);
    for (size_t i = 1; i < result.t.size(); ++i)
        EXPECT_GT(result.t[i], result.t[i-1]);
}

// ---------------------------------------------------------------------------
// RK4: higher accuracy than Euler
// ---------------------------------------------------------------------------

TEST(OdeNumerical, RK4_ExponentialDecay_Accuracy) {
    // dy/dt = -y => y(t) = e^{-t}; RK4 should be much more accurate
    const auto f = [](double /*t*/, double y) { return -y; };
    const auto result = ode_rk4(f, 0.0, 1.0, 1.0, 100);
    ASSERT_FALSE(result.y.empty());
    EXPECT_NEAR(result.y.back(), std::exp(-1.0), 1e-8);  // RK4 much more accurate
}

TEST(OdeNumerical, RK4_IsMoreAccurateThanEuler) {
    // For same step count, RK4 should be closer to exact solution
    const auto f = [](double /*t*/, double y) { return -y; };
    const auto rk4_result  = ode_rk4(f, 0.0, 1.0, 1.0, 50);
    const auto euler_result = ode_euler(f, 0.0, 1.0, 1.0, 50);
    const double exact = std::exp(-1.0);
    const double rk4_err   = std::abs(rk4_result.y.back()  - exact);
    const double euler_err = std::abs(euler_result.y.back() - exact);
    EXPECT_LT(rk4_err, euler_err);
}

TEST(OdeNumerical, RK4_SineOscillator) {
    // dy/dt = cos(t), y(0) = 0 => y(t) = sin(t); at t=pi/2 => y=1
    const auto f = [](double t, double /*y*/) { return std::cos(t); };
    const auto result = ode_rk4(f, 0.0, 0.0, M_PI / 2.0, 1000);
    ASSERT_FALSE(result.y.empty());
    EXPECT_NEAR(result.y.back(), 1.0, 1e-6);
}

TEST(OdeNumerical, RK4_HarmonicOscillator_Component) {
    // dy/dt = -y, y(0) = 1 => y(t) = e^{-t} for 1D component
    const auto f = [](double /*t*/, double y) { return -y; };
    const auto result = ode_rk4(f, 0.0, 1.0, 2.0, 200);
    ASSERT_FALSE(result.y.empty());
    EXPECT_NEAR(result.y.back(), std::exp(-2.0), 1e-6);
}

TEST(OdeNumerical, RK4_ConstantSolution) {
    // dy/dt = 0, y(0) = 5 => y(t) = 5 for all t
    const auto f = [](double /*t*/, double /*y*/) { return 0.0; };
    const auto result = ode_rk4(f, 0.0, 5.0, 10.0, 100);
    for (double y : result.y)
        EXPECT_NEAR(y, 5.0, 1e-12);
}

TEST(OdeNumerical, RK4_OutputLength) {
    const auto f = [](double /*t*/, double y) { return -y; };
    const size_t steps = 50;
    const auto result = ode_rk4(f, 0.0, 1.0, 1.0, steps);
    EXPECT_EQ(result.t.size(), steps + 1);
    EXPECT_EQ(result.y.size(), steps + 1);
}
