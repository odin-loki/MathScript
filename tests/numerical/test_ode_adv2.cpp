// MathScript Advanced ODE Tests (Wave 51)
// Tests convergence order, accuracy across step sizes, and qualitative properties.

#define _USE_MATH_DEFINES
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>

#include "ms/ode/ode.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ms;

// ---------------------------------------------------------------------------
// Convergence order: Euler O(h), RK4 O(h^4)
// ---------------------------------------------------------------------------

TEST(OdeAdv2, Euler_ConvergenceOrder_IsFirstOrder) {
    // Error ~ C*h => halving h halves the error
    const auto f = [](double, double y) { return -y; };
    const double exact = std::exp(-1.0);

    auto res1 = ode_euler(f, 0.0, 1.0, 1.0, 100);
    auto res2 = ode_euler(f, 0.0, 1.0, 1.0, 200);
    double err1 = std::abs(res1.y.back() - exact);
    double err2 = std::abs(res2.y.back() - exact);
    // ratio should be ~2 for first-order method
    double ratio = err1 / err2;
    EXPECT_GT(ratio, 1.5) << "Euler should be at least first-order convergent";
    EXPECT_LT(ratio, 4.0) << "Euler should not exceed second-order";
}

TEST(OdeAdv2, RK4_ConvergenceOrder_IsFourthOrder) {
    // Error ~ C*h^4 => halving h reduces error by factor 16
    const auto f = [](double, double y) { return -y; };
    const double exact = std::exp(-1.0);

    auto res1 = ode_rk4(f, 0.0, 1.0, 1.0, 10);
    auto res2 = ode_rk4(f, 0.0, 1.0, 1.0, 20);
    double err1 = std::abs(res1.y.back() - exact);
    double err2 = std::abs(res2.y.back() - exact);
    double ratio = err1 / err2;
    EXPECT_GT(ratio, 8.0) << "RK4 halving-h ratio should be ~16 (4th order)";
    EXPECT_LT(ratio, 100.0);
}

// ---------------------------------------------------------------------------
// ODE: dy/dt = 3*t^2 => y = t^3 + C (polynomial, exact for RK4)
// ---------------------------------------------------------------------------

TEST(OdeAdv2, RK4_PolynomialRHS_Exact) {
    const auto f = [](double t, double) { return 3.0 * t * t; };
    auto result = ode_rk4(f, 0.0, 0.0, 2.0, 200);
    ASSERT_FALSE(result.y.empty());
    // y(2) = 2^3 = 8
    EXPECT_NEAR(result.y.back(), 8.0, 1e-6);
}

TEST(OdeAdv2, Euler_PolynomialRHS_AcceptableError) {
    const auto f = [](double t, double) { return 3.0 * t * t; };
    auto result = ode_euler(f, 0.0, 0.0, 2.0, 10000);
    ASSERT_FALSE(result.y.empty());
    EXPECT_NEAR(result.y.back(), 8.0, 0.01);
}

// ---------------------------------------------------------------------------
// ODE: logistic growth dy/dt = r*y*(1 - y/K)
// ---------------------------------------------------------------------------

TEST(OdeAdv2, RK4_LogisticGrowth_ConvergesToK) {
    const double r = 2.0, K = 10.0;
    const auto f = [r, K](double, double y) { return r * y * (1.0 - y / K); };
    // Start well below K
    auto result = ode_rk4(f, 0.0, 1.0, 5.0, 500);
    ASSERT_FALSE(result.y.empty());
    // Should approach K = 10
    EXPECT_NEAR(result.y.back(), K, 0.1);
}

TEST(OdeAdv2, RK4_LogisticGrowth_MonotoneIncrease) {
    const double r = 1.0, K = 5.0;
    const auto f = [r, K](double, double y) { return r * y * (1.0 - y / K); };
    auto result = ode_rk4(f, 0.0, 0.5, 3.0, 300);
    // y starts below K, should increase monotonically
    for (size_t i = 1; i < result.y.size(); ++i)
        EXPECT_GE(result.y[i], result.y[i - 1] - 1e-10);
}

// ---------------------------------------------------------------------------
// ODE: simple harmonic component via dy/dt = -omega^2 * y (first-order approx)
// ---------------------------------------------------------------------------

TEST(OdeAdv2, RK4_HarmonicComponent_AccuracyAtPiOver2) {
    // dy/dt = cos(t), y(0)=0 => y = sin(t); y(pi/2)=1
    const auto f = [](double t, double) { return std::cos(t); };
    auto result = ode_rk4(f, 0.0, 0.0, M_PI / 2.0, 500);
    ASSERT_FALSE(result.y.empty());
    EXPECT_NEAR(result.y.back(), 1.0, 1e-8);
}

TEST(OdeAdv2, RK4_HarmonicComponent_AccuracyAtPi) {
    // y(pi) = sin(pi) = 0
    const auto f = [](double t, double) { return std::cos(t); };
    auto result = ode_rk4(f, 0.0, 0.0, M_PI, 1000);
    ASSERT_FALSE(result.y.empty());
    EXPECT_NEAR(result.y.back(), 0.0, 1e-7);
}

// ---------------------------------------------------------------------------
// ODE initial condition sensitivity
// ---------------------------------------------------------------------------

TEST(OdeAdv2, RK4_LinearODE_ICScaling) {
    // dy/dt = -y: solution is y0 * exp(-t); linear in y0
    const auto f = [](double, double y) { return -y; };
    auto r1 = ode_rk4(f, 0.0, 1.0, 1.0, 100);
    auto r2 = ode_rk4(f, 0.0, 2.0, 1.0, 100);
    ASSERT_FALSE(r1.y.empty());
    ASSERT_FALSE(r2.y.empty());
    // y2(t) should be 2 * y1(t) (linear scaling)
    EXPECT_NEAR(r2.y.back(), 2.0 * r1.y.back(), 1e-10);
}

// ---------------------------------------------------------------------------
// ODE output properties
// ---------------------------------------------------------------------------

TEST(OdeAdv2, Euler_TimeVectorMatchesStepSize) {
    const auto f = [](double, double y) { return -y; };
    size_t steps = 100;
    double t0 = 0.0, t_end = 2.0;
    auto result = ode_euler(f, t0, 1.0, t_end, steps);
    double dt = (t_end - t0) / steps;
    EXPECT_NEAR(result.t[0], t0, 1e-14);
    EXPECT_NEAR(result.t.back(), t_end, 1e-10);
    for (size_t i = 1; i < result.t.size(); ++i)
        EXPECT_NEAR(result.t[i] - result.t[i - 1], dt, 1e-12);
}

TEST(OdeAdv2, RK4_TimeVectorMatchesStepSize) {
    const auto f = [](double, double y) { return -y; };
    size_t steps = 50;
    double t0 = 1.0, t_end = 3.0;
    auto result = ode_rk4(f, t0, 1.0, t_end, steps);
    double dt = (t_end - t0) / steps;
    EXPECT_NEAR(result.t[0], t0, 1e-14);
    EXPECT_NEAR(result.t.back(), t_end, 1e-10);
    for (size_t i = 1; i < result.t.size(); ++i)
        EXPECT_NEAR(result.t[i] - result.t[i - 1], dt, 1e-12);
}

TEST(OdeAdv2, Euler_AllValuesFinite) {
    const auto f = [](double, double y) { return -0.5 * y; };
    auto result = ode_euler(f, 0.0, 10.0, 5.0, 1000);
    for (double v : result.y)
        EXPECT_TRUE(std::isfinite(v));
}

TEST(OdeAdv2, RK4_AllValuesFinite) {
    const auto f = [](double t, double y) { return std::sin(t) - y; };
    auto result = ode_rk4(f, 0.0, 0.0, 10.0, 500);
    for (double v : result.y)
        EXPECT_TRUE(std::isfinite(v));
}

// ---------------------------------------------------------------------------
// Euler vs RK4 accuracy comparison
// ---------------------------------------------------------------------------

TEST(OdeAdv2, RK4_FarMoreAccurateThanEuler_SameStepCount) {
    // With just 20 steps, RK4 should be orders of magnitude more accurate
    const auto f = [](double, double y) { return -y; };
    const double exact = std::exp(-1.0);
    auto rk4  = ode_rk4(f, 0.0, 1.0, 1.0, 20);
    auto euler = ode_euler(f, 0.0, 1.0, 1.0, 20);
    double rk4_err   = std::abs(rk4.y.back()  - exact);
    double euler_err = std::abs(euler.y.back() - exact);
    EXPECT_LT(rk4_err, euler_err / 100.0)
        << "RK4 should be at least 100x more accurate than Euler for 20 steps";
}
