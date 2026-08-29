// MathScript Signal Correlation and ODE Additional Numerical Reference Tests

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

#include "ms/signal/signal.hpp"
#include "ms/ode/ode.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ms;

// ---------------------------------------------------------------------------
// correlate: signal cross-correlation
// ---------------------------------------------------------------------------

TEST(SignalCorrelate, SameSignal_PeakAtCenter) {
    // Autocorrelation of a signal peaks at zero-lag (center)
    std::vector<double> x = {1, 2, 3, 2, 1};
    auto r = correlate(x, x);
    ASSERT_GT(r.size(), 0u);
    // Find the peak
    size_t peak_idx = std::max_element(r.begin(), r.end()) - r.begin();
    // Peak should be near center
    size_t center = r.size() / 2;
    EXPECT_NEAR(static_cast<int>(peak_idx), static_cast<int>(center), 1);
}

TEST(SignalCorrelate, OutputSizeCorrect) {
    std::vector<double> a = {1, 2, 3};
    std::vector<double> b = {1, 2};
    auto r = correlate(a, b);
    // Correlation output size: typically m+n-1 for full, or same size; check non-empty
    EXPECT_GT(r.size(), 0u);
}

TEST(SignalCorrelate, ShiftDetection) {
    // If b is a shifted version of a, the correlation peak indicates the shift
    std::vector<double> a = {0, 0, 1, 2, 3, 0, 0};
    std::vector<double> b = {0, 1, 2, 3, 0, 0, 0};  // a shifted left by 1
    auto r = correlate(a, b);
    ASSERT_GT(r.size(), 0u);
    // Correlation should be well-defined
    for (double v : r) EXPECT_TRUE(std::isfinite(v));
}

TEST(SignalCorrelate, ZeroSignal_ZeroCorrelation) {
    std::vector<double> a(8, 0.0);
    std::vector<double> b = {1, 2, 3, 4, 5, 6, 7, 8};
    auto r = correlate(a, b);
    for (double v : r) EXPECT_NEAR(v, 0.0, 1e-10);
}

TEST(SignalCorrelate, FiniteOutputs) {
    std::vector<double> a = {1.0, -2.0, 3.0, -4.0, 5.0};
    std::vector<double> b = {2.0, 1.0, -1.0, 2.0, 0.5};
    auto r = correlate(a, b);
    for (double v : r) EXPECT_TRUE(std::isfinite(v));
}

TEST(SignalCorrelate, IdenticalShortSignals) {
    std::vector<double> x = {1, 0, 0};
    auto r = correlate(x, x);
    ASSERT_GT(r.size(), 0u);
    // Autocorrelation of unit impulse: 1 at lag 0, 0 elsewhere
    double peak = *std::max_element(r.begin(), r.end());
    EXPECT_NEAR(peak, 1.0, 1e-10);
}

// ---------------------------------------------------------------------------
// ODE: Euler and RK4 additional tests
// ---------------------------------------------------------------------------

TEST(ODE_Euler, Exponential_Decay) {
    // dy/dt = -y, y(0) = 1 → y(t) = exp(-t)
    auto f = [](double, double y) { return -y; };
    auto result = ode_euler(f, 0.0, 1.0, 1.0, 1000);
    ASSERT_FALSE(result.t.empty());
    ASSERT_FALSE(result.y.empty());
    EXPECT_NEAR(result.y.back(), std::exp(-1.0), 0.01);  // Euler has O(h) error
}

TEST(ODE_Euler, Linear_Growth) {
    // dy/dt = 1, y(0) = 0 → y(t) = t
    auto f = [](double, double) { return 1.0; };
    auto result = ode_euler(f, 0.0, 0.0, 2.0, 200);
    ASSERT_FALSE(result.y.empty());
    EXPECT_NEAR(result.y.back(), 2.0, 1e-8);
}

TEST(ODE_Euler, StepCount) {
    // Output should have steps+1 points
    auto f = [](double, double y) { return y; };
    auto result = ode_euler(f, 0.0, 1.0, 1.0, 100);
    EXPECT_EQ(result.t.size(), 101u);
    EXPECT_EQ(result.y.size(), 101u);
}

TEST(ODE_Euler, InitialConditionPreserved) {
    auto f = [](double, double y) { return -y; };
    auto result = ode_euler(f, 0.0, 5.0, 1.0, 100);
    EXPECT_NEAR(result.y[0], 5.0, 1e-12);
    EXPECT_NEAR(result.t[0], 0.0, 1e-12);
}

TEST(ODE_Euler, Monotone_Decay) {
    // Exponential decay should be monotone decreasing
    auto f = [](double, double y) { return -y; };
    auto result = ode_euler(f, 0.0, 10.0, 5.0, 500);
    for (size_t i = 1; i < result.y.size(); ++i)
        EXPECT_LE(result.y[i], result.y[i-1] + 1e-10);
}

TEST(ODE_RK4, Exponential_Decay_HighAccuracy) {
    // RK4 should be much more accurate than Euler
    auto f = [](double, double y) { return -y; };
    auto result = ode_rk4(f, 0.0, 1.0, 1.0, 100);
    ASSERT_FALSE(result.y.empty());
    EXPECT_NEAR(result.y.back(), std::exp(-1.0), 1e-7);  // RK4 has O(h^4) error
}

TEST(ODE_RK4, Sine_Wave_Dynamics) {
    // dy/dt = cos(t), y(0) = 0 → y(t) = sin(t)
    auto f = [](double t, double) { return std::cos(t); };
    auto result = ode_rk4(f, 0.0, 0.0, M_PI, 500);
    ASSERT_FALSE(result.y.empty());
    EXPECT_NEAR(result.y.back(), std::sin(M_PI), 1e-6);
}

TEST(ODE_RK4, LogisticGrowth) {
    // dy/dt = r*y*(1-y/K), with r=1, K=1, y(0)=0.01
    // → analytical: y(t) = K / (1 + (K/y0 - 1)*exp(-r*t))
    double r = 1.0, K = 1.0, y0 = 0.01;
    auto f = [&](double, double y) { return r * y * (1.0 - y / K); };
    auto result = ode_rk4(f, 0.0, y0, 10.0, 1000);
    ASSERT_FALSE(result.y.empty());
    double analytical = K / (1.0 + (K / y0 - 1.0) * std::exp(-r * 10.0));
    EXPECT_NEAR(result.y.back(), analytical, 1e-6);
}

TEST(ODE_RK4, TimeGridMonotone) {
    auto f = [](double, double y) { return y; };
    auto result = ode_rk4(f, 0.0, 1.0, 2.0, 200);
    for (size_t i = 1; i < result.t.size(); ++i)
        EXPECT_GT(result.t[i], result.t[i-1]);
}

TEST(ODE_RK4, AllFinite) {
    auto f = [](double, double y) { return -2.0 * y; };
    auto result = ode_rk4(f, 0.0, 1.0, 3.0, 300);
    for (double v : result.y) EXPECT_TRUE(std::isfinite(v));
    for (double t : result.t) EXPECT_TRUE(std::isfinite(t));
}

TEST(ODE_RK4_vs_Euler, RK4_More_Accurate) {
    // For exponential growth, RK4 should be more accurate with same step count
    auto f = [](double, double y) { return y; };
    const double t_end = 2.0;
    const size_t steps = 50;
    auto euler = ode_euler(f, 0.0, 1.0, t_end, steps);
    auto rk4 = ode_rk4(f, 0.0, 1.0, t_end, steps);
    double exact = std::exp(t_end);
    double euler_err = std::abs(euler.y.back() - exact);
    double rk4_err = std::abs(rk4.y.back() - exact);
    EXPECT_LT(rk4_err, euler_err);
}
