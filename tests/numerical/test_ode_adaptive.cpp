// Wave 56: Adaptive ODE solver tests (RK45, RK23, midpoint, Heun, Adams-Bashforth2, vector)
#define _USE_MATH_DEFINES
#include "ms/ode/ode.hpp"
#include <cmath>
#include <gtest/gtest.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Exact: dy/dt = -y => y(t) = y0 * exp(-t)
static double decay(double t, double y) { (void)t; return -y; }
static double decay_exact(double t, double y0) { return y0 * std::exp(-t); }

// Exact: dy/dt = y => y(t) = y0 * exp(t)
static double growth(double t, double y) { (void)t; return y; }

// -----------------------------------------------------------------------
// RK45 Dormand-Prince
// -----------------------------------------------------------------------
TEST(OdeRK45, ExponentialDecay_MatchesExact) {
    auto r = ms::ode_rk45(decay, 0.0, 1.0, 2.0);
    ASSERT_FALSE(r.t.empty());
    double y_exact = decay_exact(r.t.back(), 1.0);
    EXPECT_NEAR(r.y.back(), y_exact, 1e-5);
}

TEST(OdeRK45, ProducesMultipleSteps) {
    auto r = ms::ode_rk45(decay, 0.0, 1.0, 1.0);
    EXPECT_GT(r.t.size(), 2u);
    EXPECT_EQ(r.t.size(), r.y.size());
}

TEST(OdeRK45, SolutionIsFinite) {
    auto r = ms::ode_rk45(growth, 0.0, 1.0, 1.0);
    for (double y : r.y) EXPECT_TRUE(std::isfinite(y));
}

TEST(OdeRK45, FirstPointIsIC) {
    auto r = ms::ode_rk45(decay, 0.0, 2.5, 1.0);
    ASSERT_FALSE(r.t.empty());
    EXPECT_NEAR(r.t.front(), 0.0, 1e-14);
    EXPECT_NEAR(r.y.front(), 2.5, 1e-14);
}

TEST(OdeRK45, LastTimeIsNearTEnd) {
    auto r = ms::ode_rk45(decay, 0.0, 1.0, 3.0);
    ASSERT_FALSE(r.t.empty());
    EXPECT_NEAR(r.t.back(), 3.0, 1e-10);
}

TEST(OdeRK45, BetterThanEulerSameSteps) {
    // RK45 should be much more accurate than Euler for same end time
    auto r45 = ms::ode_rk45(decay, 0.0, 1.0, 2.0);
    auto re   = ms::ode_euler(decay, 0.0, 1.0, 2.0, 10);
    double exact = decay_exact(2.0, 1.0);
    double err_rk45 = std::abs(r45.y.back() - exact);
    double err_euler = std::abs(re.y.back() - exact);
    EXPECT_LT(err_rk45, err_euler);
}

// -----------------------------------------------------------------------
// RK23 Bogacki-Shampine
// -----------------------------------------------------------------------
TEST(OdeRK23, ExponentialDecay_MatchesExact) {
    auto r = ms::ode_rk23(decay, 0.0, 1.0, 2.0);
    ASSERT_FALSE(r.t.empty());
    double y_exact = decay_exact(r.t.back(), 1.0);
    EXPECT_NEAR(r.y.back(), y_exact, 1e-4);
}

TEST(OdeRK23, ProducesMultipleSteps) {
    auto r = ms::ode_rk23(decay, 0.0, 1.0, 1.0);
    EXPECT_GT(r.t.size(), 2u);
}

TEST(OdeRK23, SolutionIsFinite) {
    auto r = ms::ode_rk23(decay, 0.0, 1.0, 2.0);
    for (double y : r.y) EXPECT_TRUE(std::isfinite(y));
}

TEST(OdeRK23, LastTimeNearTEnd) {
    auto r = ms::ode_rk23(decay, 0.0, 1.0, 1.5);
    EXPECT_NEAR(r.t.back(), 1.5, 1e-10);
}

// -----------------------------------------------------------------------
// Cash-Karp RK45 (embedded 4th/5th order pair)
// -----------------------------------------------------------------------
TEST(OdeCashKarp, ExponentialDecay_MatchesExact) {
    auto r = ms::ode_cashkarp(decay, 0.0, 1.0, 2.0);
    ASSERT_FALSE(r.t.empty());
    double y_exact = decay_exact(r.t.back(), 1.0);
    EXPECT_NEAR(r.y.back(), y_exact, 1e-5);
}

TEST(OdeCashKarp, ExponentialDecay_MultipleEndTimes) {
    for (double t_end : {0.25, 1.0, 2.0, 5.0}) {
        auto r = ms::ode_cashkarp(decay, 0.0, 1.0, t_end);
        ASSERT_FALSE(r.t.empty());
        double y_exact = decay_exact(r.t.back(), 1.0);
        EXPECT_NEAR(r.y.back(), y_exact, 1e-5)
            << "t_end=" << t_end;
    }
}

TEST(OdeCashKarp, ExponentialGrowth_MatchesExact) {
    for (double t_end : {0.5, 1.0, 2.0}) {
        auto r = ms::ode_cashkarp(growth, 0.0, 1.0, t_end);
        ASSERT_FALSE(r.t.empty());
        double y_exact = std::exp(r.t.back());
        EXPECT_NEAR(r.y.back(), y_exact, 1e-5) << "t_end=" << t_end;
        EXPECT_TRUE(std::isfinite(r.y.back()));
    }
}

TEST(OdeCashKarp, NonlinearForced_AgreesWithRK45) {
    // y' = -y + sin(t): smooth, non-stiff. Both RK45 and Cash-Karp are
    // 5th-order-accurate embedded pairs, so on a smooth problem they should
    // closely agree with each other at the same t_end.
    auto forced = [](double t, double y) { return -y + std::sin(t); };
    const double t_end = 4.0;
    auto rck = ms::ode_cashkarp(forced, 0.0, 1.0, t_end);
    auto r45 = ms::ode_rk45(forced, 0.0, 1.0, t_end);
    ASSERT_FALSE(rck.t.empty());
    ASSERT_FALSE(r45.t.empty());
    EXPECT_NEAR(rck.y.back(), r45.y.back(), 1e-4);
}

TEST(OdeCashKarp, ProducesMultipleSteps) {
    auto r = ms::ode_cashkarp(decay, 0.0, 1.0, 1.0);
    EXPECT_GT(r.t.size(), 2u);
    EXPECT_EQ(r.t.size(), r.y.size());
}

TEST(OdeCashKarp, SolutionIsFinite) {
    auto r = ms::ode_cashkarp(growth, 0.0, 1.0, 1.0);
    for (double y : r.y) EXPECT_TRUE(std::isfinite(y));
}

TEST(OdeCashKarp, FirstPointIsIC) {
    auto r = ms::ode_cashkarp(decay, 0.0, 2.5, 1.0);
    ASSERT_FALSE(r.t.empty());
    EXPECT_NEAR(r.t.front(), 0.0, 1e-14);
    EXPECT_NEAR(r.y.front(), 2.5, 1e-14);
}

TEST(OdeCashKarp, LastTimeIsNearTEnd) {
    auto r = ms::ode_cashkarp(decay, 0.0, 1.0, 3.0);
    ASSERT_FALSE(r.t.empty());
    EXPECT_NEAR(r.t.back(), 3.0, 1e-10);
}

TEST(OdeCashKarp, BetterThanEulerSameSteps) {
    auto rck = ms::ode_cashkarp(decay, 0.0, 1.0, 2.0);
    auto re  = ms::ode_euler(decay, 0.0, 1.0, 2.0, 10);
    double exact = decay_exact(2.0, 1.0);
    double err_ck = std::abs(rck.y.back() - exact);
    double err_euler = std::abs(re.y.back() - exact);
    EXPECT_LT(err_ck, err_euler);
}

TEST(OdeCashKarp, TighterTolerance_ProducesMoreSteps) {
    // Sanity check that step-size control is actually adaptive: tightening
    // rtol/atol should require more (smaller) accepted steps.
    auto loose = ms::ode_cashkarp(decay, 0.0, 1.0, 5.0, 1e-3, 1e-6);
    auto tight = ms::ode_cashkarp(decay, 0.0, 1.0, 5.0, 1e-10, 1e-13);
    EXPECT_LT(loose.t.size(), tight.t.size());
}

TEST(OdeCashKarp, TighterTolerance_SmallerError) {
    const double t_end = 3.0;
    const double exact = decay_exact(t_end, 1.0);
    auto loose = ms::ode_cashkarp(decay, 0.0, 1.0, t_end, 1e-3, 1e-6);
    auto tight = ms::ode_cashkarp(decay, 0.0, 1.0, t_end, 1e-10, 1e-13);
    double err_loose = std::abs(loose.y.back() - exact);
    double err_tight = std::abs(tight.y.back() - exact);
    EXPECT_LT(err_tight, err_loose);
}

TEST(OdeCashKarp, TEndEqualsT0_ReturnsInitialPointOnly) {
    auto r = ms::ode_cashkarp(decay, 1.0, 3.0, 1.0);
    ASSERT_EQ(r.t.size(), 1u);
    ASSERT_EQ(r.y.size(), 1u);
    EXPECT_NEAR(r.t.front(), 1.0, 1e-14);
    EXPECT_NEAR(r.y.front(), 3.0, 1e-14);
}

TEST(OdeCashKarp, TEndLessThanT0_ReturnsInitialPointOnly) {
    auto r = ms::ode_cashkarp(decay, 2.0, 3.0, 1.0);
    ASSERT_EQ(r.t.size(), 1u);
    ASSERT_EQ(r.y.size(), 1u);
    EXPECT_NEAR(r.t.front(), 2.0, 1e-14);
    EXPECT_NEAR(r.y.front(), 3.0, 1e-14);
}

// -----------------------------------------------------------------------
// Midpoint method
// -----------------------------------------------------------------------
TEST(OdeMidpoint, DecayCorrect) {
    auto r = ms::ode_midpoint(decay, 0.0, 1.0, 2.0, 200);
    EXPECT_NEAR(r.y.back(), decay_exact(2.0, 1.0), 1e-5);
}

TEST(OdeMidpoint, SizeCheck) {
    auto r = ms::ode_midpoint(decay, 0.0, 1.0, 1.0, 50);
    EXPECT_EQ(r.t.size(), 51u);
    EXPECT_EQ(r.y.size(), 51u);
}

// -----------------------------------------------------------------------
// RK2 (Heun)
// -----------------------------------------------------------------------
TEST(OdeRK2, DecayCorrect) {
    auto r = ms::ode_rk2(decay, 0.0, 1.0, 2.0, 200);
    EXPECT_NEAR(r.y.back(), decay_exact(2.0, 1.0), 1e-5);
}

TEST(OdeRK2, BetterThanEuler) {
    auto r2 = ms::ode_rk2(decay, 0.0, 1.0, 1.0, 50);
    auto re = ms::ode_euler(decay, 0.0, 1.0, 1.0, 50);
    double exact = decay_exact(1.0, 1.0);
    EXPECT_LT(std::abs(r2.y.back() - exact),
              std::abs(re.y.back() - exact));
}

// -----------------------------------------------------------------------
// Adams-Bashforth 2-step
// -----------------------------------------------------------------------
TEST(OdeAdamsBashforth2, DecayCorrect) {
    auto r = ms::ode_adams_bashforth2(decay, 0.0, 1.0, 2.0, 500);
    EXPECT_NEAR(r.y.back(), decay_exact(2.0, 1.0), 0.01);
}

TEST(OdeAdamsBashforth2, SizeCheck) {
    auto r = ms::ode_adams_bashforth2(decay, 0.0, 1.0, 1.0, 100);
    EXPECT_EQ(r.t.size(), 101u);
}

// -----------------------------------------------------------------------
// Vector ODE (2D harmonic oscillator)
// -----------------------------------------------------------------------
TEST(OdeVectorEuler, HarmonicOscillator_Finite) {
    // y'' + y = 0 => y = [cos(t), -sin(t)] from [1, 0]
    auto f = [](double, const std::vector<double>& y) -> std::vector<double> {
        return {y[1], -y[0]};
    };
    auto r = ms::ode_euler_vec(f, 0.0, {1.0, 0.0}, 2.0 * M_PI, 1000);
    ASSERT_FALSE(r.t.empty());
    for (auto& yv : r.y) {
        EXPECT_TRUE(std::isfinite(yv[0]));
        EXPECT_TRUE(std::isfinite(yv[1]));
    }
}

TEST(OdeVectorRK4, HarmonicOscillator_Accurate) {
    auto f = [](double, const std::vector<double>& y) -> std::vector<double> {
        return {y[1], -y[0]};
    };
    auto r = ms::ode_rk4_vec(f, 0.0, {1.0, 0.0}, 2.0 * M_PI, 500);
    // y[0] should return near 1.0 at t=2*pi
    EXPECT_NEAR(r.y.back()[0], 1.0, 0.01);
    EXPECT_NEAR(r.y.back()[1], 0.0, 0.01);
}

TEST(OdeVectorRK4, SizeConsistent) {
    auto f = [](double, const std::vector<double>& y) -> std::vector<double> {
        return {y[1], -y[0]};
    };
    auto r = ms::ode_rk4_vec(f, 0.0, {1.0, 0.0}, 1.0, 100);
    EXPECT_EQ(r.t.size(), 101u);
    EXPECT_EQ(r.y.size(), 101u);
    for (auto& yv : r.y) EXPECT_EQ(yv.size(), 2u);
}

TEST(OdeVectorEuler, ExponentialDecayVec) {
    auto f = [](double, const std::vector<double>& y) -> std::vector<double> {
        return {-y[0]};
    };
    auto r = ms::ode_euler_vec(f, 0.0, {1.0}, 1.0, 1000);
    EXPECT_NEAR(r.y.back()[0], std::exp(-1.0), 0.005);
}
