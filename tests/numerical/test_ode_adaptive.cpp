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

// -----------------------------------------------------------------------
// Rosenbrock23 (linearly-implicit 2-stage Rosenbrock-Wanner method)
// -----------------------------------------------------------------------
TEST(OdeRosenbrock23, TEndLessThanT0_ReturnsInitialPointOnly) {
    auto f = [](double, const std::vector<double>& y) -> std::vector<double> {
        return {-y[0]};
    };
    auto r = ms::ode_rosenbrock23_vec(f, 2.0, {3.0}, 1.0, 10);
    ASSERT_EQ(r.t.size(), 1u);
    ASSERT_EQ(r.y.size(), 1u);
    EXPECT_NEAR(r.t.front(), 2.0, 1e-14);
    EXPECT_NEAR(r.y.front()[0], 3.0, 1e-14);
}

TEST(OdeRosenbrock23, ZeroSteps_ReturnsInitialPointOnly) {
    auto f = [](double, const std::vector<double>& y) -> std::vector<double> {
        return {-y[0]};
    };
    auto r = ms::ode_rosenbrock23_vec(f, 0.0, {1.0}, 1.0, 0);
    ASSERT_EQ(r.t.size(), 1u);
    ASSERT_EQ(r.y.size(), 1u);
    EXPECT_NEAR(r.y.front()[0], 1.0, 1e-14);
}

TEST(OdeRosenbrock23, NegativeSteps_ReturnsInitialPointOnly) {
    auto f = [](double, const std::vector<double>& y) -> std::vector<double> {
        return {-y[0]};
    };
    auto r = ms::ode_rosenbrock23_vec(f, 0.0, {1.0}, 1.0, -5);
    ASSERT_EQ(r.t.size(), 1u);
    ASSERT_EQ(r.y.size(), 1u);
    EXPECT_NEAR(r.y.front()[0], 1.0, 1e-14);
}

TEST(OdeRosenbrock23, EmptyY0_ReturnsEmpty) {
    auto f = [](double, const std::vector<double>&) -> std::vector<double> {
        return {};
    };
    auto r = ms::ode_rosenbrock23_vec(f, 0.0, {}, 1.0, 10);
    EXPECT_TRUE(r.t.empty());
    EXPECT_TRUE(r.y.empty());
}

TEST(OdeRosenbrock23, FirstPointIsIC) {
    auto f = [](double, const std::vector<double>& y) -> std::vector<double> {
        return {-y[0]};
    };
    auto r = ms::ode_rosenbrock23_vec(f, 0.5, {2.5}, 1.5, 20);
    ASSERT_FALSE(r.t.empty());
    EXPECT_NEAR(r.t.front(), 0.5, 1e-14);
    EXPECT_NEAR(r.y.front()[0], 2.5, 1e-14);
}

TEST(OdeRosenbrock23, TrajectoryShape_SizeAndMonotoneGrid) {
    auto f = [](double, const std::vector<double>& y) -> std::vector<double> {
        return {-y[0]};
    };
    auto r = ms::ode_rosenbrock23_vec(f, 0.0, {1.0}, 1.0, 25);
    ASSERT_EQ(r.t.size(), 26u);
    ASSERT_EQ(r.y.size(), 26u);
    for (size_t i = 1; i < r.t.size(); ++i) {
        EXPECT_GT(r.t[i], r.t[i - 1]);
    }
    EXPECT_NEAR(r.t.back(), 1.0, 1e-10);
}

TEST(OdeRosenbrock23, StiffDecay_RemainsStableWhereEulerBlowsUp) {
    // y' = -k*y, k large enough that explicit Euler is unstable at this h.
    const double k = 1000.0;
    const auto f_scalar = [k](double, double y) { return -k * y; };
    auto f = [k](double, const std::vector<double>& y) -> std::vector<double> {
        return {-k * y[0]};
    };
    const size_t steps = 100; // h = 0.01, h*k = 10 >> 2 (explicit Euler stability limit)
    const auto fwd = ms::ode_euler(f_scalar, 0.0, 1.0, 1.0, steps);
    const auto rb = ms::ode_rosenbrock23_vec(f, 0.0, {1.0}, 1.0, static_cast<int>(steps));
    ASSERT_FALSE(fwd.y.empty());
    ASSERT_FALSE(rb.y.empty());
    EXPECT_TRUE(!std::isfinite(fwd.y.back()) || std::abs(fwd.y.back()) > 1e4);
    EXPECT_TRUE(std::isfinite(rb.y.back()[0]));
    EXPECT_LT(std::abs(rb.y.back()[0]), 1.0);
}

TEST(OdeRosenbrock23, StiffDecay_MatchesAnalytic) {
    // y' = -k*y, y(0) = 1 => y(t) = exp(-k*t)
    const double k = 5.0;
    auto f = [k](double, const std::vector<double>& y) -> std::vector<double> {
        return {-k * y[0]};
    };
    const auto r = ms::ode_rosenbrock23_vec(f, 0.0, {1.0}, 1.0, 50);
    ASSERT_FALSE(r.y.empty());
    const double exact = std::exp(-k * 1.0);
    EXPECT_NEAR(r.y.back()[0], exact, 0.01);
}

TEST(OdeRosenbrock23, StiffCosProblem_MatchesAnalytic) {
    // dy/dt = -k*(y - cos(t)) - sin(t), analytic solution y = cos(t)
    const double k = 100.0;
    auto f = [k](double t, const std::vector<double>& y) -> std::vector<double> {
        return {-k * (y[0] - std::cos(t)) - std::sin(t)};
    };
    const auto r = ms::ode_rosenbrock23_vec(f, 0.0, {1.0}, 1.0, 100);
    ASSERT_FALSE(r.y.empty());
    EXPECT_TRUE(std::isfinite(r.y.back()[0]));
    EXPECT_NEAR(r.y.back()[0], std::cos(1.0), 0.05);
}

TEST(OdeRosenbrock23, NonStiff_ExponentialGrowth_MatchesAnalytic) {
    // y' = y, y(0) = 1 => y(t) = exp(t); Rosenbrock23 should be reasonable
    // on easy non-stiff problems too, not just stiff ones.
    auto f = [](double, const std::vector<double>& y) -> std::vector<double> {
        return {y[0]};
    };
    const auto r = ms::ode_rosenbrock23_vec(f, 0.0, {1.0}, 1.0, 50);
    ASSERT_FALSE(r.y.empty());
    EXPECT_NEAR(r.y.back()[0], std::exp(1.0), 0.01);
}

TEST(OdeRosenbrock23, NonStiff_ExponentialDecay_MatchesAnalytic) {
    auto f = [](double, const std::vector<double>& y) -> std::vector<double> {
        return {-y[0]};
    };
    const auto r = ms::ode_rosenbrock23_vec(f, 0.0, {1.0}, 1.0, 50);
    ASSERT_FALSE(r.y.empty());
    EXPECT_NEAR(r.y.back()[0], std::exp(-1.0), 1e-3);
}

TEST(OdeRosenbrock23, CoupledLinearSystem2D_MatchesAnalytic) {
    // y1' = -2*y1 + y2, y2' = -3*y2, y1(0) = y2(0) = 1.
    // Closed form: y2(t) = exp(-3t); y1(t) = exp(-2t)*(2 - exp(-t)).
    // Off-diagonal coupling exercises the full 2x2 finite-difference
    // Jacobian and dense linear solve, not just a diagonal system.
    auto f = [](double, const std::vector<double>& y) -> std::vector<double> {
        return {-2.0 * y[0] + y[1], -3.0 * y[1]};
    };
    const auto r = ms::ode_rosenbrock23_vec(f, 0.0, {1.0, 1.0}, 1.0, 200);
    ASSERT_FALSE(r.y.empty());
    const double t_end = r.t.back();
    const double exact_y2 = std::exp(-3.0 * t_end);
    const double exact_y1 = std::exp(-2.0 * t_end) * (2.0 - std::exp(-t_end));
    EXPECT_NEAR(r.y.back()[0], exact_y1, 0.01);
    EXPECT_NEAR(r.y.back()[1], exact_y2, 0.01);
}

TEST(OdeRosenbrock23, SolutionRemainsFinite) {
    const double k = 300.0;
    auto f = [k](double, const std::vector<double>& y) -> std::vector<double> {
        return {-k * y[0]};
    };
    const auto r = ms::ode_rosenbrock23_vec(f, 0.0, {1.0}, 2.0, 400);
    for (const auto& state : r.y) {
        for (double v : state) {
            EXPECT_TRUE(std::isfinite(v));
        }
    }
}

TEST(OdeRosenbrock23, ConvergesWithMoreSteps) {
    const double k = 100.0;
    auto f = [k](double t, const std::vector<double>& y) -> std::vector<double> {
        return {-k * (y[0] - std::cos(t)) - std::sin(t)};
    };
    const double exact = std::cos(1.0);
    const auto coarse = ms::ode_rosenbrock23_vec(f, 0.0, {1.0}, 1.0, 20);
    const auto fine = ms::ode_rosenbrock23_vec(f, 0.0, {1.0}, 1.0, 400);
    ASSERT_FALSE(coarse.y.empty());
    ASSERT_FALSE(fine.y.empty());
    const double err_coarse = std::abs(coarse.y.back()[0] - exact);
    const double err_fine = std::abs(fine.y.back()[0] - exact);
    EXPECT_LT(err_fine, err_coarse);
}

TEST(OdeRosenbrock23, ScalarOverload_MatchesVecWrapper) {
    const double k = 20.0;
    auto f_scalar = [k](double, double y) { return -k * y; };
    auto f_vec = [k](double, const std::vector<double>& y) -> std::vector<double> {
        return {-k * y[0]};
    };
    const auto r_scalar = ms::ode_rosenbrock23(f_scalar, 0.0, 1.0, 1.0, 60);
    const auto r_vec = ms::ode_rosenbrock23_vec(f_vec, 0.0, {1.0}, 1.0, 60);
    ASSERT_EQ(r_scalar.y.size(), r_vec.y.size());
    for (size_t i = 0; i < r_scalar.y.size(); ++i) {
        EXPECT_NEAR(r_scalar.y[i], r_vec.y[i][0], 1e-12);
    }
}

TEST(OdeRosenbrock23, ScalarOverload_StiffDecay_MatchesAnalytic) {
    const double k = 10.0;
    auto f = [k](double, double y) { return -k * y; };
    const auto r = ms::ode_rosenbrock23(f, 0.0, 1.0, 1.0, 80);
    ASSERT_FALSE(r.y.empty());
    EXPECT_NEAR(r.y.back(), std::exp(-k * 1.0), 0.01);
}
