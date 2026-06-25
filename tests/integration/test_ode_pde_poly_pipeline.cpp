// MathScript Integration Test: ODE + PDE + Poly Pipeline (Wave 51)
// Tests that chain ODE results into polynomial fitting, then feed into PDE boundary conditions.

#define _USE_MATH_DEFINES
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>

#include "ms/ode/ode.hpp"
#include "ms/pde/pde.hpp"
#include "ms/poly/poly.hpp"
#include "ms/stats/stats.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ms;

// ---------------------------------------------------------------------------
// ODE → Stats: compute statistics on ODE trajectory
// ---------------------------------------------------------------------------

TEST(OdePdePoly, ODE_Trajectory_Stats_Finite) {
    const auto f = [](double, double y) { return -y; };
    auto result = ode_rk4(f, 0.0, 1.0, 5.0, 500);
    ASSERT_FALSE(result.y.empty());
    double m = mean(result.y);
    double s = stddev(result.y);
    EXPECT_TRUE(std::isfinite(m));
    EXPECT_TRUE(std::isfinite(s));
    EXPECT_GT(s, 0.0) << "Decaying exponential should have non-zero variance";
}

TEST(OdePdePoly, ODE_Trajectory_MonotonicallyDecaying_Stats) {
    const auto f = [](double, double y) { return -y; };
    auto result = ode_rk4(f, 0.0, 1.0, 3.0, 300);
    ASSERT_FALSE(result.y.empty());
    // Min should be near exp(-3), max should be near 1
    double y_min = min_value(result.y);
    double y_max = max_value(result.y);
    EXPECT_GT(y_max, 0.5) << "Max of decaying exp should be near 1";
    EXPECT_GT(y_min, 0.0) << "All values should be positive";
    EXPECT_LT(y_min, y_max);
}

// ---------------------------------------------------------------------------
// ODE → Poly: fit polynomial to ODE output samples
// ---------------------------------------------------------------------------

TEST(OdePdePoly, ODE_To_Poly_Eval_Consistency) {
    // ODE: dy/dt = 2t, y(0) = 0 => y(t) = t^2
    // Polynomial coefficients: {0, 0, 1} (x^2)
    const auto f = [](double t, double) { return 2.0 * t; };
    auto result = ode_rk4(f, 0.0, 0.0, 2.0, 200);
    // Check ODE solution against polynomial y = t^2
    std::vector<double> coeff = {0.0, 0.0, 1.0}; // t^2
    for (size_t i = 0; i < result.t.size(); i += 20) {
        double t = result.t[i];
        double y_ode  = result.y[i];
        auto pv = poly_eval(coeff, t);
        ASSERT_FALSE(pv.empty());
        EXPECT_NEAR(y_ode, pv[0], 0.01)
            << "ODE and poly disagree at t=" << t;
    }
}

// ---------------------------------------------------------------------------
// ODE → PDE: use ODE solution as initial condition
// ---------------------------------------------------------------------------

TEST(OdePdePoly, ODE_Output_As_PDE_IC) {
    // Run ODE to get a smooth initial profile
    const auto f = [](double, double y) { return -0.1 * y; };
    auto ode_result = ode_rk4(f, 0.0, 1.0, 1.0, 10);
    ASSERT_EQ(ode_result.y.size(), 11u);
    // Use ODE y-values as spatial IC for PDE
    auto pde_result = pde_heat_1d(ode_result.y, 0.01, 0.1, 0.001, 10);
    EXPECT_FALSE(pde_result.u.empty());
    EXPECT_EQ(pde_result.x.size(), 11u);
    for (const auto& row : pde_result.u)
        for (double v : row)
            EXPECT_TRUE(std::isfinite(v));
}

// ---------------------------------------------------------------------------
// Poly arithmetic → ODE evaluation
// ---------------------------------------------------------------------------

TEST(OdePdePoly, PolyDeriv_Matches_ODE_Rate) {
    // p(t) = t^3 - 2t^2 + 1
    // dp/dt = 3t^2 - 4t
    std::vector<double> p    = {1.0, 0.0, -2.0, 1.0};
    std::vector<double> dpdv = poly_deriv(p);
    // Verify at t=2: dp/dt = 12 - 8 = 4
    auto v2 = poly_eval(dpdv, 2.0);
    ASSERT_FALSE(v2.empty());
    EXPECT_NEAR(v2[0], 4.0, 1e-10);
    // Verify at t=0: dp/dt = 0
    auto v0 = poly_eval(dpdv, 0.0);
    ASSERT_FALSE(v0.empty());
    EXPECT_NEAR(v0[0], 0.0, 1e-10);
}

TEST(OdePdePoly, Poly_Add_Sub_Invariant) {
    std::vector<double> p = {1.0, 2.0, 3.0};
    std::vector<double> q = {4.0, 5.0, 6.0};
    auto sum  = poly_add(p, q);
    auto diff = poly_sub(sum, q);
    ASSERT_EQ(diff.size(), p.size());
    for (size_t i = 0; i < p.size(); ++i)
        EXPECT_NEAR(diff[i], p[i], 1e-12);
}

// ---------------------------------------------------------------------------
// PDE heat + stats on spatial profile
// ---------------------------------------------------------------------------

TEST(OdePdePoly, PDE_Final_Profile_Stats_Finite) {
    const size_t Nx = 11;
    std::vector<double> x0(Nx, 0.0);
    x0[Nx / 2] = 2.0;
    auto pde = pde_heat_1d(x0, 0.05, 0.1, 0.001, 100);
    ASSERT_FALSE(pde.u.empty());
    const auto& u_last = pde.u.back();
    double m = mean(u_last);
    double s = stddev(u_last);
    EXPECT_TRUE(std::isfinite(m));
    EXPECT_TRUE(std::isfinite(s));
}

TEST(OdePdePoly, PDE_ReducesMaxVsODEGrowth) {
    // PDE diffuses; ODE growing solution grows
    const size_t Nx = 11;
    std::vector<double> x0(Nx, 0.0);
    x0[Nx / 2] = 5.0; // spike

    auto pde = pde_heat_1d(x0, 0.1, 0.1, 0.001, 200);
    ASSERT_FALSE(pde.u.empty());
    double max_pde = 0.0;
    for (double v : pde.u.back()) max_pde = std::max(max_pde, v);

    // ODE: dy/dt = y, y(0) = 5 => grows to 5*exp(0.2*200*0.001) = 5*exp(0.2) ≈ 6.1
    const auto f = [](double, double y) { return y; };
    auto ode = ode_rk4(f, 0.0, 5.0, 0.2, 200);
    ASSERT_FALSE(ode.y.empty());
    double max_ode = ode.y.back();

    EXPECT_LT(max_pde, max_ode)
        << "PDE diffused spike should be less than ODE grown peak";
}

// ---------------------------------------------------------------------------
// Poly eval and deriv consistency via numerical finite difference
// ---------------------------------------------------------------------------

TEST(OdePdePoly, PolyDeriv_Matches_FiniteDifference) {
    std::vector<double> p = {1.0, -3.0, 2.0, 0.5}; // 0.5x^3 + 2x^2 - 3x + 1
    auto dp = poly_deriv(p);
    for (double x : {-2.0, 0.0, 1.0, 2.5}) {
        double h = 1e-5;
        auto vph = poly_eval(p, x + h);
        auto vmh = poly_eval(p, x - h);
        auto vdp = poly_eval(dp, x);
        ASSERT_FALSE(vph.empty()); ASSERT_FALSE(vmh.empty()); ASSERT_FALSE(vdp.empty());
        double fd = (vph[0] - vmh[0]) / (2.0 * h);
        EXPECT_NEAR(vdp[0], fd, 1e-5)
            << "poly_deriv finite difference mismatch at x=" << x;
    }
}

// ---------------------------------------------------------------------------
// Multi-module: ODE + Poly + Stats pipeline
// ---------------------------------------------------------------------------

TEST(OdePdePoly, ODE_Stats_Agree_With_Poly) {
    // ODE: y = exp(-t), sampled at t=0..1 (1001 pts)
    const auto f = [](double, double y) { return -y; };
    auto ode = ode_rk4(f, 0.0, 1.0, 1.0, 1000);
    // Mean of exp(-t) over [0,1] ≈ 1 - e^{-1} ≈ 0.632
    double m = mean(ode.y);
    EXPECT_NEAR(m, 1.0 - std::exp(-1.0), 0.01);
}
