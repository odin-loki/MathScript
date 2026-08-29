#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
#include "ms/ode/ode.hpp"
#include "ms/pde/pde.hpp"

// ---------------------------------------------------------------------------
// OdeAdvanced tests
// ---------------------------------------------------------------------------

// 1a. Exponential decay – Euler (loose tolerance)
TEST(OdeAdvanced, ExponentialDecayEuler) {
    ms::OdeFunc f = [](double /*t*/, double y) { return -y; };
    auto res = ms::ode_euler(f, 0.0, 1.0, 1.0, 1000);
    ASSERT_FALSE(res.t.empty());
    double y_final = res.y.back();
    EXPECT_NEAR(y_final, std::exp(-1.0), 0.05);
}

// 1b. Exponential decay – RK4 (tight tolerance)
TEST(OdeAdvanced, ExponentialDecayRK4) {
    ms::OdeFunc f = [](double /*t*/, double y) { return -y; };
    auto res = ms::ode_rk4(f, 0.0, 1.0, 1.0, 1000);
    ASSERT_FALSE(res.t.empty());
    double y_final = res.y.back();
    EXPECT_NEAR(y_final, std::exp(-1.0), 0.001);
}

// 2. Simple growth – RK4, y(1) = e
TEST(OdeAdvanced, SimpleGrowthRK4) {
    ms::OdeFunc f = [](double /*t*/, double y) { return y; };
    auto res = ms::ode_rk4(f, 0.0, 1.0, 1.0, 1000);
    ASSERT_FALSE(res.t.empty());
    double y_final = res.y.back();
    EXPECT_NEAR(y_final, std::exp(1.0), 0.001);
}

// 3a. Step count effect – more steps give better Euler accuracy
TEST(OdeAdvanced, StepCountEffectEuler) {
    ms::OdeFunc f = [](double /*t*/, double y) { return -y; };
    double exact = std::exp(-1.0);
    auto coarse = ms::ode_euler(f, 0.0, 1.0, 1.0, 100);
    auto fine   = ms::ode_euler(f, 0.0, 1.0, 1.0, 1000);
    double err_coarse = std::abs(coarse.y.back() - exact);
    double err_fine   = std::abs(fine.y.back()   - exact);
    EXPECT_LT(err_fine, err_coarse);
}

// 3b. Step count effect verified for growth ODE as well
TEST(OdeAdvanced, StepCountEffectEulerGrowth) {
    ms::OdeFunc f = [](double /*t*/, double y) { return y; };
    double exact = std::exp(1.0);
    auto coarse = ms::ode_euler(f, 0.0, 1.0, 1.0, 50);
    auto fine   = ms::ode_euler(f, 0.0, 1.0, 1.0, 500);
    double err_coarse = std::abs(coarse.y.back() - exact);
    double err_fine   = std::abs(fine.y.back()   - exact);
    EXPECT_LT(err_fine, err_coarse);
}

// 4a. ODE result structure – Euler
TEST(OdeAdvanced, ResultStructureEuler) {
    ms::OdeFunc f = [](double /*t*/, double y) { return -y; };
    auto res = ms::ode_euler(f, 0.0, 1.0, 2.0, 200);
    ASSERT_EQ(res.t.size(), res.y.size());
    EXPECT_DOUBLE_EQ(res.t.front(), 0.0);
    EXPECT_DOUBLE_EQ(res.y.front(), 1.0);
    EXPECT_NEAR(res.t.back(), 2.0, 1e-9);
}

// 4b. ODE result structure – RK4
TEST(OdeAdvanced, ResultStructureRK4) {
    ms::OdeFunc f = [](double /*t*/, double y) { return -y; };
    auto res = ms::ode_rk4(f, 0.5, 2.0, 3.5, 150);
    ASSERT_EQ(res.t.size(), res.y.size());
    EXPECT_DOUBLE_EQ(res.t.front(), 0.5);
    EXPECT_DOUBLE_EQ(res.y.front(), 2.0);
    EXPECT_NEAR(res.t.back(), 3.5, 1e-9);
}

// 5. RK4 vs Euler accuracy comparison
TEST(OdeAdvanced, RK4MoreAccurateThanEuler) {
    ms::OdeFunc f = [](double /*t*/, double y) { return -y; };
    double exact = std::exp(-1.0);
    // Use the same (moderate) step count so Euler is visibly less accurate
    auto euler = ms::ode_euler(f, 0.0, 1.0, 1.0, 100);
    auto rk4   = ms::ode_rk4(f,   0.0, 1.0, 1.0, 100);
    double err_euler = std::abs(euler.y.back() - exact);
    double err_rk4   = std::abs(rk4.y.back()   - exact);
    EXPECT_LT(err_rk4, err_euler);
}

// 10. Zero derivative – y stays constant
TEST(OdeAdvanced, ZeroDerivativeEuler) {
    ms::OdeFunc f = [](double /*t*/, double /*y*/) { return 0.0; };
    auto res = ms::ode_euler(f, 0.0, 5.0, 3.0, 300);
    for (double v : res.y) {
        EXPECT_DOUBLE_EQ(v, 5.0);
    }
}

TEST(OdeAdvanced, ZeroDerivativeRK4) {
    ms::OdeFunc f = [](double /*t*/, double /*y*/) { return 0.0; };
    auto res = ms::ode_rk4(f, 0.0, 5.0, 3.0, 300);
    for (double v : res.y) {
        EXPECT_DOUBLE_EQ(v, 5.0);
    }
}

// 11. Constant derivative dy/dt = 1 → y(t) = t
TEST(OdeAdvanced, ConstantDerivativeEuler) {
    ms::OdeFunc f = [](double /*t*/, double /*y*/) { return 1.0; };
    auto res = ms::ode_euler(f, 0.0, 0.0, 5.0, 500);
    ASSERT_EQ(res.t.size(), res.y.size());
    for (size_t i = 0; i < res.t.size(); ++i) {
        EXPECT_NEAR(res.y[i], res.t[i], 1e-9);
    }
}

TEST(OdeAdvanced, ConstantDerivativeRK4) {
    ms::OdeFunc f = [](double /*t*/, double /*y*/) { return 1.0; };
    auto res = ms::ode_rk4(f, 0.0, 0.0, 5.0, 500);
    ASSERT_EQ(res.t.size(), res.y.size());
    for (size_t i = 0; i < res.t.size(); ++i) {
        EXPECT_NEAR(res.y[i], res.t[i], 1e-9);
    }
}

// 12. RK4 4th-order convergence: halving step size reduces error by ~16x
TEST(OdeAdvanced, RK4FourthOrderConvergence) {
    ms::OdeFunc f = [](double /*t*/, double y) { return -y; };
    double exact = std::exp(-1.0);

    auto res_coarse = ms::ode_rk4(f, 0.0, 1.0, 1.0, 10);
    auto res_fine   = ms::ode_rk4(f, 0.0, 1.0, 1.0, 20);  // double steps

    double err_coarse = std::abs(res_coarse.y.back() - exact);
    double err_fine   = std::abs(res_fine.y.back()   - exact);

    // Doubling steps ≈ halving h → error shrinks by 2^4 = 16.
    // Use a lenient threshold (factor > 10) to allow for floating-point effects.
    EXPECT_GT(err_coarse / err_fine, 10.0);
}

// ---------------------------------------------------------------------------
// PdeAdvanced tests
// ---------------------------------------------------------------------------

namespace {
// Build a flat initial condition with a central spike
std::vector<double> make_spike(size_t n) {
    std::vector<double> u(n, 0.0);
    u[n / 2] = 1.0;
    return u;
}

// Build a uniform initial condition
std::vector<double> make_uniform(size_t n, double val = 1.0) {
    return std::vector<double>(n, val);
}
}  // namespace

// 6. Heat conservation – total heat should not increase over time
TEST(PdeAdvanced, HeatConservation) {
    const size_t nx = 21;
    double dx = 1.0 / (nx - 1);
    double alpha = 0.01;
    double dt = 0.4 * dx * dx / alpha;  // stable CFL
    auto x0 = make_uniform(nx, 1.0);

    auto res = ms::pde_heat_1d(x0, alpha, dx, dt, 50);
    ASSERT_GE(res.u.size(), 2u);

    double sum_initial = std::accumulate(res.u.front().begin(), res.u.front().end(), 0.0);
    double sum_final   = std::accumulate(res.u.back().begin(),  res.u.back().end(),  0.0);

    // Heat may leak through boundaries, so final sum ≤ initial sum (with tolerance).
    EXPECT_LE(sum_final, sum_initial * 1.01);
}

// 7. Diffusion direction – central spike spreads (max value decreases)
TEST(PdeAdvanced, DiffusionSpreadsSpike) {
    const size_t nx = 31;
    double dx = 1.0 / (nx - 1);
    double alpha = 0.01;
    double dt = 0.4 * dx * dx / alpha;
    auto x0 = make_spike(nx);

    auto res = ms::pde_heat_1d(x0, alpha, dx, dt, 100);
    ASSERT_GE(res.u.size(), 2u);

    double max_initial = *std::max_element(res.u.front().begin(), res.u.front().end());
    double max_final   = *std::max_element(res.u.back().begin(),  res.u.back().end());

    EXPECT_LT(max_final, max_initial);
}

// 8. PDE result structure
TEST(PdeAdvanced, ResultStructure) {
    const size_t nx = 11;
    double dx = 0.1;
    double alpha = 0.01;
    double dt = 0.4 * dx * dx / alpha;
    size_t steps = 10;
    auto x0 = make_uniform(nx);

    auto res = ms::pde_heat_1d(x0, alpha, dx, dt, steps);

    EXPECT_EQ(res.x.size(), nx);
    // t should have steps+1 entries (t=0 through t=steps)
    ASSERT_GE(res.t.size(), 2u);
    // u should be 2-D with consistent inner dimension
    ASSERT_FALSE(res.u.empty());
    for (const auto& row : res.u) {
        EXPECT_EQ(row.size(), nx);
    }
    EXPECT_EQ(res.u.size(), res.t.size());
}

// 9. PDE finite values
TEST(PdeAdvanced, AllValuesFinite) {
    const size_t nx = 21;
    double dx = 1.0 / (nx - 1);
    double alpha = 0.01;
    double dt = 0.4 * dx * dx / alpha;
    auto x0 = make_spike(nx);

    auto res = ms::pde_heat_1d(x0, alpha, dx, dt, 200);
    for (const auto& row : res.u) {
        for (double v : row) {
            EXPECT_TRUE(std::isfinite(v)) << "Non-finite value found: " << v;
        }
    }
}

// Extra: PDE x-grid spacing is uniform
TEST(PdeAdvanced, XGridUniform) {
    const size_t nx = 11;
    double dx = 0.1;
    double alpha = 0.01;
    double dt = 0.4 * dx * dx / alpha;
    auto x0 = make_uniform(nx);

    auto res = ms::pde_heat_1d(x0, alpha, dx, dt, 5);
    ASSERT_EQ(res.x.size(), nx);
    for (size_t i = 1; i < res.x.size(); ++i) {
        EXPECT_NEAR(res.x[i] - res.x[i - 1], dx, 1e-9);
    }
}

// Extra: PDE t-grid spacing is uniform
TEST(PdeAdvanced, TGridUniform) {
    const size_t nx = 11;
    double dx = 0.1;
    double alpha = 0.01;
    double dt = 0.4 * dx * dx / alpha;
    auto x0 = make_uniform(nx);

    auto res = ms::pde_heat_1d(x0, alpha, dx, dt, 10);
    ASSERT_GE(res.t.size(), 2u);
    for (size_t i = 1; i < res.t.size(); ++i) {
        EXPECT_NEAR(res.t[i] - res.t[i - 1], dt, 1e-9);
    }
}

// Extra: PDE symmetry – symmetric initial condition stays symmetric
TEST(PdeAdvanced, SymmetryPreserved) {
    const size_t nx = 21;  // odd so there is a true centre
    double dx = 1.0 / (nx - 1);
    double alpha = 0.01;
    double dt = 0.4 * dx * dx / alpha;
    auto x0 = make_spike(nx);  // spike at exact centre

    auto res = ms::pde_heat_1d(x0, alpha, dx, dt, 50);
    for (const auto& row : res.u) {
        for (size_t i = 0; i < nx / 2; ++i) {
            EXPECT_NEAR(row[i], row[nx - 1 - i], 1e-10)
                << "Asymmetry detected at index " << i;
        }
    }
}

// Extra: ODE – minimum result size (must have at least 2 points: t0 and t_end)
TEST(OdeAdvanced, MinimumResultSize) {
    ms::OdeFunc f = [](double /*t*/, double y) { return -y; };
    auto euler = ms::ode_euler(f, 0.0, 1.0, 1.0, 1);
    auto rk4   = ms::ode_rk4(f,   0.0, 1.0, 1.0, 1);
    EXPECT_GE(euler.t.size(), 2u);
    EXPECT_GE(rk4.t.size(),   2u);
}

// Extra: ODE – non-zero start time
TEST(OdeAdvanced, NonZeroStartTime) {
    ms::OdeFunc f = [](double /*t*/, double y) { return -y; };
    double t0 = 2.0, y0 = std::exp(-2.0), t_end = 4.0;
    auto res = ms::ode_rk4(f, t0, y0, t_end, 500);
    ASSERT_FALSE(res.t.empty());
    EXPECT_DOUBLE_EQ(res.t.front(), t0);
    EXPECT_NEAR(res.y.back(), std::exp(-t_end), 0.001);
}

