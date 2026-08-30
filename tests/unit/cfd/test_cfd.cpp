#include "ms/cfd/cfd.hpp"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <numeric>

using namespace ms::cfd;

namespace {

double pulse_mass(const std::vector<double>& u, double dx) {
    return integrated_mass(u, dx);
}

std::size_t peak_index(const std::vector<double>& u) {
    return static_cast<std::size_t>(
        std::distance(u.begin(), std::max_element(u.begin(), u.end())));
}

} // namespace

TEST(CfdGrid1D, UniformSpacing) {
    const auto grid = grid1d(0.0, 1.0, 10);
    ASSERT_EQ(grid.n, 10u);
    EXPECT_NEAR(grid.dx, 0.1, 1e-12);
    EXPECT_NEAR(grid.x.front(), 0.05, 1e-12);
    EXPECT_NEAR(grid.x.back(), 0.95, 1e-12);
}

TEST(CfdGrid1D, InvalidDomainReturnsEmpty) {
    const auto grid = grid1d(1.0, 1.0, 10);
    EXPECT_EQ(grid.n, 0u);
    EXPECT_TRUE(grid.x.empty());
}

TEST(CfdInitialCondition, SquarePulseCentered) {
    const auto grid = grid1d(0.0, 10.0, 100);
    const auto u0 = square_pulse(grid, 5.0, 2.0, 3.0);
    ASSERT_EQ(u0.size(), grid.n);

    int nonzero = 0;
    double mass = 0.0;
    for (std::size_t i = 0; i < grid.n; ++i) {
        if (u0[i] > 0.0) {
            ++nonzero;
            EXPECT_NEAR(u0[i], 3.0, 1e-12);
            EXPECT_LE(std::abs(grid.x[i] - 5.0), 1.0 + 0.5 * grid.dx);
            mass += u0[i];
        } else {
            EXPECT_NEAR(u0[i], 0.0, 1e-12);
        }
    }
    EXPECT_GT(nonzero, 0);
    EXPECT_NEAR(mass * grid.dx, 2.0 * 3.0, 1e-9);
}

TEST(CfdVelocityField, ConstantVelocityLength) {
    const auto v = constant_velocity(8, 1.5);
    ASSERT_EQ(v.size(), 8u);
    for (double vi : v) {
        EXPECT_NEAR(vi, 1.5, 1e-12);
    }
}

TEST(CfdUpwindFvm, SingleStepPositiveVelocity) {
    const std::vector<double> u0 = {0.0, 0.0, 1.0, 0.0, 0.0};
    const std::vector<double> v = {1.0};
    const double dx = 0.1;
    const double dt = 0.05;
    const auto u1 = upwind_fvm_advection(u0, v, dt, dx, BoundaryCondition::Periodic);
    ASSERT_EQ(u1.size(), u0.size());
    EXPECT_NEAR(u1[2], 0.5, 1e-12);
    EXPECT_NEAR(u1[3], 0.5, 1e-12);
    EXPECT_NEAR(integrated_mass(u1, dx), integrated_mass(u0, dx), 1e-12);
}

TEST(CfdUpwindFvm, RejectsCflViolation) {
    const std::vector<double> u0(10, 1.0);
    const std::vector<double> v = {2.0};
    const auto u1 = upwind_fvm_advection(u0, v, 0.1, 0.1, BoundaryCondition::Periodic);
    EXPECT_TRUE(u1.empty());
}

TEST(CfdUpwindFvm, ZeroVelocityIsIdentity) {
    const std::vector<double> u0 = {0.0, 2.0, 4.0, 1.0};
    const std::vector<double> v = {0.0};
    const auto u1 = upwind_fvm_advection(u0, v, 0.01, 0.1, BoundaryCondition::Periodic);
    ASSERT_EQ(u1.size(), u0.size());
    for (std::size_t i = 0; i < u0.size(); ++i) {
        EXPECT_NEAR(u1[i], u0[i], 1e-12);
    }
}

TEST(CfdRunAdvection, PeriodicSquarePulseMassConserved) {
    const auto grid = grid1d(0.0, 1.0, 100);
    const auto u0 = square_pulse(grid, 0.2, 0.1, 1.0);
    const auto v = constant_velocity(grid.n, 1.0);
    const double t_end = 0.3;
    const double dt = 0.01;

    const auto result = run_advection(u0, v, t_end, dt, grid.dx, BoundaryCondition::Periodic);
    ASSERT_FALSE(result.u.empty());
    EXPECT_NEAR(result.t.back(), t_end, 1e-12);

    const double m0 = pulse_mass(u0, grid.dx);
    const double m1 = pulse_mass(result.u.back(), grid.dx);
    EXPECT_NEAR(m0, 0.1, 1e-9);
    EXPECT_NEAR(m1, m0, 1e-9);
}

TEST(CfdRunAdvection, PeriodicDeltaShiftsRight) {
    const std::size_t n = 20;
    const auto grid = grid1d(0.0, static_cast<double>(n) * 0.1, n);
    std::vector<double> u0(n, 0.0);
    u0[3] = 1.0;
    const auto v = constant_velocity(n, 1.0);
    const double dt = 0.05;
    const double t_end = 0.2;
    const auto result = run_advection(u0, v, t_end, dt, grid.dx, BoundaryCondition::Periodic);
    ASSERT_FALSE(result.u.empty());
    const std::size_t shift =
        static_cast<std::size_t>(std::lround(v.front() * t_end / grid.dx));
    EXPECT_EQ(peak_index(result.u.back()), (3u + shift) % n);
}

TEST(CfdRunAdvection, ZeroFluxSquarePulseMassConserved) {
    const auto grid = grid1d(0.0, 1.0, 80);
    const auto u0 = square_pulse(grid, 0.5, 0.2, 2.0);
    const auto v = constant_velocity(grid.n, 0.8);
    const double dt = 0.005;
    const double t_end = 0.25;

    const auto result = run_advection(u0, v, t_end, dt, grid.dx, BoundaryCondition::ZeroFlux);
    ASSERT_FALSE(result.u.empty());

    const double m0 = pulse_mass(u0, grid.dx);
    const double m1 = pulse_mass(result.u.back(), grid.dx);
    EXPECT_NEAR(m1, m0, 1e-9);
}

TEST(CfdRunAdvection, ZeroFluxLeftBoundaryNoInflow) {
    const auto grid = grid1d(0.0, 1.0, 20);
    auto u0 = square_pulse(grid, grid.x.front(), grid.dx, 1.0);
    const auto v = constant_velocity(grid.n, 1.0);
    const double m0 = pulse_mass(u0, grid.dx);
    const auto u1 = upwind_fvm_advection(u0, v, 0.05, grid.dx, BoundaryCondition::ZeroFlux);
    ASSERT_EQ(u1.size(), u0.size());
    EXPECT_NEAR(pulse_mass(u1, grid.dx), m0, 1e-12);
    EXPECT_LT(u1.front(), u0.front());
}

TEST(CfdRunAdvection, NegativeVelocityPeriodicWrap) {
    const auto grid = grid1d(0.0, 1.0, 20);
    std::vector<double> u0(grid.n, 0.0);
    u0[0] = 1.0;
    const auto v = constant_velocity(grid.n, -1.0);
    const double dt = 0.05;
    const double t_end = 0.1;
    const auto result = run_advection(u0, v, t_end, dt, grid.dx, BoundaryCondition::Periodic);
    ASSERT_FALSE(result.u.empty());
    const std::size_t shift =
        static_cast<std::size_t>(std::lround(std::abs(v.front()) * t_end / grid.dx));
    EXPECT_EQ(peak_index(result.u.back()), (grid.n - shift) % grid.n);
}

TEST(CfdRunAdvection, VariableVelocityFieldAccepted) {
    const auto grid = grid1d(0.0, 1.0, 16);
    const auto u0 = square_pulse(grid, 0.3, 0.2, 1.0);
    auto v = constant_velocity(grid.n, 0.5);
    v[grid.n / 2] = 0.5;
    const auto result = run_advection(u0, v, 0.1, 0.01, grid.dx, BoundaryCondition::Periodic);
    ASSERT_FALSE(result.u.empty());
    EXPECT_NEAR(result.t.back(), 0.1, 1e-12);
}

TEST(CfdRunAdvection, InvalidInputReturnsEmpty) {
    const std::vector<double> u0 = {1.0, 2.0};
    const std::vector<double> v = {1.0, 2.0, 3.0};
    const auto result = run_advection(u0, v, 1.0, 0.1, 0.1, BoundaryCondition::Periodic);
    EXPECT_TRUE(result.u.empty());
}

TEST(CfdIntegratedMass, ScalesWithDx) {
    const std::vector<double> u = {1.0, 2.0, 3.0};
    EXPECT_NEAR(integrated_mass(u, 0.5), 3.0, 1e-12);
    EXPECT_NEAR(integrated_mass(u, 2.0), 12.0, 1e-12);
}

TEST(CfdGrid1D, TooFewNodesReturnsEmpty) {
    const auto grid = grid1d(0.0, 1.0, 1);
    EXPECT_EQ(grid.n, 0u);
    EXPECT_TRUE(grid.x.empty());
}

TEST(CfdInitialCondition, NonPositiveWidthReturnsZeros) {
    const auto grid = grid1d(0.0, 1.0, 10);
    const auto u = square_pulse(grid, 0.5, 0.0, 3.0);
    ASSERT_EQ(u.size(), grid.n);
    for (double ui : u) {
        EXPECT_NEAR(ui, 0.0, 1e-15);
    }
}

TEST(CfdVelocityField, ZeroLength) {
    const auto v = constant_velocity(0, 1.5);
    EXPECT_TRUE(v.empty());
}

TEST(CfdIntegratedMass, EmptyOrNonPositiveDx) {
    const std::vector<double> empty;
    const std::vector<double> two{1.0, 2.0};
    EXPECT_NEAR(integrated_mass(empty, 0.1), 0.0, 1e-15);
    EXPECT_NEAR(integrated_mass(two, 0.0), 0.0, 1e-15);
}

TEST(CfdRunAdvection, NonPositiveHorizonReturnsEmpty) {
    const std::vector<double> u0 = {1.0, 0.0};
    const std::vector<double> v = {1.0};
    const auto result = run_advection(u0, v, 0.0, 0.1, 0.1);
    EXPECT_TRUE(result.u.empty());
}

TEST(CfdRunAdvection, LastStepClampedToHorizon) {
    const auto grid = grid1d(0.0, 1.0, 20);
    const auto u0 = square_pulse(grid, 0.3, 0.2, 1.0);
    const auto v = constant_velocity(grid.n, 0.5);
    const auto result = run_advection(u0, v, 0.25, 0.1, grid.dx);
    ASSERT_FALSE(result.u.empty());
    EXPECT_NEAR(result.t.back(), 0.25, 1e-12);
}

TEST(CfdUpwindFvm, EmptyFieldOrVelocity) {
    std::vector<double> empty;
    std::vector<double> u = {1.0, 0.0};
    std::vector<double> v = {0.5};
    EXPECT_TRUE(upwind_fvm_advection(empty, v, 0.01, 0.1).empty());
    EXPECT_TRUE(upwind_fvm_advection(u, empty, 0.01, 0.1).empty());
}

TEST(CfdUpwindFvm, NonPositiveDtOrDx) {
    std::vector<double> u = {1.0, 0.0, 0.0};
    std::vector<double> v = {0.5};
    EXPECT_TRUE(upwind_fvm_advection(u, v, 0.0, 0.1).empty());
    EXPECT_TRUE(upwind_fvm_advection(u, v, 0.01, 0.0).empty());
}

TEST(CfdUpwindFvm, VelocityLengthMismatch) {
    std::vector<double> u = {1.0, 0.0};
    std::vector<double> v = {0.5, 0.4, 0.3};
    EXPECT_TRUE(upwind_fvm_advection(u, v, 0.01, 0.1).empty());
}

TEST(CfdRunAdvection, EmptyField) {
    std::vector<double> empty;
    std::vector<double> v = {1.0};
    const auto result = run_advection(empty, v, 1.0, 0.1, 0.1);
    EXPECT_TRUE(result.u.empty());
}

TEST(CfdInitialCondition, EmptyGridPulseIsEmpty) {
    const auto grid = grid1d(1.0, 0.0, 10);
    const auto u = square_pulse(grid, 0.5, 0.2, 1.0);
    EXPECT_TRUE(u.empty());
}

TEST(CfdUpwindFvm, ZeroFluxNegativeVelocityRightGhost) {
    const auto grid = grid1d(0.0, 1.0, 8);
    std::vector<double> u0(grid.n, 0.0);
    u0.back() = 1.0;
    const auto v = constant_velocity(grid.n, -1.0);
    const auto u1 = upwind_fvm_advection(u0, v, 0.05, grid.dx, BoundaryCondition::ZeroFlux);
    ASSERT_EQ(u1.size(), u0.size());
    EXPECT_TRUE(std::isfinite(u1.back()));
}

TEST(CfdUpwindFvm, EmptyFieldAndScalarVelocity) {
    std::vector<double> empty;
    const std::vector<double> scalar_v{0.5};
    EXPECT_TRUE(upwind_fvm_advection(empty, scalar_v, 0.01, 0.1).empty());

    const auto grid = grid1d(0.0, 1.0, 6);
    const std::vector<double> u0(grid.n, 1.0);
    const auto stepped = upwind_fvm_advection(u0, scalar_v, 0.01, grid.dx, BoundaryCondition::ZeroFlux);
    ASSERT_EQ(stepped.size(), u0.size());
}

TEST(CfdUpwindFvm, VelocityLengthMismatchVsN) {
    const std::vector<double> u{1.0, 2.0, 3.0, 4.0};
    const std::vector<double> v{0.1, 0.2};
    EXPECT_TRUE(upwind_fvm_advection(u, v, 0.01, 0.1).empty());
}

TEST(CfdRunAdvection, EmptyVelocity) {
    const std::vector<double> u0{1.0, 0.0, 0.0};
    const std::vector<double> empty;
    const auto result = run_advection(u0, empty, 1.0, 0.1, 0.1);
    EXPECT_TRUE(result.u.empty());
}

TEST(CfdRunAdvection, CflViolationAndNonPositiveDx) {
    const std::vector<double> u0(8, 1.0);
    const std::vector<double> v{2.0};
    EXPECT_TRUE(run_advection(u0, v, 1.0, 0.1, 0.1).u.empty());
    EXPECT_TRUE(run_advection(u0, v, 1.0, 0.0, 0.1).u.empty());
    EXPECT_TRUE(run_advection(u0, v, 1.0, 0.1, 0.0).u.empty());
}

TEST(CfdUpwindFvm, CellWiseVelocityLastFace) {
    const std::vector<double> u{0.0, 0.0, 1.0, 0.0};
    const std::vector<double> v{0.5, 0.4, 0.3, 0.2};
    const auto u1 = upwind_fvm_advection(u, v, 0.05, 0.2, BoundaryCondition::Periodic);
    ASSERT_EQ(u1.size(), u.size());
    EXPECT_TRUE(std::isfinite(u1.back()));
    EXPECT_NEAR(integrated_mass(u1, 0.2), integrated_mass(u, 0.2), 1e-12);

    const auto grid = grid1d(0.0, 1.0, 8);
    const auto neg_width = square_pulse(grid, 0.5, -0.2, 1.0);
    ASSERT_EQ(neg_width.size(), grid.n);
    for (double ui : neg_width) {
        EXPECT_NEAR(ui, 0.0, 1e-15);
    }
}

TEST(CfdInitialCondition, DefaultAmplitudeAndOffGridZeros) {
    const auto grid = grid1d(0.0, 1.0, 20);
    const auto u_def = square_pulse(grid, 0.5, 0.2);
    ASSERT_EQ(u_def.size(), grid.n);
    int nonzero = 0;
    for (double ui : u_def) {
        if (ui > 0.0) {
            ++nonzero;
            EXPECT_NEAR(ui, 1.0, 1e-12);
        }
    }
    EXPECT_GT(nonzero, 0);

    const auto u_off = square_pulse(grid, 50.0, 0.2, 4.0);
    ASSERT_EQ(u_off.size(), grid.n);
    for (double ui : u_off) {
        EXPECT_NEAR(ui, 0.0, 1e-15);
    }
}

TEST(CfdRunAdvection, CellWiseVelocityMassConserved) {
    const auto grid = grid1d(0.0, 1.0, 16);
    const auto u0 = square_pulse(grid, 0.4, 0.2, 1.0);
    const std::vector<double> v(grid.n, 0.4);
    const auto result = run_advection(u0, v, 0.1, 0.01, grid.dx, BoundaryCondition::Periodic);
    ASSERT_FALSE(result.u.empty());
    EXPECT_NEAR(pulse_mass(result.u.back(), grid.dx), pulse_mass(u0, grid.dx), 1e-9);
}

TEST(CfdUpwindFvm, CflEqualsOneAccepted) {
    const std::vector<double> u = {0.0, 1.0, 0.0, 0.0};
    const std::vector<double> v = {1.0};
    const auto u1 = upwind_fvm_advection(u, v, 0.1, 0.1, BoundaryCondition::Periodic);
    ASSERT_EQ(u1.size(), u.size());
    EXPECT_NEAR(u1[1], 0.0, 1e-12);
    EXPECT_NEAR(u1[2], 1.0, 1e-12);
    EXPECT_NEAR(integrated_mass(u1, 0.1), integrated_mass(u, 0.1), 1e-12);
}

TEST(CfdUpwindFvm, TwoDZeroFluxCflOneAndMismatch) {
    const std::vector<std::vector<double>> u0 = {
        {0.0, 1.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 1.0, 0.0},
    };
    const std::vector<double> vx{1.0};
    const std::vector<double> vy{0.0};
    const auto u1 = upwind_fvm_advection_2d(
        u0, vx, vy, 0.2, 0.2, 0.2, BoundaryCondition::ZeroFlux, BoundaryCondition::ZeroFlux);
    ASSERT_EQ(u1.size(), u0.size());
    ASSERT_EQ(u1[0].size(), u0[0].size());
    EXPECT_NEAR(integrated_mass_2d(u1, 0.2, 0.2), integrated_mass_2d(u0, 0.2, 0.2), 1e-12);

    const std::vector<double> bad{0.1, 0.2};
    EXPECT_TRUE(upwind_fvm_advection_2d(u0, bad, vy, 0.01, 0.2, 0.2).empty());
    EXPECT_TRUE(upwind_fvm_advection_2d({}, vx, vy, 0.01, 0.2, 0.2).empty());
}

TEST(CfdUpwindFvm, ThreeDZeroFluxNegativeAndCflOne) {
    std::vector<std::vector<std::vector<double>>> u0(
        2, std::vector<std::vector<double>>(2, std::vector<double>(2, 0.0)));
    u0[1][1][1] = 1.0;
    const std::vector<double> vx_neg{-1.0};
    const std::vector<double> v0{0.0};
    const auto u1 = upwind_fvm_advection_3d(
        u0, vx_neg, v0, v0, 0.2, 0.2, 0.2, 0.2,
        BoundaryCondition::ZeroFlux, BoundaryCondition::ZeroFlux, BoundaryCondition::ZeroFlux);
    ASSERT_EQ(u1.size(), 2u);
    EXPECT_NEAR(integrated_mass_3d(u1, 0.2, 0.2, 0.2), integrated_mass_3d(u0, 0.2, 0.2, 0.2), 1e-12);

    const std::vector<double> vx_pos{1.0};
    const auto u2 = upwind_fvm_advection_3d(u0, vx_pos, v0, v0, 0.2, 0.2, 0.2, 0.2);
    ASSERT_EQ(u2.size(), 2u);
    EXPECT_NEAR(integrated_mass_3d(u2, 0.2, 0.2, 0.2), integrated_mass_3d(u0, 0.2, 0.2, 0.2), 1e-12);
}

TEST(CfdUpwindFvm, TwoDThreeDCellWiseLastFace) {
    const std::vector<std::vector<double>> u2 = {
        {1.0, 0.0},
        {0.0, 1.0},
    };
    const std::vector<double> vx2{0.2, 0.2, 0.2, 0.2};
    const std::vector<double> vy2{0.0, 0.0, 0.0, 0.0};
    const auto s2 = upwind_fvm_advection_2d(u2, vx2, vy2, 0.05, 0.2, 0.2);
    ASSERT_EQ(s2.size(), 2u);
    EXPECT_TRUE(std::isfinite(s2[0][0]));
    EXPECT_NEAR(integrated_mass_2d(s2, 0.2, 0.2), integrated_mass_2d(u2, 0.2, 0.2), 1e-12);

    std::vector<std::vector<std::vector<double>>> u3(
        2, std::vector<std::vector<double>>(2, std::vector<double>(2, 0.0)));
    u3[0][0][0] = 1.0;
    const std::vector<double> vx3(8, 0.2);
    const std::vector<double> v03(8, 0.0);
    const auto s3 = upwind_fvm_advection_3d(u3, vx3, v03, v03, 0.05, 0.2, 0.2, 0.2);
    ASSERT_EQ(s3.size(), 2u);
    EXPECT_NEAR(integrated_mass_3d(s3, 0.2, 0.2, 0.2), integrated_mass_3d(u3, 0.2, 0.2, 0.2), 1e-12);
    EXPECT_TRUE(upwind_fvm_advection_3d({}, vx3, v03, v03, 0.01, 0.2, 0.2, 0.2).empty());
}

TEST(CfdUpwindFvm, ScalarVelocityVsLengthMismatch) {
    const std::vector<double> u{0.0, 1.0, 0.0, 0.0};
    const std::vector<double> scalar_v{0.5};
    const auto stepped = upwind_fvm_advection(u, scalar_v, 0.05, 0.2, BoundaryCondition::Periodic);
    ASSERT_EQ(stepped.size(), u.size());
    EXPECT_NEAR(integrated_mass(stepped, 0.2), integrated_mass(u, 0.2), 1e-12);

    const std::vector<double> mismatch{0.1, 0.2, 0.3};
    EXPECT_TRUE(upwind_fvm_advection(u, mismatch, 0.05, 0.2).empty());

    const std::vector<std::vector<double>> u2 = {
        {1.0, 0.0},
        {0.0, 1.0},
    };
    const std::vector<double> vx_s{0.3};
    const std::vector<double> vy_s{0.0};
    const auto s2 = upwind_fvm_advection_2d(
        u2, vx_s, vy_s, 0.05, 0.2, 0.2, BoundaryCondition::Periodic, BoundaryCondition::Periodic);
    ASSERT_EQ(s2.size(), 2u);
    EXPECT_NEAR(integrated_mass_2d(s2, 0.2, 0.2), integrated_mass_2d(u2, 0.2, 0.2), 1e-12);
    EXPECT_TRUE(upwind_fvm_advection_2d(u2, mismatch, vy_s, 0.05, 0.2, 0.2).empty());

    std::vector<std::vector<std::vector<double>>> u3(
        2, std::vector<std::vector<double>>(2, std::vector<double>(2, 0.0)));
    u3[0][0][0] = 1.0;
    const std::vector<double> v0{0.0};
    const auto s3 = upwind_fvm_advection_3d(
        u3, vx_s, v0, v0, 0.05, 0.2, 0.2, 0.2,
        BoundaryCondition::Periodic, BoundaryCondition::Periodic, BoundaryCondition::Periodic);
    ASSERT_EQ(s3.size(), 2u);
    EXPECT_NEAR(integrated_mass_3d(s3, 0.2, 0.2, 0.2), integrated_mass_3d(u3, 0.2, 0.2, 0.2), 1e-12);
    EXPECT_TRUE(upwind_fvm_advection_3d(u3, mismatch, v0, v0, 0.05, 0.2, 0.2, 0.2).empty());
}
