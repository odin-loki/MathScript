#include "ms/cfd/cfd.hpp"

#include <gtest/gtest.h>

using namespace ms::cfd;

namespace {

double field_mass(const std::vector<std::vector<double>>& u, double dx, double dy) {
    return integrated_mass_2d(u, dx, dy);
}

} // namespace

TEST(CfdGrid2D, UniformSpacing) {
    const auto grid = grid2d(0.0, 2.0, 0.0, 1.0, 20, 10);
    ASSERT_EQ(grid.nx, 20u);
    ASSERT_EQ(grid.ny, 10u);
    EXPECT_NEAR(grid.dx, 0.1, 1e-12);
    EXPECT_NEAR(grid.dy, 0.1, 1e-12);
    EXPECT_NEAR(grid.x.front(), 0.05, 1e-12);
    EXPECT_NEAR(grid.y.back(), 0.95, 1e-12);
}

TEST(CfdInitialCondition2D, SquarePulseCentered) {
    const auto grid = grid2d(0.0, 1.0, 0.0, 1.0, 50, 40);
    const auto u0 = square_pulse_2d(grid, 0.5, 0.5, 0.2, 0.2, 2.0);
    ASSERT_EQ(u0.size(), grid.ny);
    ASSERT_EQ(u0.front().size(), grid.nx);

    int nonzero = 0;
    for (const auto& row : u0) {
        for (double ui : row) {
            if (ui > 0.0) {
                ++nonzero;
                EXPECT_NEAR(ui, 2.0, 1e-12);
            }
        }
    }
    EXPECT_GT(nonzero, 0);
    EXPECT_NEAR(field_mass(u0, grid.dx, grid.dy), 0.2 * 0.2 * 2.0, 1e-9);
}

TEST(CfdRunAdvection2D, PeriodicSquarePulseMassConserved) {
    const auto grid = grid2d(0.0, 1.0, 0.0, 1.0, 80, 60);
    const auto u0 = square_pulse_2d(grid, 0.35, 0.35, 0.1, 0.1, 1.0);
    const auto vx = constant_velocity(grid.nx * grid.ny, 1.0);
    const auto vy = constant_velocity(grid.nx * grid.ny, 0.0);
    const double t_end = 0.4;
    const double dt = 0.01;

    const auto result = run_advection_2d(
        u0,
        vx,
        vy,
        t_end,
        dt,
        grid.dx,
        grid.dy,
        BoundaryCondition::Periodic,
        BoundaryCondition::Periodic);
    ASSERT_FALSE(result.u.empty());
    EXPECT_NEAR(result.t.back(), t_end, 1e-12);

    const double m0 = field_mass(u0, grid.dx, grid.dy);
    const double m1 = field_mass(result.u.back(), grid.dx, grid.dy);
    EXPECT_NEAR(m0, 0.1 * 0.1, 1e-9);
    EXPECT_NEAR(m1, m0, 1e-9);
}

TEST(CfdUpwindFvm2D, RejectsCflViolation) {
    const std::size_t nx = 8;
    const std::size_t ny = 6;
    const std::vector<std::vector<double>> u0(ny, std::vector<double>(nx, 1.0));
    const std::vector<double> vx = {2.0};
    const std::vector<double> vy = {0.0};
    const auto u1 = upwind_fvm_advection_2d(
        u0, vx, vy, 0.1, 0.1, 0.1, BoundaryCondition::Periodic, BoundaryCondition::Periodic);
    EXPECT_TRUE(u1.empty());
}

TEST(CfdIntegratedMass2D, ScalesWithCellArea) {
    const std::vector<std::vector<double>> u = {
        {1.0, 1.0},
        {1.0, 1.0},
    };
    EXPECT_NEAR(integrated_mass_2d(u, 0.5, 0.25), 0.5, 1e-12);
}

TEST(CfdGrid2D, InvalidDomainOrResolutionReturnsEmpty) {
    const auto too_few = grid2d(0.0, 1.0, 0.0, 1.0, 1, 10);
    EXPECT_EQ(too_few.nx, 0u);
    EXPECT_TRUE(too_few.x.empty());
    const auto inverted = grid2d(1.0, 0.0, 0.0, 1.0, 8, 8);
    EXPECT_EQ(inverted.ny, 0u);
}

TEST(CfdInitialCondition2D, NonPositiveWidthReturnsZeros) {
    const auto grid = grid2d(0.0, 1.0, 0.0, 1.0, 8, 8);
    const auto u = square_pulse_2d(grid, 0.5, 0.5, 0.0, 0.2, 2.0);
    ASSERT_EQ(u.size(), grid.ny);
    for (const auto& row : u) {
        for (double ui : row) {
            EXPECT_NEAR(ui, 0.0, 1e-15);
        }
    }
}

TEST(CfdIntegratedMass2D, EmptyOrNonPositiveSpacing) {
    EXPECT_NEAR(integrated_mass_2d({}, 0.1, 0.1), 0.0, 1e-15);
    const std::vector<std::vector<double>> u = {{1.0, 1.0}, {1.0, 1.0}};
    EXPECT_NEAR(integrated_mass_2d(u, 0.0, 0.1), 0.0, 1e-15);
}

TEST(CfdRunAdvection2D, InvalidInputReturnsEmpty) {
    const std::vector<std::vector<double>> u0 = {{1.0, 1.0}, {1.0, 1.0}};
    const std::vector<double> v = {1.0};
    const auto result = run_advection_2d(u0, v, v, 0.0, 0.1, 0.1, 0.1);
    EXPECT_TRUE(result.u.empty());
}

TEST(CfdUpwindFvm2D, ZeroFluxAccepted) {
    const std::vector<std::vector<double>> u0 = {
        {0.0, 1.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 1.0, 0.0},
    };
    const std::vector<double> vx = {0.4};
    const std::vector<double> vy = {0.0};
    const auto u1 = upwind_fvm_advection_2d(
        u0, vx, vy, 0.05, 0.2, 0.2, BoundaryCondition::ZeroFlux, BoundaryCondition::ZeroFlux);
    ASSERT_EQ(u1.size(), u0.size());
    EXPECT_NEAR(integrated_mass_2d(u1, 0.2, 0.2), integrated_mass_2d(u0, 0.2, 0.2), 1e-12);
}

TEST(CfdUpwindFvm2D, EmptyVelocity) {
    const std::vector<std::vector<double>> u0 = {{1.0, 0.0}, {0.0, 1.0}};
    std::vector<double> empty;
    std::vector<double> v = {0.1};
    EXPECT_TRUE(upwind_fvm_advection_2d(u0, empty, v, 0.01, 0.1, 0.1).empty());
    EXPECT_TRUE(upwind_fvm_advection_2d(u0, v, empty, 0.01, 0.1, 0.1).empty());
}

TEST(CfdUpwindFvm2D, NonPositiveSpacing) {
    const std::vector<std::vector<double>> u0 = {{1.0, 0.0}, {0.0, 1.0}};
    std::vector<double> vx = {0.1};
    std::vector<double> vy = {0.0};
    EXPECT_TRUE(upwind_fvm_advection_2d(u0, vx, vy, 0.01, 0.0, 0.1).empty());
    EXPECT_TRUE(upwind_fvm_advection_2d(u0, vx, vy, 0.0, 0.1, 0.1).empty());
}

TEST(CfdUpwindFvm2D, ZeroVelocityIdentity) {
    const std::vector<std::vector<double>> u0 = {{1.0, 2.0}, {3.0, 4.0}};
    std::vector<double> vx = {0.0};
    std::vector<double> vy = {0.0};
    const auto u1 = upwind_fvm_advection_2d(u0, vx, vy, 0.01, 0.1, 0.1);
    ASSERT_EQ(u1.size(), u0.size());
    EXPECT_NEAR(u1[0][0], 1.0, 1e-12);
    EXPECT_NEAR(u1[1][1], 4.0, 1e-12);
}

TEST(CfdRunAdvection2D, LastStepClampedToHorizon) {
    const auto grid = grid2d(0.0, 1.0, 0.0, 1.0, 8, 8);
    const auto u0 = square_pulse_2d(grid, 0.3, 0.3, 0.2, 0.2, 1.0);
    std::vector<double> vx = {0.3};
    std::vector<double> vy = {0.0};
    const auto result = run_advection_2d(u0, vx, vy, 0.25, 0.1, grid.dx, grid.dy);
    ASSERT_FALSE(result.u.empty());
    EXPECT_NEAR(result.t.back(), 0.25, 1e-12);
}

TEST(CfdRunAdvection2D, EmptyVelocity) {
    const std::vector<std::vector<double>> u0 = {{1.0, 0.0}, {0.0, 1.0}};
    std::vector<double> empty;
    std::vector<double> v = {0.1};
    EXPECT_TRUE(run_advection_2d(u0, empty, v, 0.1, 0.01, 0.1, 0.1).u.empty());
}

TEST(CfdGrid2D, InvertedYReturnsEmpty) {
    const auto grid = grid2d(0.0, 1.0, 1.0, 0.0, 8, 8);
    EXPECT_EQ(grid.nx, 0u);
    EXPECT_TRUE(grid.y.empty());
}

TEST(CfdInitialCondition2D, EmptyGridPulse) {
    const auto grid = grid2d(1.0, 0.0, 0.0, 1.0, 8, 8);
    const auto u = square_pulse_2d(grid, 0.5, 0.5, 0.2, 0.2, 1.0);
    EXPECT_TRUE(u.empty());
}

TEST(CfdIntegratedMass2D, RaggedFieldIsZero) {
    const std::vector<std::vector<double>> ragged = {{1.0, 2.0}, {3.0}};
    EXPECT_NEAR(integrated_mass_2d(ragged, 0.1, 0.1), 0.0, 1e-15);
}

TEST(CfdGrid2D, TooFewNyReturnsEmpty) {
    const auto grid = grid2d(0.0, 1.0, 0.0, 1.0, 8, 1);
    EXPECT_EQ(grid.ny, 0u);
    EXPECT_TRUE(grid.y.empty());
}

TEST(CfdInitialCondition2D, NonPositiveWidthYReturnsZeros) {
    const auto grid = grid2d(0.0, 1.0, 0.0, 1.0, 8, 8);
    const auto u = square_pulse_2d(grid, 0.5, 0.5, 0.2, -0.1, 2.0);
    ASSERT_EQ(u.size(), grid.ny);
    for (const auto& row : u) {
        for (double ui : row) {
            EXPECT_NEAR(ui, 0.0, 1e-15);
        }
    }
}

TEST(CfdInitialCondition2D, DefaultAmplitudeIsOne) {
    const auto grid = grid2d(0.0, 1.0, 0.0, 1.0, 10, 10);
    const auto u = square_pulse_2d(grid, 0.5, 0.5, 0.2, 0.2);
    int nonzero = 0;
    for (const auto& row : u) {
        for (double ui : row) {
            if (ui > 0.0) {
                ++nonzero;
                EXPECT_NEAR(ui, 1.0, 1e-12);
            }
        }
    }
    EXPECT_GT(nonzero, 0);
}

TEST(CfdIntegratedMass2D, EmptyRowsOrNonPositiveDy) {
    const std::vector<std::vector<double>> empty_rows(2);
    EXPECT_NEAR(integrated_mass_2d(empty_rows, 0.1, 0.1), 0.0, 1e-15);
    const std::vector<std::vector<double>> u = {{1.0, 1.0}, {1.0, 1.0}};
    EXPECT_NEAR(integrated_mass_2d(u, 0.1, 0.0), 0.0, 1e-15);
    EXPECT_NEAR(integrated_mass_2d(u, 0.1, -0.2), 0.0, 1e-15);
}

TEST(CfdUpwindFvm2D, EmptyOrRaggedField) {
    std::vector<double> vx = {0.1};
    std::vector<double> vy = {0.0};
    const std::vector<std::vector<double>> empty;
    EXPECT_TRUE(upwind_fvm_advection_2d(empty, vx, vy, 0.01, 0.1, 0.1).empty());
    const std::vector<std::vector<double>> empty_rows(2);
    EXPECT_TRUE(upwind_fvm_advection_2d(empty_rows, vx, vy, 0.01, 0.1, 0.1).empty());
    const std::vector<std::vector<double>> ragged = {{1.0, 2.0}, {3.0}};
    EXPECT_TRUE(upwind_fvm_advection_2d(ragged, vx, vy, 0.01, 0.1, 0.1).empty());
}

TEST(CfdUpwindFvm2D, NonPositiveDy) {
    const std::vector<std::vector<double>> u0 = {{1.0, 0.0}, {0.0, 1.0}};
    std::vector<double> vx = {0.1};
    std::vector<double> vy = {0.0};
    EXPECT_TRUE(upwind_fvm_advection_2d(u0, vx, vy, 0.01, 0.1, 0.0).empty());
    EXPECT_TRUE(upwind_fvm_advection_2d(u0, vx, vy, 0.01, 0.1, -0.1).empty());
    EXPECT_TRUE(upwind_fvm_advection_2d(u0, vx, vy, -0.01, 0.1, 0.1).empty());
}

TEST(CfdUpwindFvm2D, VelocityLengthMismatch) {
    const std::vector<std::vector<double>> u0 = {{1.0, 0.0}, {0.0, 1.0}};
    std::vector<double> bad_vx = {0.1, 0.2};
    std::vector<double> vy = {0.0};
    std::vector<double> vx = {0.1};
    std::vector<double> bad_vy = {0.0, 0.1, 0.2};
    EXPECT_TRUE(upwind_fvm_advection_2d(u0, bad_vx, vy, 0.01, 0.1, 0.1).empty());
    EXPECT_TRUE(upwind_fvm_advection_2d(u0, vx, bad_vy, 0.01, 0.1, 0.1).empty());
}

TEST(CfdUpwindFvm2D, VariableVelocityAndNegativeY) {
    const std::vector<std::vector<double>> u0 = {
        {0.0, 1.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 1.0, 0.0},
    };
    std::vector<double> vx(9, 0.0);
    std::vector<double> vy(9, -0.2);
    const auto u1 = upwind_fvm_advection_2d(u0, vx, vy, 0.05, 0.2, 0.2);
    ASSERT_EQ(u1.size(), u0.size());
    EXPECT_NEAR(integrated_mass_2d(u1, 0.2, 0.2), integrated_mass_2d(u0, 0.2, 0.2), 1e-12);
}

TEST(CfdUpwindFvm2D, MixedBoundaryConditions) {
    const std::vector<std::vector<double>> u0 = {
        {0.0, 1.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 1.0, 0.0},
    };
    std::vector<double> vx = {0.3};
    std::vector<double> vy = {0.2};
    const auto u1 = upwind_fvm_advection_2d(
        u0, vx, vy, 0.05, 0.2, 0.2, BoundaryCondition::Periodic, BoundaryCondition::ZeroFlux);
    ASSERT_EQ(u1.size(), u0.size());
    EXPECT_NEAR(integrated_mass_2d(u1, 0.2, 0.2), integrated_mass_2d(u0, 0.2, 0.2), 1e-12);
}

TEST(CfdRunAdvection2D, EmptyOrRaggedField) {
    std::vector<double> vx = {0.1};
    std::vector<double> vy = {0.0};
    const std::vector<std::vector<double>> empty;
    EXPECT_TRUE(run_advection_2d(empty, vx, vy, 0.1, 0.01, 0.1, 0.1).u.empty());
    const std::vector<std::vector<double>> ragged = {{1.0, 2.0}, {3.0}};
    EXPECT_TRUE(run_advection_2d(ragged, vx, vy, 0.1, 0.01, 0.1, 0.1).u.empty());
}

TEST(CfdRunAdvection2D, NonPositiveSpacing) {
    const std::vector<std::vector<double>> u0 = {{1.0, 0.0}, {0.0, 1.0}};
    std::vector<double> vx = {0.1};
    std::vector<double> vy = {0.0};
    EXPECT_TRUE(run_advection_2d(u0, vx, vy, 0.1, 0.01, 0.0, 0.1).u.empty());
    EXPECT_TRUE(run_advection_2d(u0, vx, vy, 0.1, 0.01, 0.1, 0.0).u.empty());
    EXPECT_TRUE(run_advection_2d(u0, vx, vy, 0.1, -0.01, 0.1, 0.1).u.empty());
}

TEST(CfdRunAdvection2D, VelocityLengthMismatch) {
    const std::vector<std::vector<double>> u0 = {{1.0, 0.0}, {0.0, 1.0}};
    std::vector<double> bad = {0.1, 0.2};
    std::vector<double> v = {0.1};
    EXPECT_TRUE(run_advection_2d(u0, bad, v, 0.1, 0.01, 0.1, 0.1).u.empty());
    EXPECT_TRUE(run_advection_2d(u0, v, bad, 0.1, 0.01, 0.1, 0.1).u.empty());
}

TEST(CfdRunAdvection2D, EmptyVy) {
    const std::vector<std::vector<double>> u0 = {{1.0, 0.0}, {0.0, 1.0}};
    std::vector<double> empty;
    std::vector<double> v = {0.1};
    EXPECT_TRUE(run_advection_2d(u0, v, empty, 0.1, 0.01, 0.1, 0.1).u.empty());
}

TEST(CfdRunAdvection2D, CflViolation) {
    const std::vector<std::vector<double>> u0 = {{1.0, 0.0}, {0.0, 1.0}};
    std::vector<double> vx = {2.0};
    std::vector<double> vy = {0.0};
    EXPECT_TRUE(run_advection_2d(u0, vx, vy, 0.2, 0.1, 0.1, 0.1).u.empty());
}
