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
