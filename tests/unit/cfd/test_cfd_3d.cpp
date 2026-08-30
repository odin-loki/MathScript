#include "ms/cfd/cfd.hpp"

#include <gtest/gtest.h>

using namespace ms::cfd;

namespace {

double field_mass(
    const std::vector<std::vector<std::vector<double>>>& u,
    double dx,
    double dy,
    double dz) {
    return integrated_mass_3d(u, dx, dy, dz);
}

} // namespace

TEST(CfdGrid3D, UniformSpacing) {
    const auto grid = grid3d(0.0, 2.0, 0.0, 1.0, 0.0, 0.5, 20, 10, 8);
    ASSERT_EQ(grid.nx, 20u);
    ASSERT_EQ(grid.ny, 10u);
    ASSERT_EQ(grid.nz, 8u);
    EXPECT_NEAR(grid.dx, 0.1, 1e-12);
    EXPECT_NEAR(grid.dy, 0.1, 1e-12);
    EXPECT_NEAR(grid.dz, 0.0625, 1e-12);
    EXPECT_NEAR(grid.x.front(), 0.05, 1e-12);
    EXPECT_NEAR(grid.y.back(), 0.95, 1e-12);
    EXPECT_NEAR(grid.z.back(), 0.46875, 1e-12);
}

TEST(CfdInitialCondition3D, SquarePulseCentered) {
    const auto grid = grid3d(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 50, 40, 30);
    const auto u0 = square_pulse_3d(grid, 0.5, 0.5, 0.5, 0.2, 0.2, 0.2, 2.0);
    ASSERT_EQ(u0.size(), grid.nz);
    ASSERT_EQ(u0.front().size(), grid.ny);
    ASSERT_EQ(u0.front().front().size(), grid.nx);

    int nonzero = 0;
    for (const auto& layer : u0) {
        for (const auto& row : layer) {
            for (double ui : row) {
                if (ui > 0.0) {
                    ++nonzero;
                    EXPECT_NEAR(ui, 2.0, 1e-12);
                }
            }
        }
    }
    EXPECT_GT(nonzero, 0);
    EXPECT_NEAR(field_mass(u0, grid.dx, grid.dy, grid.dz), 0.2 * 0.2 * 0.2 * 2.0, 1e-9);
}

TEST(CfdRunAdvection3D, PeriodicSquarePulseMassConserved) {
    const auto grid = grid3d(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 40, 40, 40);
    const auto u0 = square_pulse_3d(grid, 0.35, 0.35, 0.35, 0.1, 0.1, 0.1, 1.0);
    const auto vx = constant_velocity(grid.nx * grid.ny * grid.nz, 1.0);
    const auto vy = constant_velocity(grid.nx * grid.ny * grid.nz, 0.0);
    const auto vz = constant_velocity(grid.nx * grid.ny * grid.nz, 0.0);
    const double t_end = 0.3;
    const double dt = 0.01;

    const auto result = run_advection_3d(
        u0,
        vx,
        vy,
        vz,
        t_end,
        dt,
        grid.dx,
        grid.dy,
        grid.dz,
        BoundaryCondition::Periodic,
        BoundaryCondition::Periodic,
        BoundaryCondition::Periodic);
    ASSERT_FALSE(result.u.empty());
    EXPECT_NEAR(result.t.back(), t_end, 1e-12);

    const double m0 = field_mass(u0, grid.dx, grid.dy, grid.dz);
    const double m1 = field_mass(result.u.back(), grid.dx, grid.dy, grid.dz);
    EXPECT_NEAR(m0, 0.1 * 0.1 * 0.1, 1e-9);
    EXPECT_NEAR(m1, m0, 1e-9);
}

TEST(CfdUpwindFvm3D, RejectsCflViolation) {
    const std::size_t nx = 8;
    const std::size_t ny = 6;
    const std::size_t nz = 5;
    const std::vector<std::vector<std::vector<double>>> u0(
        nz, std::vector<std::vector<double>>(ny, std::vector<double>(nx, 1.0)));
    const std::vector<double> vx = {2.0};
    const std::vector<double> vy = {0.0};
    const std::vector<double> vz = {0.0};
    const auto u1 = upwind_fvm_advection_3d(
        u0,
        vx,
        vy,
        vz,
        0.1,
        0.1,
        0.1,
        0.1,
        BoundaryCondition::Periodic,
        BoundaryCondition::Periodic,
        BoundaryCondition::Periodic);
    EXPECT_TRUE(u1.empty());
}

TEST(CfdIntegratedMass3D, ScalesWithCellVolume) {
    const std::vector<std::vector<std::vector<double>>> u = {
        {
            {1.0, 1.0},
            {1.0, 1.0},
        },
        {
            {1.0, 1.0},
            {1.0, 1.0},
        },
    };
    EXPECT_NEAR(integrated_mass_3d(u, 0.5, 0.25, 0.2), 0.2, 1e-12);
}

TEST(CfdGrid3D, InvalidDomainOrResolutionReturnsEmpty) {
    const auto too_few = grid3d(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 8, 8, 1);
    EXPECT_EQ(too_few.nz, 0u);
    EXPECT_TRUE(too_few.z.empty());
    const auto inverted = grid3d(0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 4, 4, 4);
    EXPECT_EQ(inverted.nx, 0u);
}

TEST(CfdInitialCondition3D, NonPositiveWidthReturnsZeros) {
    const auto grid = grid3d(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 6, 6, 6);
    const auto u = square_pulse_3d(grid, 0.5, 0.5, 0.5, 0.2, 0.2, 0.0, 2.0);
    ASSERT_EQ(u.size(), grid.nz);
    for (const auto& layer : u) {
        for (const auto& row : layer) {
            for (double ui : row) {
                EXPECT_NEAR(ui, 0.0, 1e-15);
            }
        }
    }
}

TEST(CfdIntegratedMass3D, EmptyOrNonPositiveSpacing) {
    EXPECT_NEAR(integrated_mass_3d({}, 0.1, 0.1, 0.1), 0.0, 1e-15);
    const std::vector<std::vector<std::vector<double>>> u = {{{1.0}}};
    EXPECT_NEAR(integrated_mass_3d(u, 0.1, 0.1, 0.0), 0.0, 1e-15);
}

TEST(CfdRunAdvection3D, InvalidInputReturnsEmpty) {
    const std::vector<std::vector<std::vector<double>>> u0 = {{{1.0, 1.0}, {1.0, 1.0}}};
    const std::vector<double> v = {1.0};
    const auto result = run_advection_3d(u0, v, v, v, 0.0, 0.1, 0.1, 0.1, 0.1);
    EXPECT_TRUE(result.u.empty());
}

TEST(CfdUpwindFvm3D, EmptyVelocity) {
    const std::vector<std::vector<std::vector<double>>> u0 = {
        {{1.0, 0.0}, {0.0, 1.0}},
        {{0.0, 1.0}, {1.0, 0.0}},
    };
    std::vector<double> empty;
    std::vector<double> v = {0.1};
    EXPECT_TRUE(upwind_fvm_advection_3d(u0, empty, v, v, 0.01, 0.2, 0.2, 0.2).empty());
    EXPECT_TRUE(upwind_fvm_advection_3d(u0, v, empty, v, 0.01, 0.2, 0.2, 0.2).empty());
    EXPECT_TRUE(upwind_fvm_advection_3d(u0, v, v, empty, 0.01, 0.2, 0.2, 0.2).empty());
}

TEST(CfdUpwindFvm3D, NonPositiveSpacing) {
    const std::vector<std::vector<std::vector<double>>> u0 = {
        {{1.0, 0.0}, {0.0, 1.0}},
        {{0.0, 1.0}, {1.0, 0.0}},
    };
    std::vector<double> vx = {0.1};
    std::vector<double> vy = {0.0};
    std::vector<double> vz = {0.0};
    EXPECT_TRUE(upwind_fvm_advection_3d(u0, vx, vy, vz, 0.01, 0.0, 0.2, 0.2).empty());
    EXPECT_TRUE(upwind_fvm_advection_3d(u0, vx, vy, vz, 0.0, 0.2, 0.2, 0.2).empty());
}

TEST(CfdUpwindFvm3D, ZeroFluxAccepted) {
    const std::vector<std::vector<std::vector<double>>> u0 = {
        {{0.0, 1.0}, {0.0, 1.0}},
        {{0.0, 1.0}, {0.0, 1.0}},
    };
    std::vector<double> vx = {0.3};
    std::vector<double> vy = {0.0};
    std::vector<double> vz = {0.0};
    const auto u1 = upwind_fvm_advection_3d(
        u0, vx, vy, vz, 0.05, 0.2, 0.2, 0.2,
        BoundaryCondition::ZeroFlux, BoundaryCondition::ZeroFlux, BoundaryCondition::ZeroFlux);
    ASSERT_EQ(u1.size(), u0.size());
    EXPECT_NEAR(integrated_mass_3d(u1, 0.2, 0.2, 0.2), integrated_mass_3d(u0, 0.2, 0.2, 0.2),
                1e-12);
}

TEST(CfdRunAdvection3D, LastStepClampedToHorizon) {
    const auto grid = grid3d(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 6, 6, 6);
    const auto u0 = square_pulse_3d(grid, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 1.0);
    std::vector<double> vx = {0.3};
    std::vector<double> vy = {0.0};
    std::vector<double> vz = {0.0};
    const auto result = run_advection_3d(u0, vx, vy, vz, 0.25, 0.1, grid.dx, grid.dy, grid.dz);
    ASSERT_FALSE(result.u.empty());
    EXPECT_NEAR(result.t.back(), 0.25, 1e-12);
}

TEST(CfdRunAdvection3D, EmptyVelocity) {
    const std::vector<std::vector<std::vector<double>>> u0 = {
        {{1.0, 0.0}, {0.0, 1.0}},
        {{0.0, 1.0}, {1.0, 0.0}},
    };
    std::vector<double> empty;
    std::vector<double> v = {0.1};
    EXPECT_TRUE(run_advection_3d(u0, empty, v, v, 0.1, 0.01, 0.2, 0.2, 0.2).u.empty());
}

TEST(CfdGrid3D, InvertedYReturnsEmpty) {
    const auto grid = grid3d(0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 4, 4, 4);
    EXPECT_EQ(grid.nx, 0u);
    EXPECT_TRUE(grid.y.empty());
}

TEST(CfdInitialCondition3D, EmptyGridPulse) {
    const auto grid = grid3d(1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 4, 4, 4);
    const auto u = square_pulse_3d(grid, 0.5, 0.5, 0.5, 0.2, 0.2, 0.2, 1.0);
    EXPECT_TRUE(u.empty());
}

TEST(CfdIntegratedMass3D, RaggedFieldIsZero) {
    const std::vector<std::vector<std::vector<double>>> ragged = {
        {{1.0, 1.0}, {1.0}},
    };
    EXPECT_NEAR(integrated_mass_3d(ragged, 0.1, 0.1, 0.1), 0.0, 1e-15);
}

TEST(CfdGrid3D, TooFewNyReturnsEmpty) {
    const auto grid = grid3d(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 4, 1, 4);
    EXPECT_EQ(grid.ny, 0u);
    EXPECT_TRUE(grid.y.empty());
}

TEST(CfdInitialCondition3D, NonPositiveWidthXOrYReturnsZeros) {
    const auto grid = grid3d(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 6, 6, 6);
    const auto ux = square_pulse_3d(grid, 0.5, 0.5, 0.5, 0.0, 0.2, 0.2, 2.0);
    const auto uy = square_pulse_3d(grid, 0.5, 0.5, 0.5, 0.2, -0.1, 0.2, 2.0);
    ASSERT_EQ(ux.size(), grid.nz);
    ASSERT_EQ(uy.size(), grid.nz);
    for (const auto& layer : ux) {
        for (const auto& row : layer) {
            for (double ui : row) {
                EXPECT_NEAR(ui, 0.0, 1e-15);
            }
        }
    }
    for (const auto& layer : uy) {
        for (const auto& row : layer) {
            for (double ui : row) {
                EXPECT_NEAR(ui, 0.0, 1e-15);
            }
        }
    }
}

TEST(CfdIntegratedMass3D, EmptyRowsOrNonPositiveDxDy) {
    const std::vector<std::vector<std::vector<double>>> empty_rows = {{{}, {}}};
    EXPECT_NEAR(integrated_mass_3d(empty_rows, 0.1, 0.1, 0.1), 0.0, 1e-15);
    const std::vector<std::vector<std::vector<double>>> u = {{{1.0}}};
    EXPECT_NEAR(integrated_mass_3d(u, 0.0, 0.1, 0.1), 0.0, 1e-15);
    EXPECT_NEAR(integrated_mass_3d(u, 0.1, -0.2, 0.1), 0.0, 1e-15);
}

TEST(CfdIntegratedMass3D, RaggedLayersIsZero) {
    const std::vector<std::vector<std::vector<double>>> ragged_layers = {
        {{1.0, 1.0}, {1.0, 1.0}},
        {{1.0}},
    };
    EXPECT_NEAR(integrated_mass_3d(ragged_layers, 0.1, 0.1, 0.1), 0.0, 1e-15);
}

TEST(CfdUpwindFvm3D, EmptyOrRaggedField) {
    std::vector<double> vx = {0.1};
    std::vector<double> vy = {0.0};
    std::vector<double> vz = {0.0};
    const std::vector<std::vector<std::vector<double>>> empty;
    EXPECT_TRUE(upwind_fvm_advection_3d(empty, vx, vy, vz, 0.01, 0.2, 0.2, 0.2).empty());
    const std::vector<std::vector<std::vector<double>>> ragged = {
        {{1.0, 0.0}, {0.0}},
        {{0.0, 1.0}, {1.0, 0.0}},
    };
    EXPECT_TRUE(upwind_fvm_advection_3d(ragged, vx, vy, vz, 0.01, 0.2, 0.2, 0.2).empty());
}

TEST(CfdUpwindFvm3D, NonPositiveDyOrDz) {
    const std::vector<std::vector<std::vector<double>>> u0 = {
        {{1.0, 0.0}, {0.0, 1.0}},
        {{0.0, 1.0}, {1.0, 0.0}},
    };
    std::vector<double> vx = {0.1};
    std::vector<double> vy = {0.0};
    std::vector<double> vz = {0.0};
    EXPECT_TRUE(upwind_fvm_advection_3d(u0, vx, vy, vz, 0.01, 0.2, 0.0, 0.2).empty());
    EXPECT_TRUE(upwind_fvm_advection_3d(u0, vx, vy, vz, 0.01, 0.2, 0.2, 0.0).empty());
    EXPECT_TRUE(upwind_fvm_advection_3d(u0, vx, vy, vz, -0.01, 0.2, 0.2, 0.2).empty());
}

TEST(CfdUpwindFvm3D, VelocityLengthMismatch) {
    const std::vector<std::vector<std::vector<double>>> u0 = {
        {{1.0, 0.0}, {0.0, 1.0}},
        {{0.0, 1.0}, {1.0, 0.0}},
    };
    std::vector<double> v = {0.1};
    std::vector<double> bad = {0.1, 0.2};
    EXPECT_TRUE(upwind_fvm_advection_3d(u0, bad, v, v, 0.01, 0.2, 0.2, 0.2).empty());
    EXPECT_TRUE(upwind_fvm_advection_3d(u0, v, bad, v, 0.01, 0.2, 0.2, 0.2).empty());
    EXPECT_TRUE(upwind_fvm_advection_3d(u0, v, v, bad, 0.01, 0.2, 0.2, 0.2).empty());
}

TEST(CfdUpwindFvm3D, ZeroVelocityIdentity) {
    const std::vector<std::vector<std::vector<double>>> u0 = {
        {{1.0, 2.0}, {3.0, 4.0}},
        {{5.0, 6.0}, {7.0, 8.0}},
    };
    std::vector<double> vx = {0.0};
    std::vector<double> vy = {0.0};
    std::vector<double> vz = {0.0};
    const auto u1 = upwind_fvm_advection_3d(u0, vx, vy, vz, 0.01, 0.2, 0.2, 0.2);
    ASSERT_EQ(u1.size(), u0.size());
    EXPECT_NEAR(u1[0][0][0], 1.0, 1e-12);
    EXPECT_NEAR(u1[1][1][1], 8.0, 1e-12);
}

TEST(CfdUpwindFvm3D, VariableVelocityAndNegativeZ) {
    const std::vector<std::vector<std::vector<double>>> u0 = {
        {{0.0, 1.0}, {0.0, 1.0}},
        {{0.0, 1.0}, {0.0, 1.0}},
    };
    std::vector<double> vx(8, 0.0);
    std::vector<double> vy(8, 0.0);
    std::vector<double> vz(8, -0.2);
    const auto u1 = upwind_fvm_advection_3d(u0, vx, vy, vz, 0.05, 0.2, 0.2, 0.2);
    ASSERT_EQ(u1.size(), u0.size());
    EXPECT_NEAR(integrated_mass_3d(u1, 0.2, 0.2, 0.2), integrated_mass_3d(u0, 0.2, 0.2, 0.2),
                1e-12);
}

TEST(CfdUpwindFvm3D, MixedBoundaryConditions) {
    const std::vector<std::vector<std::vector<double>>> u0 = {
        {{0.0, 1.0}, {0.0, 1.0}},
        {{0.0, 1.0}, {0.0, 1.0}},
    };
    std::vector<double> vx = {0.2};
    std::vector<double> vy = {0.1};
    std::vector<double> vz = {0.1};
    const auto u1 = upwind_fvm_advection_3d(
        u0, vx, vy, vz, 0.05, 0.2, 0.2, 0.2,
        BoundaryCondition::Periodic, BoundaryCondition::ZeroFlux, BoundaryCondition::ZeroFlux);
    ASSERT_EQ(u1.size(), u0.size());
    EXPECT_NEAR(integrated_mass_3d(u1, 0.2, 0.2, 0.2), integrated_mass_3d(u0, 0.2, 0.2, 0.2),
                1e-12);
}

TEST(CfdRunAdvection3D, EmptyOrRaggedField) {
    std::vector<double> v = {0.1};
    const std::vector<std::vector<std::vector<double>>> empty;
    EXPECT_TRUE(run_advection_3d(empty, v, v, v, 0.1, 0.01, 0.2, 0.2, 0.2).u.empty());
    const std::vector<std::vector<std::vector<double>>> ragged = {
        {{1.0, 0.0}, {0.0}},
        {{0.0, 1.0}, {1.0, 0.0}},
    };
    EXPECT_TRUE(run_advection_3d(ragged, v, v, v, 0.1, 0.01, 0.2, 0.2, 0.2).u.empty());
}

TEST(CfdRunAdvection3D, NonPositiveSpacing) {
    const std::vector<std::vector<std::vector<double>>> u0 = {
        {{1.0, 0.0}, {0.0, 1.0}},
        {{0.0, 1.0}, {1.0, 0.0}},
    };
    std::vector<double> v = {0.1};
    EXPECT_TRUE(run_advection_3d(u0, v, v, v, 0.1, 0.01, 0.0, 0.2, 0.2).u.empty());
    EXPECT_TRUE(run_advection_3d(u0, v, v, v, 0.1, 0.01, 0.2, 0.0, 0.2).u.empty());
    EXPECT_TRUE(run_advection_3d(u0, v, v, v, 0.1, 0.01, 0.2, 0.2, 0.0).u.empty());
    EXPECT_TRUE(run_advection_3d(u0, v, v, v, 0.1, -0.01, 0.2, 0.2, 0.2).u.empty());
}

TEST(CfdRunAdvection3D, VelocityLengthMismatch) {
    const std::vector<std::vector<std::vector<double>>> u0 = {
        {{1.0, 0.0}, {0.0, 1.0}},
        {{0.0, 1.0}, {1.0, 0.0}},
    };
    std::vector<double> v = {0.1};
    std::vector<double> bad = {0.1, 0.2};
    EXPECT_TRUE(run_advection_3d(u0, bad, v, v, 0.1, 0.01, 0.2, 0.2, 0.2).u.empty());
    EXPECT_TRUE(run_advection_3d(u0, v, bad, v, 0.1, 0.01, 0.2, 0.2, 0.2).u.empty());
    EXPECT_TRUE(run_advection_3d(u0, v, v, bad, 0.1, 0.01, 0.2, 0.2, 0.2).u.empty());
}

TEST(CfdRunAdvection3D, EmptyVyOrVz) {
    const std::vector<std::vector<std::vector<double>>> u0 = {
        {{1.0, 0.0}, {0.0, 1.0}},
        {{0.0, 1.0}, {1.0, 0.0}},
    };
    std::vector<double> empty;
    std::vector<double> v = {0.1};
    EXPECT_TRUE(run_advection_3d(u0, v, empty, v, 0.1, 0.01, 0.2, 0.2, 0.2).u.empty());
    EXPECT_TRUE(run_advection_3d(u0, v, v, empty, 0.1, 0.01, 0.2, 0.2, 0.2).u.empty());
}

TEST(CfdRunAdvection3D, CflViolation) {
    const std::vector<std::vector<std::vector<double>>> u0 = {
        {{1.0, 0.0}, {0.0, 1.0}},
        {{0.0, 1.0}, {1.0, 0.0}},
    };
    std::vector<double> vx = {2.0};
    std::vector<double> vy = {0.0};
    std::vector<double> vz = {0.0};
    EXPECT_TRUE(run_advection_3d(u0, vx, vy, vz, 0.2, 0.1, 0.1, 0.1, 0.1).u.empty());
}
