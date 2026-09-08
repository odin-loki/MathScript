// Regression test for the periodic face velocities in the upwind FV advection.
//
// A finite-volume scheme conserves the cell-integrated field exactly when the
// flux sum telescopes, which requires that the two periodic boundary faces --
// face 0 and face n, which are the SAME face, between cell n-1 and cell 0 --
// carry identical flux.
//
// face_velocity did not take the boundary condition. Face 0 clamped its left
// cell to 0 and got 0.5*(v[0]+v[1]), the velocity of face 1; face n fell
// through to the one-sided v[n-1]. The mismatch injected
// u_up * (v[n-1] - 0.5*(v[0]+v[1])) of mass every step: with u = {0,0,0,1},
// v = {0.5,0.4,0.3,0.2}, dx = 0.2, dt = 0.05 the total grew from 0.2000 to
// 0.2566 in five steps, a 28% gain. The same held for the 2D and 3D variants.

#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <vector>

#include "ms/cfd/cfd.hpp"

namespace {

double total_mass(const std::vector<double>& f, double dx) {
    return std::accumulate(f.begin(), f.end(), 0.0) * dx;
}

} // namespace

TEST(CfdPeriodicConservation, OneDimensionalNonUniformVelocity) {
    // The exact case from the audit.
    const std::vector<double> v{0.5, 0.4, 0.3, 0.2};
    const double dx = 0.2;
    const double dt = 0.05;
    std::vector<double> u{0.0, 0.0, 0.0, 1.0};

    const double m0 = total_mass(u, dx);
    for (int step = 0; step < 5; ++step) {
        u = ms::cfd::upwind_fvm_advection(u, v, dt, dx, ms::cfd::BoundaryCondition::Periodic);
        ASSERT_FALSE(u.empty()) << "step " << step;
        EXPECT_NEAR(total_mass(u, dx), m0, 1e-12) << "mass drifted at step " << step;
    }
}

TEST(CfdPeriodicConservation, OneDimensionalUniformVelocity) {
    const std::vector<double> v(6, 0.7);
    const double dx = 0.1;
    const double dt = 0.01;
    std::vector<double> u{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    const double m0 = total_mass(u, dx);
    for (int step = 0; step < 20; ++step) {
        u = ms::cfd::upwind_fvm_advection(u, v, dt, dx, ms::cfd::BoundaryCondition::Periodic);
        ASSERT_FALSE(u.empty());
    }
    EXPECT_NEAR(total_mass(u, dx), m0, 1e-12);
}

TEST(CfdPeriodicConservation, OneDimensionalNegativeVelocity) {
    // Upwinding takes the other branch when the face velocity is negative; the
    // wrap-around face must still match.
    const std::vector<double> v{-0.5, -0.4, -0.3, -0.6, -0.2};
    const double dx = 0.25;
    const double dt = 0.05;
    std::vector<double> u{2.0, 0.0, 1.0, 0.0, 3.0};

    const double m0 = total_mass(u, dx);
    for (int step = 0; step < 10; ++step) {
        u = ms::cfd::upwind_fvm_advection(u, v, dt, dx, ms::cfd::BoundaryCondition::Periodic);
        ASSERT_FALSE(u.empty());
        EXPECT_NEAR(total_mass(u, dx), m0, 1e-12) << "step " << step;
    }
}

TEST(CfdPeriodicConservation, ZeroFluxBoundariesStillHoldTheirContract) {
    // With zero-flux walls the boundary fluxes are zero by construction, so the
    // interior faces must be unaffected by the fix and mass is still conserved.
    const std::vector<double> v(5, 0.4);
    const double dx = 0.2;
    const double dt = 0.05;
    std::vector<double> u{0.0, 1.0, 2.0, 1.0, 0.0};

    const double m0 = total_mass(u, dx);
    for (int step = 0; step < 5; ++step) {
        u = ms::cfd::upwind_fvm_advection(u, v, dt, dx, ms::cfd::BoundaryCondition::ZeroFlux);
        ASSERT_FALSE(u.empty());
    }
    // Zero-flux walls trap everything inside, so the total is unchanged.
    EXPECT_NEAR(total_mass(u, dx), m0, 1e-12);
}

TEST(CfdPeriodicConservation, TwoDimensionalPeriodic) {
    const std::size_t nx = 4;
    const std::size_t ny = 3;
    std::vector<std::vector<double>> u(ny, std::vector<double>(nx, 0.0));
    u[1][2] = 1.0;
    std::vector<double> vx(nx * ny);
    std::vector<double> vy(nx * ny);
    for (std::size_t k = 0; k < vx.size(); ++k) {
        vx[k] = 0.3 + 0.05 * static_cast<double>(k % 3);
        vy[k] = 0.2 + 0.05 * static_cast<double>(k % 2);
    }
    const double dx = 0.25;
    const double dy = 0.25;
    const double dt = 0.05;

    auto mass2d = [&](const std::vector<std::vector<double>>& f) {
        double m = 0.0;
        for (const auto& row : f) {
            for (double x : row) {
                m += x;
            }
        }
        return m * dx * dy;
    };

    const double m0 = mass2d(u);
    for (int step = 0; step < 5; ++step) {
        u = ms::cfd::upwind_fvm_advection_2d(u, vx, vy, dt, dx, dy,
                                             ms::cfd::BoundaryCondition::Periodic,
                                             ms::cfd::BoundaryCondition::Periodic);
        ASSERT_FALSE(u.empty()) << "step " << step;
        EXPECT_NEAR(mass2d(u), m0, 1e-12) << "mass drifted at step " << step;
    }
}
