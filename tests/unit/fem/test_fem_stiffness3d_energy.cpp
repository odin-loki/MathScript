// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
// Regression test for the 3D P1 stiffness assembly.
//
// add_tet_stiffness applied J^-1 to the reference gradients where the chain rule
// calls for J^-T. invert3x3(e1, e2, e3, .) inverts the matrix holding the edge
// vectors as ROWS -- that is J^T -- so its result is already J^-T, and the extra
// transpose turned it back into J^-1. The two agree only when J is symmetric, so
// the error was invisible on any test that only checked structure.
//
// The property below catches it: a P1 space reproduces linear functions exactly,
// so for u(x) = a.x + b the discrete Dirichlet energy u^T K u must equal the
// exact integral |a|^2 * Volume. Against the pre-fix code u = x on the unit cube
// gave 1.667 instead of 1, u = z gave 1.333, and u = x + 2y + 3z gave 35
// instead of 14.
//
// Note that K * ones = 0 and symmetry hold either way -- they survive any
// invertible transform of the gradients -- which is why they never caught this.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "ms/fem/fem.hpp"

namespace {

struct Linear {
    double ax, ay, az, b;
};

double dirichlet_energy(const ms::fem::Mesh3D& mesh,
                        const ms::ColMatrix<double>& K,
                        const Linear& u) {
    const std::size_t n = mesh.nodes.size();
    std::vector<double> vals(n);
    for (std::size_t i = 0; i < n; ++i) {
        vals[i] = u.ax * mesh.nodes[i][0] + u.ay * mesh.nodes[i][1] +
                  u.az * mesh.nodes[i][2] + u.b;
    }
    double e = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            e += vals[i] * K(i, j) * vals[j];
        }
    }
    return e;
}

} // namespace

TEST(FemStiffness3D, ReproducesLinearFieldEnergyOnTheUnitCube) {
    const auto mesh = ms::fem::mesh3d_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 3, 3, 3);
    ASSERT_TRUE(mesh.has_value());
    const auto K = ms::fem::assemble_stiffness_3d(*mesh);
    ASSERT_TRUE(K.has_value());

    // Volume is 1, so the exact energy of u = a.x + b is |a|^2.
    const std::vector<Linear> cases{
        {1.0, 0.0, 0.0, 0.0},
        {0.0, 1.0, 0.0, 0.0},
        {0.0, 0.0, 1.0, 0.0},
        {1.0, 2.0, 3.0, 0.5},
        {-2.0, 1.0, 4.0, -1.0},
    };
    for (const auto& u : cases) {
        const double exact = u.ax * u.ax + u.ay * u.ay + u.az * u.az;
        EXPECT_NEAR(dirichlet_energy(*mesh, *K, u), exact, 1e-10 * exact)
            << "a = (" << u.ax << ", " << u.ay << ", " << u.az << ")";
    }
}

TEST(FemStiffness3D, EnergyIsIsotropicOnASymmetricMesh) {
    // The pre-fix code gave 1.667 along x and y but 1.333 along z on a mesh that
    // is symmetric under permuting the axes -- an asymmetry with no physical
    // meaning, and the clearest signal that the gradients were wrong.
    const auto mesh = ms::fem::mesh3d_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 3, 3, 3);
    ASSERT_TRUE(mesh.has_value());
    const auto K = ms::fem::assemble_stiffness_3d(*mesh);
    ASSERT_TRUE(K.has_value());

    const double ex = dirichlet_energy(*mesh, *K, {1.0, 0.0, 0.0, 0.0});
    const double ey = dirichlet_energy(*mesh, *K, {0.0, 1.0, 0.0, 0.0});
    const double ez = dirichlet_energy(*mesh, *K, {0.0, 0.0, 1.0, 0.0});
    EXPECT_NEAR(ex, ey, 1e-12);
    EXPECT_NEAR(ey, ez, 1e-12);
}

TEST(FemStiffness3D, EnergyScalesWithTheDomain) {
    // On [0,2]^3 the volume is 8, so u = x has energy |a|^2 * V = 8.
    const auto mesh = ms::fem::mesh3d_box(0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2, 2, 2);
    ASSERT_TRUE(mesh.has_value());
    const auto K = ms::fem::assemble_stiffness_3d(*mesh);
    ASSERT_TRUE(K.has_value());
    EXPECT_NEAR(dirichlet_energy(*mesh, *K, {1.0, 0.0, 0.0, 0.0}), 8.0, 1e-9);
}

TEST(FemStiffness3D, StructuralPropertiesStillHold) {
    // These held before the fix too; keep them so a future change cannot trade
    // one correctness property for another.
    const auto mesh = ms::fem::mesh3d_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2, 2, 2);
    ASSERT_TRUE(mesh.has_value());
    const auto K = ms::fem::assemble_stiffness_3d(*mesh);
    ASSERT_TRUE(K.has_value());
    const std::size_t n = mesh->nodes.size();

    for (std::size_t i = 0; i < n; ++i) {
        double row = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            row += (*K)(i, j);
            EXPECT_NEAR((*K)(i, j), (*K)(j, i), 1e-12);  // symmetric
        }
        EXPECT_NEAR(row, 0.0, 1e-12);  // annihilates constants
    }
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_GT((*K)(i, i), 0.0);  // positive diagonal
    }
}
