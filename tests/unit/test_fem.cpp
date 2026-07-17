#include "ms/fem/fem.hpp"
#include <cmath>
#include <gtest/gtest.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ms;
using namespace ms::fem;

TEST(FemMesh1D, uniform_nodes_and_connectivity) {
    const Mesh1D mesh = mesh1d(0.0, 1.0, 4);
    ASSERT_EQ(mesh.nodes.size(), 5u);
    ASSERT_EQ(mesh.connectivity.size(), 4u);
    EXPECT_NEAR(mesh.nodes.front(), 0.0, 1e-14);
    EXPECT_NEAR(mesh.nodes.back(), 1.0, 1e-14);
    EXPECT_NEAR(mesh.nodes[2], 0.5, 1e-14);
    EXPECT_EQ(mesh.connectivity[0][0], 0u);
    EXPECT_EQ(mesh.connectivity[0][1], 1u);
    EXPECT_EQ(mesh.connectivity[3][0], 3u);
    EXPECT_EQ(mesh.connectivity[3][1], 4u);
}

TEST(FemMesh1D, rejects_invalid_domain) {
    EXPECT_THROW(mesh1d(1.0, 0.0, 4), std::invalid_argument);
    EXPECT_THROW(mesh1d(0.0, 0.0, 4), std::invalid_argument);
    EXPECT_THROW(mesh1d(0.0, 1.0, 0), std::invalid_argument);
}

TEST(FemLagrangeBasis, partition_of_unity) {
    const LagrangeBasis basis = lagrange_basis();
    EXPECT_EQ(basis.degree, 1);
    for (double xi : {0.0, 0.25, 0.5, 0.75, 1.0}) {
        const auto N = basis.evaluate(xi);
        EXPECT_NEAR(N[0] + N[1], 1.0, 1e-14);
    }
}

TEST(FemLagrangeBasis, endpoint_interpolation) {
    const LagrangeBasis basis = lagrange_basis();
    const auto N0 = basis.evaluate(0.0);
    const auto N1 = basis.evaluate(1.0);
    EXPECT_NEAR(N0[0], 1.0, 1e-14);
    EXPECT_NEAR(N0[1], 0.0, 1e-14);
    EXPECT_NEAR(N1[0], 0.0, 1e-14);
    EXPECT_NEAR(N1[1], 1.0, 1e-14);
}

TEST(FemLagrangeBasis, constant_derivatives) {
    const LagrangeBasis basis = lagrange_basis();
    const auto d0 = basis.derivative(0.3);
    const auto d1 = basis.derivative(0.8);
    EXPECT_NEAR(d0[0], -1.0, 1e-14);
    EXPECT_NEAR(d0[1], 1.0, 1e-14);
    EXPECT_NEAR(d1[0], -1.0, 1e-14);
    EXPECT_NEAR(d1[1], 1.0, 1e-14);
}

TEST(FemStiffness1D, single_element_pattern) {
    const Mesh1D mesh = mesh1d(0.0, 2.0, 1);
    const ColMatrix<double> K = assemble_stiffness_1d(mesh);
    ASSERT_EQ(K.rows(), 2u);
    EXPECT_NEAR(K(0, 0), 0.5, 1e-14);
    EXPECT_NEAR(K(0, 1), -0.5, 1e-14);
    EXPECT_NEAR(K(1, 0), -0.5, 1e-14);
    EXPECT_NEAR(K(1, 1), 0.5, 1e-14);
}

TEST(FemStiffness1D, tridiagonal_structure) {
    const Mesh1D mesh = mesh1d(0.0, 1.0, 5);
    const ColMatrix<double> K = assemble_stiffness_1d(mesh);
    ASSERT_EQ(K.rows(), 6u);
    for (std::size_t i = 0; i < K.rows(); ++i) {
        for (std::size_t j = 0; j < K.cols(); ++j) {
            if (i + 1 < j || j + 1 < i) {
                EXPECT_NEAR(K(i, j), 0.0, 1e-14);
            }
        }
    }
    EXPECT_GT(std::abs(K(0, 1)), 0.0);
    EXPECT_GT(std::abs(K(2, 3)), 0.0);
}

TEST(FemStiffness1D, symmetry) {
    const Mesh1D mesh = mesh1d(-1.0, 3.0, 8);
    const ColMatrix<double> K = assemble_stiffness_1d(mesh);
    for (std::size_t i = 0; i < K.rows(); ++i) {
        for (std::size_t j = 0; j < K.cols(); ++j) {
            EXPECT_NEAR(K(i, j), K(j, i), 1e-14);
        }
    }
}

TEST(FemDirichlet, enforces_boundary_values) {
    const Mesh1D mesh = mesh1d(0.0, 1.0, 2);
    ColMatrix<double> K = assemble_stiffness_1d(mesh);
    ColMatrix<double> f(3, 1, 1.0);
    apply_dirichlet(K, f, {0, 2}, {0.0, 0.0});

    for (std::size_t j = 0; j < 3; ++j) {
        if (j != 0) {
            EXPECT_NEAR(K(0, j), 0.0, 1e-14);
        }
        if (j != 2) {
            EXPECT_NEAR(K(2, j), 0.0, 1e-14);
        }
    }
    EXPECT_NEAR(K(0, 0), 1.0, 1e-14);
    EXPECT_NEAR(K(2, 2), 1.0, 1e-14);
    EXPECT_NEAR(f(0, 0), 0.0, 1e-14);
    EXPECT_NEAR(f(2, 0), 0.0, 1e-14);
}

TEST(FemLoad1D, integrates_constant_source) {
    const Mesh1D mesh = mesh1d(0.0, 1.0, 2);
    const ColMatrix<double> load = assemble_load_1d(mesh, [](double) { return 1.0; });
    double total = 0.0;
    for (std::size_t i = 0; i < load.rows(); ++i) {
        total += load(i, 0);
    }
    EXPECT_NEAR(total, 1.0, 1e-12);
}

TEST(FemSolve, poisson_sin_pi) {
    const Mesh1D mesh = mesh1d(0.0, 1.0, 100);
    ColMatrix<double> K = assemble_stiffness_1d(mesh);
    ColMatrix<double> f = assemble_load_1d(mesh, [](double x) {
        return M_PI * M_PI * std::sin(M_PI * x);
    });

    apply_dirichlet(K, f, {0, mesh.nodes.size() - 1}, {0.0, 0.0});

    const auto u = solve_fem(K, f);
    ASSERT_TRUE(u.has_value());

    for (std::size_t i = 0; i < mesh.nodes.size(); ++i) {
        const double x = mesh.nodes[i];
        EXPECT_NEAR((*u)(i, 0), std::sin(M_PI * x), 5e-3);
    }
}

TEST(FemSolve, linear_exact_on_uniform_mesh) {
    const Mesh1D mesh = mesh1d(0.0, 1.0, 4);
    ColMatrix<double> K = assemble_stiffness_1d(mesh);
    ColMatrix<double> f = assemble_load_1d(mesh, [](double) { return 0.0; });
    apply_dirichlet(K, f, {0, mesh.nodes.size() - 1}, {0.0, 1.0});

    const auto u = solve_fem(K, f);
    ASSERT_TRUE(u.has_value());

    for (std::size_t i = 0; i < mesh.nodes.size(); ++i) {
        EXPECT_NEAR((*u)(i, 0), mesh.nodes[i], 1e-10);
    }
}
