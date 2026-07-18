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

namespace {

std::vector<std::size_t> rectangular_boundary_nodes(std::size_t nx, std::size_t ny) {
    const std::size_t n_nodes_x = nx + 1;
    std::vector<std::size_t> boundary;
    for (std::size_t j = 0; j <= ny; ++j) {
        for (std::size_t i = 0; i <= nx; ++i) {
            if (i == 0 || i == nx || j == 0 || j == ny) {
                boundary.push_back(i + j * n_nodes_x);
            }
        }
    }
    return boundary;
}

Mesh2D single_right_triangle() {
    Mesh2D mesh;
    mesh.nodes = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};
    mesh.triangles = {{0, 1, 2}};
    return mesh;
}

} // namespace

TEST(FemMesh2D, rectangular_nodes_and_triangles) {
    const Mesh2D mesh = mesh2d_rectangular(0.0, 0.0, 2.0, 1.0, 2, 1);
    ASSERT_EQ(mesh.nodes.size(), 6u);
    ASSERT_EQ(mesh.triangles.size(), 4u);
    EXPECT_NEAR(mesh.nodes[0][0], 0.0, 1e-14);
    EXPECT_NEAR(mesh.nodes[0][1], 0.0, 1e-14);
    EXPECT_NEAR(mesh.nodes[2][0], 2.0, 1e-14);
    EXPECT_NEAR(mesh.nodes[5][0], 2.0, 1e-14);
    EXPECT_NEAR(mesh.nodes[5][1], 1.0, 1e-14);
    EXPECT_EQ(mesh.triangles[0][0], 0u);
    EXPECT_EQ(mesh.triangles[0][1], 1u);
    EXPECT_EQ(mesh.triangles[0][2], 3u);
}

TEST(FemMesh2D, rejects_invalid_domain) {
    EXPECT_THROW(mesh2d_rectangular(1.0, 0.0, 0.0, 1.0, 2, 2), std::invalid_argument);
    EXPECT_THROW(mesh2d_rectangular(0.0, 1.0, 1.0, 0.0, 2, 2), std::invalid_argument);
    EXPECT_THROW(mesh2d_rectangular(0.0, 0.0, 1.0, 1.0, 0, 2), std::invalid_argument);
    EXPECT_THROW(mesh2d_rectangular(0.0, 0.0, 1.0, 1.0, 2, 0), std::invalid_argument);
}

TEST(FemStiffness2D, single_triangle_pattern) {
    const Mesh2D mesh = single_right_triangle();
    const ColMatrix<double> K = assemble_stiffness_2d(mesh);
    ASSERT_EQ(K.rows(), 3u);
    EXPECT_NEAR(K(0, 0), 1.0, 1e-14);
    EXPECT_NEAR(K(1, 1), 0.5, 1e-14);
    EXPECT_NEAR(K(2, 2), 0.5, 1e-14);
    EXPECT_NEAR(K(0, 1), -0.5, 1e-14);
    EXPECT_NEAR(K(0, 2), -0.5, 1e-14);
    EXPECT_NEAR(K(1, 2), 0.0, 1e-14);
}

TEST(FemStiffness2D, symmetry) {
    const Mesh2D mesh = mesh2d_rectangular(0.0, 0.0, 1.0, 1.0, 3, 3);
    const ColMatrix<double> K = assemble_stiffness_2d(mesh);
    for (std::size_t i = 0; i < K.rows(); ++i) {
        for (std::size_t j = 0; j < K.cols(); ++j) {
            EXPECT_NEAR(K(i, j), K(j, i), 1e-14);
        }
    }
}

TEST(FemLoad2D, integrates_constant_source) {
    const Mesh2D mesh = mesh2d_rectangular(0.0, 0.0, 1.0, 1.0, 2, 2);
    const ColMatrix<double> load = assemble_load_2d(mesh, [](double, double) { return 1.0; });
    double total = 0.0;
    for (std::size_t i = 0; i < load.rows(); ++i) {
        total += load(i, 0);
    }
    EXPECT_NEAR(total, 1.0, 1e-12);
}

TEST(FemSolve2D, poisson_sin_pi) {
    constexpr std::size_t nx = 20;
    constexpr std::size_t ny = 20;
    const Mesh2D mesh = mesh2d_rectangular(0.0, 0.0, 1.0, 1.0, nx, ny);
    ColMatrix<double> K = assemble_stiffness_2d(mesh);
    ColMatrix<double> f = assemble_load_2d(mesh, [](double x, double y) {
        return 2.0 * M_PI * M_PI * std::sin(M_PI * x) * std::sin(M_PI * y);
    });

    const auto boundary = rectangular_boundary_nodes(nx, ny);
    std::vector<double> boundary_values(boundary.size(), 0.0);
    apply_dirichlet(K, f, boundary, boundary_values);

    const auto u = solve_fem(K, f);
    ASSERT_TRUE(u.has_value());

    for (std::size_t i = 0; i < mesh.nodes.size(); ++i) {
        const double x = mesh.nodes[i][0];
        const double y = mesh.nodes[i][1];
        EXPECT_NEAR((*u)(i, 0), std::sin(M_PI * x) * std::sin(M_PI * y), 5e-2);
    }
}

TEST(FemSolve2D, linear_exact_on_uniform_mesh) {
    constexpr std::size_t nx = 4;
    constexpr std::size_t ny = 4;
    const Mesh2D mesh = mesh2d_rectangular(0.0, 0.0, 1.0, 1.0, nx, ny);
    ColMatrix<double> K = assemble_stiffness_2d(mesh);
    ColMatrix<double> f = assemble_load_2d(mesh, [](double, double) { return 0.0; });

    const auto boundary = rectangular_boundary_nodes(nx, ny);
    std::vector<double> boundary_values;
    boundary_values.reserve(boundary.size());
    for (std::size_t idx : boundary) {
        const double x = mesh.nodes[idx][0];
        const double y = mesh.nodes[idx][1];
        boundary_values.push_back(x + y);
    }
    apply_dirichlet(K, f, boundary, boundary_values);

    const auto u = solve_fem(K, f);
    ASSERT_TRUE(u.has_value());

    for (std::size_t i = 0; i < mesh.nodes.size(); ++i) {
        const double x = mesh.nodes[i][0];
        const double y = mesh.nodes[i][1];
        EXPECT_NEAR((*u)(i, 0), x + y, 1e-10);
    }
}

namespace {

Mesh3D single_corner_tetrahedron() {
    Mesh3D mesh;
    mesh.nodes = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    mesh.tetrahedra = {{0, 1, 2, 3}};
    return mesh;
}

std::vector<std::size_t> box_boundary_nodes(
    std::size_t nx, std::size_t ny, std::size_t nz) {
    const std::size_t n_nodes_x = nx + 1;
    const std::size_t n_nodes_y = ny + 1;
    const std::size_t n_nodes_xy = n_nodes_x * n_nodes_y;
    std::vector<std::size_t> boundary;
    for (std::size_t k = 0; k <= nz; ++k) {
        for (std::size_t j = 0; j <= ny; ++j) {
            for (std::size_t i = 0; i <= nx; ++i) {
                if (i == 0 || i == nx || j == 0 || j == ny || k == 0 || k == nz) {
                    boundary.push_back(i + j * n_nodes_x + k * n_nodes_xy);
                }
            }
        }
    }
    return boundary;
}

} // namespace

TEST(FemMesh3D, box_nodes_and_tetrahedra) {
    const Mesh3D mesh = mesh3d_box(0.0, 0.0, 0.0, 2.0, 1.0, 1.0, 2, 1, 1);
    ASSERT_EQ(mesh.nodes.size(), 12u);
    ASSERT_EQ(mesh.tetrahedra.size(), 12u);
    EXPECT_NEAR(mesh.nodes[0][0], 0.0, 1e-14);
    EXPECT_NEAR(mesh.nodes[0][1], 0.0, 1e-14);
    EXPECT_NEAR(mesh.nodes[0][2], 0.0, 1e-14);
    EXPECT_NEAR(mesh.nodes[11][0], 2.0, 1e-14);
    EXPECT_NEAR(mesh.nodes[11][1], 1.0, 1e-14);
    EXPECT_NEAR(mesh.nodes[11][2], 1.0, 1e-14);
    EXPECT_EQ(mesh.tetrahedra[0][0], 0u);
    EXPECT_EQ(mesh.tetrahedra[0][1], 1u);
    EXPECT_EQ(mesh.tetrahedra[0][2], 4u);
    EXPECT_EQ(mesh.tetrahedra[0][3], 10u);
}

TEST(FemMesh3D, rejects_invalid_domain) {
    EXPECT_THROW(mesh3d_box(1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1, 1, 1), std::invalid_argument);
    EXPECT_THROW(mesh3d_box(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1, 1, 1), std::invalid_argument);
    EXPECT_THROW(mesh3d_box(0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1, 1, 1), std::invalid_argument);
    EXPECT_THROW(mesh3d_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0, 1, 1), std::invalid_argument);
    EXPECT_THROW(mesh3d_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1, 0, 1), std::invalid_argument);
    EXPECT_THROW(mesh3d_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1, 1, 0), std::invalid_argument);
}

TEST(FemStiffness3D, single_tetrahedron_pattern) {
    const Mesh3D mesh = single_corner_tetrahedron();
    const ColMatrix<double> K = assemble_stiffness_3d(mesh);
    ASSERT_EQ(K.rows(), 4u);
    EXPECT_NEAR(K(0, 0), 0.5, 1e-14);
    EXPECT_NEAR(K(1, 1), 1.0 / 6.0, 1e-14);
    EXPECT_NEAR(K(2, 2), 1.0 / 6.0, 1e-14);
    EXPECT_NEAR(K(3, 3), 1.0 / 6.0, 1e-14);
    EXPECT_NEAR(K(0, 1), -1.0 / 6.0, 1e-14);
    EXPECT_NEAR(K(0, 2), -1.0 / 6.0, 1e-14);
    EXPECT_NEAR(K(0, 3), -1.0 / 6.0, 1e-14);
    EXPECT_NEAR(K(1, 2), 0.0, 1e-14);
    EXPECT_NEAR(K(1, 3), 0.0, 1e-14);
    EXPECT_NEAR(K(2, 3), 0.0, 1e-14);
}

TEST(FemStiffness3D, symmetry) {
    const Mesh3D mesh = mesh3d_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2, 2, 2);
    const ColMatrix<double> K = assemble_stiffness_3d(mesh);
    for (std::size_t i = 0; i < K.rows(); ++i) {
        for (std::size_t j = 0; j < K.cols(); ++j) {
            EXPECT_NEAR(K(i, j), K(j, i), 1e-14);
        }
    }
}

TEST(FemStiffness3D, row_sums_vanish) {
    const Mesh3D mesh = mesh3d_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1, 1, 1);
    const ColMatrix<double> K = assemble_stiffness_3d(mesh);
    for (std::size_t i = 0; i < K.rows(); ++i) {
        double row_sum = 0.0;
        for (std::size_t j = 0; j < K.cols(); ++j) {
            row_sum += K(i, j);
        }
        EXPECT_NEAR(row_sum, 0.0, 1e-12);
    }
}

TEST(FemSolve3D, linear_exact_on_uniform_mesh) {
    constexpr std::size_t nx = 3;
    constexpr std::size_t ny = 3;
    constexpr std::size_t nz = 3;
    const Mesh3D mesh = mesh3d_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, nx, ny, nz);
    ColMatrix<double> K = assemble_stiffness_3d(mesh);
    ColMatrix<double> f = assemble_load_3d(mesh, [](double, double, double) { return 0.0; });

    const auto boundary = box_boundary_nodes(nx, ny, nz);
    std::vector<double> boundary_values;
    boundary_values.reserve(boundary.size());
    for (std::size_t idx : boundary) {
        const double x = mesh.nodes[idx][0];
        const double y = mesh.nodes[idx][1];
        const double z = mesh.nodes[idx][2];
        boundary_values.push_back(x + y + z);
    }
    apply_dirichlet(K, f, boundary, boundary_values);

    const auto u = solve_fem_3d(K, f);
    ASSERT_TRUE(u.has_value());

    for (std::size_t i = 0; i < mesh.nodes.size(); ++i) {
        const double x = mesh.nodes[i][0];
        const double y = mesh.nodes[i][1];
        const double z = mesh.nodes[i][2];
        EXPECT_NEAR((*u)(i, 0), x + y + z, 1e-9);
    }
}

TEST(FemLoad3D, integrates_constant_source) {
    const Mesh3D mesh = mesh3d_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2, 2, 2);
    const ColMatrix<double> load = assemble_load_3d(mesh, [](double, double, double) { return 1.0; });
    double total = 0.0;
    for (std::size_t i = 0; i < load.rows(); ++i) {
        total += load(i, 0);
    }
    EXPECT_NEAR(total, 1.0, 1e-12);
}

TEST(FemSolve3D, poisson_sin_pi) {
    constexpr std::size_t nx = 8;
    constexpr std::size_t ny = 8;
    constexpr std::size_t nz = 8;
    const Mesh3D mesh = mesh3d_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, nx, ny, nz);
    ColMatrix<double> K = assemble_stiffness_3d(mesh);
    ColMatrix<double> f = assemble_load_3d(mesh, [](double x, double y, double z) {
        return 3.0 * M_PI * M_PI * std::sin(M_PI * x) * std::sin(M_PI * y) * std::sin(M_PI * z);
    });

    const auto boundary = box_boundary_nodes(nx, ny, nz);
    std::vector<double> boundary_values(boundary.size(), 0.0);
    apply_dirichlet(K, f, boundary, boundary_values);

    const auto u = solve_fem_3d(K, f);
    ASSERT_TRUE(u.has_value());

    for (std::size_t i = 0; i < mesh.nodes.size(); ++i) {
        const double x = mesh.nodes[i][0];
        const double y = mesh.nodes[i][1];
        const double z = mesh.nodes[i][2];
        EXPECT_NEAR(
            (*u)(i, 0),
            std::sin(M_PI * x) * std::sin(M_PI * y) * std::sin(M_PI * z),
            8e-2);
    }
}
