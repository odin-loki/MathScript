#include "ms/fem/fem.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <variant>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ms;
using namespace ms::fem;

TEST(FemMesh1D, uniform_nodes_and_connectivity) {
    const auto mesh_r = mesh1d(0.0, 1.0, 4);
    ASSERT_TRUE(mesh_r.has_value());
    const Mesh1D& mesh = *mesh_r;
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
    {
        const auto r = mesh1d(1.0, 0.0, 4);
        ASSERT_FALSE(r.has_value());
    }
    {
        const auto r = mesh1d(0.0, 0.0, 4);
        ASSERT_FALSE(r.has_value());
    }
    {
        const auto r = mesh1d(0.0, 1.0, 0);
        ASSERT_FALSE(r.has_value());
    }
}

TEST(FemLagrangeBasis, partition_of_unity) {
    const LagrangeBasis basis = lagrange_basis();
    EXPECT_EQ(basis.degree, 1);
    for (double xi : {0.0, 0.25, 0.5, 0.75, 1.0}) {
        const auto N_r = basis.evaluate(xi);
        ASSERT_TRUE(N_r.has_value());
        const auto& N = *N_r;
        EXPECT_NEAR(N[0] + N[1], 1.0, 1e-14);
    }
}

TEST(FemLagrangeBasis, endpoint_interpolation) {
    const LagrangeBasis basis = lagrange_basis();
    const auto N0_r = basis.evaluate(0.0);
    ASSERT_TRUE(N0_r.has_value());
    const auto N1_r = basis.evaluate(1.0);
    ASSERT_TRUE(N1_r.has_value());
    const auto& N0 = *N0_r;
    const auto& N1 = *N1_r;
    EXPECT_NEAR(N0[0], 1.0, 1e-14);
    EXPECT_NEAR(N0[1], 0.0, 1e-14);
    EXPECT_NEAR(N1[0], 0.0, 1e-14);
    EXPECT_NEAR(N1[1], 1.0, 1e-14);
}

TEST(FemLagrangeBasis, constant_derivatives) {
    const LagrangeBasis basis = lagrange_basis();
    const auto d0_r = basis.derivative(0.3);
    ASSERT_TRUE(d0_r.has_value());
    const auto d1_r = basis.derivative(0.8);
    ASSERT_TRUE(d1_r.has_value());
    const auto& d0 = *d0_r;
    const auto& d1 = *d1_r;
    EXPECT_NEAR(d0[0], -1.0, 1e-14);
    EXPECT_NEAR(d0[1], 1.0, 1e-14);
    EXPECT_NEAR(d1[0], -1.0, 1e-14);
    EXPECT_NEAR(d1[1], 1.0, 1e-14);
}

TEST(FemStiffness1D, single_element_pattern) {
    const auto mesh_r = mesh1d(0.0, 2.0, 1);
    ASSERT_TRUE(mesh_r.has_value());
    const Mesh1D& mesh = *mesh_r;
    const auto K_r = assemble_stiffness_1d(mesh);
    ASSERT_TRUE(K_r.has_value());
    const ColMatrix<double>& K = *K_r;
    ASSERT_EQ(K.rows(), 2u);
    EXPECT_NEAR(K(0, 0), 0.5, 1e-14);
    EXPECT_NEAR(K(0, 1), -0.5, 1e-14);
    EXPECT_NEAR(K(1, 0), -0.5, 1e-14);
    EXPECT_NEAR(K(1, 1), 0.5, 1e-14);
}

TEST(FemStiffness1D, tridiagonal_structure) {
    const auto mesh_r = mesh1d(0.0, 1.0, 5);
    ASSERT_TRUE(mesh_r.has_value());
    const Mesh1D& mesh = *mesh_r;
    const auto K_r = assemble_stiffness_1d(mesh);
    ASSERT_TRUE(K_r.has_value());
    const ColMatrix<double>& K = *K_r;
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
    const auto mesh_r = mesh1d(-1.0, 3.0, 8);
    ASSERT_TRUE(mesh_r.has_value());
    const Mesh1D& mesh = *mesh_r;
    const auto K_r = assemble_stiffness_1d(mesh);
    ASSERT_TRUE(K_r.has_value());
    const ColMatrix<double>& K = *K_r;
    for (std::size_t i = 0; i < K.rows(); ++i) {
        for (std::size_t j = 0; j < K.cols(); ++j) {
            EXPECT_NEAR(K(i, j), K(j, i), 1e-14);
        }
    }
}

TEST(FemDirichlet, enforces_boundary_values) {
    const auto mesh_r = mesh1d(0.0, 1.0, 2);
    ASSERT_TRUE(mesh_r.has_value());
    const Mesh1D& mesh = *mesh_r;
    auto K_r = assemble_stiffness_1d(mesh);
    ASSERT_TRUE(K_r.has_value());
    ColMatrix<double> K = *K_r;
    ColMatrix<double> f(3, 1, 1.0);
    const auto dir_r = apply_dirichlet(K, f, {0, 2}, {0.0, 0.0});
    ASSERT_TRUE(dir_r.has_value());

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
    const auto mesh_r = mesh1d(0.0, 1.0, 2);
    ASSERT_TRUE(mesh_r.has_value());
    const Mesh1D& mesh = *mesh_r;
    const auto load_r = assemble_load_1d(mesh, [](double) { return 1.0; });
    ASSERT_TRUE(load_r.has_value());
    const ColMatrix<double>& load = *load_r;
    double total = 0.0;
    for (std::size_t i = 0; i < load.rows(); ++i) {
        total += load(i, 0);
    }
    EXPECT_NEAR(total, 1.0, 1e-12);
}

TEST(FemSolve, poisson_sin_pi) {
    const auto mesh_r = mesh1d(0.0, 1.0, 100);
    ASSERT_TRUE(mesh_r.has_value());
    const Mesh1D& mesh = *mesh_r;
    auto K_r = assemble_stiffness_1d(mesh);
    ASSERT_TRUE(K_r.has_value());
    ColMatrix<double> K = *K_r;
    auto f_r = assemble_load_1d(mesh, [](double x) {
        return M_PI * M_PI * std::sin(M_PI * x);
    });
    ASSERT_TRUE(f_r.has_value());
    ColMatrix<double> f = *f_r;

    const auto dir_r = apply_dirichlet(K, f, {0, mesh.nodes.size() - 1}, {0.0, 0.0});
    ASSERT_TRUE(dir_r.has_value());

    const auto u = solve_fem(K, f);
    ASSERT_TRUE(u.has_value());

    for (std::size_t i = 0; i < mesh.nodes.size(); ++i) {
        const double x = mesh.nodes[i];
        EXPECT_NEAR((*u)(i, 0), std::sin(M_PI * x), 5e-3);
    }
}

TEST(FemSolve, linear_exact_on_uniform_mesh) {
    const auto mesh_r = mesh1d(0.0, 1.0, 4);
    ASSERT_TRUE(mesh_r.has_value());
    const Mesh1D& mesh = *mesh_r;
    auto K_r = assemble_stiffness_1d(mesh);
    ASSERT_TRUE(K_r.has_value());
    ColMatrix<double> K = *K_r;
    auto f_r = assemble_load_1d(mesh, [](double) { return 0.0; });
    ASSERT_TRUE(f_r.has_value());
    ColMatrix<double> f = *f_r;
    const auto dir_r = apply_dirichlet(K, f, {0, mesh.nodes.size() - 1}, {0.0, 1.0});
    ASSERT_TRUE(dir_r.has_value());

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
    const auto mesh_r = mesh2d_rectangular(0.0, 0.0, 2.0, 1.0, 2, 1);
    ASSERT_TRUE(mesh_r.has_value());
    const Mesh2D& mesh = *mesh_r;
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
    {
        const auto r = mesh2d_rectangular(1.0, 0.0, 0.0, 1.0, 2, 2);
        ASSERT_FALSE(r.has_value());
    }
    {
        const auto r = mesh2d_rectangular(0.0, 1.0, 1.0, 0.0, 2, 2);
        ASSERT_FALSE(r.has_value());
    }
    {
        const auto r = mesh2d_rectangular(0.0, 0.0, 1.0, 1.0, 0, 2);
        ASSERT_FALSE(r.has_value());
    }
    {
        const auto r = mesh2d_rectangular(0.0, 0.0, 1.0, 1.0, 2, 0);
        ASSERT_FALSE(r.has_value());
    }
}

TEST(FemStiffness2D, single_triangle_pattern) {
    const Mesh2D mesh = single_right_triangle();
    const auto K_r = assemble_stiffness_2d(mesh);
    ASSERT_TRUE(K_r.has_value());
    const ColMatrix<double>& K = *K_r;
    ASSERT_EQ(K.rows(), 3u);
    EXPECT_NEAR(K(0, 0), 1.0, 1e-14);
    EXPECT_NEAR(K(1, 1), 0.5, 1e-14);
    EXPECT_NEAR(K(2, 2), 0.5, 1e-14);
    EXPECT_NEAR(K(0, 1), -0.5, 1e-14);
    EXPECT_NEAR(K(0, 2), -0.5, 1e-14);
    EXPECT_NEAR(K(1, 2), 0.0, 1e-14);
}

TEST(FemStiffness2D, symmetry) {
    const auto mesh_r = mesh2d_rectangular(0.0, 0.0, 1.0, 1.0, 3, 3);
    ASSERT_TRUE(mesh_r.has_value());
    const Mesh2D& mesh = *mesh_r;
    const auto K_r = assemble_stiffness_2d(mesh);
    ASSERT_TRUE(K_r.has_value());
    const ColMatrix<double>& K = *K_r;
    for (std::size_t i = 0; i < K.rows(); ++i) {
        for (std::size_t j = 0; j < K.cols(); ++j) {
            EXPECT_NEAR(K(i, j), K(j, i), 1e-14);
        }
    }
}

TEST(FemLoad2D, integrates_constant_source) {
    const auto mesh_r = mesh2d_rectangular(0.0, 0.0, 1.0, 1.0, 2, 2);
    ASSERT_TRUE(mesh_r.has_value());
    const Mesh2D& mesh = *mesh_r;
    const auto load_r = assemble_load_2d(mesh, [](double, double) { return 1.0; });
    ASSERT_TRUE(load_r.has_value());
    const ColMatrix<double>& load = *load_r;
    double total = 0.0;
    for (std::size_t i = 0; i < load.rows(); ++i) {
        total += load(i, 0);
    }
    EXPECT_NEAR(total, 1.0, 1e-12);
}

TEST(FemSolve2D, poisson_sin_pi) {
    constexpr std::size_t nx = 20;
    constexpr std::size_t ny = 20;
    const auto mesh_r = mesh2d_rectangular(0.0, 0.0, 1.0, 1.0, nx, ny);
    ASSERT_TRUE(mesh_r.has_value());
    const Mesh2D& mesh = *mesh_r;
    auto K_r = assemble_stiffness_2d(mesh);
    ASSERT_TRUE(K_r.has_value());
    ColMatrix<double> K = *K_r;
    auto f_r = assemble_load_2d(mesh, [](double x, double y) {
        return 2.0 * M_PI * M_PI * std::sin(M_PI * x) * std::sin(M_PI * y);
    });
    ASSERT_TRUE(f_r.has_value());
    ColMatrix<double> f = *f_r;

    const auto boundary = rectangular_boundary_nodes(nx, ny);
    std::vector<double> boundary_values(boundary.size(), 0.0);
    const auto dir_r = apply_dirichlet(K, f, boundary, boundary_values);
    ASSERT_TRUE(dir_r.has_value());

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
    const auto mesh_r = mesh2d_rectangular(0.0, 0.0, 1.0, 1.0, nx, ny);
    ASSERT_TRUE(mesh_r.has_value());
    const Mesh2D& mesh = *mesh_r;
    auto K_r = assemble_stiffness_2d(mesh);
    ASSERT_TRUE(K_r.has_value());
    ColMatrix<double> K = *K_r;
    auto f_r = assemble_load_2d(mesh, [](double, double) { return 0.0; });
    ASSERT_TRUE(f_r.has_value());
    ColMatrix<double> f = *f_r;

    const auto boundary = rectangular_boundary_nodes(nx, ny);
    std::vector<double> boundary_values;
    boundary_values.reserve(boundary.size());
    for (std::size_t idx : boundary) {
        const double x = mesh.nodes[idx][0];
        const double y = mesh.nodes[idx][1];
        boundary_values.push_back(x + y);
    }
    const auto dir_r = apply_dirichlet(K, f, boundary, boundary_values);
    ASSERT_TRUE(dir_r.has_value());

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
    const auto mesh_r = mesh3d_box(0.0, 0.0, 0.0, 2.0, 1.0, 1.0, 2, 1, 1);
    ASSERT_TRUE(mesh_r.has_value());
    const Mesh3D& mesh = *mesh_r;
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
    {
        const auto r = mesh3d_box(1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1, 1, 1);
        ASSERT_FALSE(r.has_value());
    }
    {
        const auto r = mesh3d_box(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1, 1, 1);
        ASSERT_FALSE(r.has_value());
    }
    {
        const auto r = mesh3d_box(0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1, 1, 1);
        ASSERT_FALSE(r.has_value());
    }
    {
        const auto r = mesh3d_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0, 1, 1);
        ASSERT_FALSE(r.has_value());
    }
    {
        const auto r = mesh3d_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1, 0, 1);
        ASSERT_FALSE(r.has_value());
    }
    {
        const auto r = mesh3d_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1, 1, 0);
        ASSERT_FALSE(r.has_value());
    }
}

TEST(FemStiffness3D, single_tetrahedron_pattern) {
    const Mesh3D mesh = single_corner_tetrahedron();
    const auto K_r = assemble_stiffness_3d(mesh);
    ASSERT_TRUE(K_r.has_value());
    const ColMatrix<double>& K = *K_r;
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
    const auto mesh_r = mesh3d_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2, 2, 2);
    ASSERT_TRUE(mesh_r.has_value());
    const Mesh3D& mesh = *mesh_r;
    const auto K_r = assemble_stiffness_3d(mesh);
    ASSERT_TRUE(K_r.has_value());
    const ColMatrix<double>& K = *K_r;
    for (std::size_t i = 0; i < K.rows(); ++i) {
        for (std::size_t j = 0; j < K.cols(); ++j) {
            EXPECT_NEAR(K(i, j), K(j, i), 1e-14);
        }
    }
}

TEST(FemStiffness3D, row_sums_vanish) {
    const auto mesh_r = mesh3d_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1, 1, 1);
    ASSERT_TRUE(mesh_r.has_value());
    const Mesh3D& mesh = *mesh_r;
    const auto K_r = assemble_stiffness_3d(mesh);
    ASSERT_TRUE(K_r.has_value());
    const ColMatrix<double>& K = *K_r;
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
    const auto mesh_r = mesh3d_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, nx, ny, nz);
    ASSERT_TRUE(mesh_r.has_value());
    const Mesh3D& mesh = *mesh_r;
    auto K_r = assemble_stiffness_3d(mesh);
    ASSERT_TRUE(K_r.has_value());
    ColMatrix<double> K = *K_r;
    auto f_r = assemble_load_3d(mesh, [](double, double, double) { return 0.0; });
    ASSERT_TRUE(f_r.has_value());
    ColMatrix<double> f = *f_r;

    const auto boundary = box_boundary_nodes(nx, ny, nz);
    std::vector<double> boundary_values;
    boundary_values.reserve(boundary.size());
    for (std::size_t idx : boundary) {
        const double x = mesh.nodes[idx][0];
        const double y = mesh.nodes[idx][1];
        const double z = mesh.nodes[idx][2];
        boundary_values.push_back(x + y + z);
    }
    const auto dir_r = apply_dirichlet(K, f, boundary, boundary_values);
    ASSERT_TRUE(dir_r.has_value());

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
    const auto mesh_r = mesh3d_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2, 2, 2);
    ASSERT_TRUE(mesh_r.has_value());
    const Mesh3D& mesh = *mesh_r;
    const auto load_r = assemble_load_3d(mesh, [](double, double, double) { return 1.0; });
    ASSERT_TRUE(load_r.has_value());
    const ColMatrix<double>& load = *load_r;
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
    const auto mesh_r = mesh3d_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, nx, ny, nz);
    ASSERT_TRUE(mesh_r.has_value());
    const Mesh3D& mesh = *mesh_r;
    auto K_r = assemble_stiffness_3d(mesh);
    ASSERT_TRUE(K_r.has_value());
    ColMatrix<double> K = *K_r;
    auto f_r = assemble_load_3d(mesh, [](double x, double y, double z) {
        return 3.0 * M_PI * M_PI * std::sin(M_PI * x) * std::sin(M_PI * y) * std::sin(M_PI * z);
    });
    ASSERT_TRUE(f_r.has_value());
    ColMatrix<double> f = *f_r;

    const auto boundary = box_boundary_nodes(nx, ny, nz);
    std::vector<double> boundary_values(boundary.size(), 0.0);
    const auto dir_r = apply_dirichlet(K, f, boundary, boundary_values);
    ASSERT_TRUE(dir_r.has_value());

    const auto u = solve_fem_3d(K, f);
    ASSERT_TRUE(u.has_value());

    for (std::size_t i = 0; i < mesh.nodes.size(); ++i) {
        const double x = mesh.nodes[i][0];
        const double y = mesh.nodes[i][1];
        const double z = mesh.nodes[i][2];
        EXPECT_NEAR(
            (*u)(i, 0),
            std::sin(M_PI * x) * std::sin(M_PI * y) * std::sin(M_PI * z),
            0.35);
    }
}

TEST(FemLagrangeBasis, nonzero_degree_rejected) {
    LagrangeBasis basis;
    basis.degree = 2;
    EXPECT_FALSE(basis.evaluate(0.5).has_value());
    EXPECT_FALSE(basis.derivative(0.25).has_value());
}

TEST(FemMesh1D, single_element) {
    const auto mesh_r = mesh1d(0.0, 1.0, 1);
    ASSERT_TRUE(mesh_r.has_value());
    EXPECT_EQ(mesh_r->nodes.size(), 2u);
    EXPECT_EQ(mesh_r->connectivity.size(), 1u);
    auto K = assemble_stiffness_1d(*mesh_r);
    ASSERT_TRUE(K.has_value());
    EXPECT_EQ(K->rows(), 2u);
}

TEST(FemMesh2D, single_cell) {
    const auto mesh_r = mesh2d_rectangular(0.0, 0.0, 1.0, 1.0, 1, 1);
    ASSERT_TRUE(mesh_r.has_value());
    EXPECT_EQ(mesh_r->nodes.size(), 4u);
    EXPECT_EQ(mesh_r->triangles.size(), 2u);
    auto K = assemble_stiffness_2d(*mesh_r);
    ASSERT_TRUE(K.has_value());
}

TEST(FemMesh3D, single_cell) {
    const auto mesh_r = mesh3d_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1, 1, 1);
    ASSERT_TRUE(mesh_r.has_value());
    EXPECT_EQ(mesh_r->nodes.size(), 8u);
    EXPECT_FALSE(mesh_r->tetrahedra.empty());
    auto K = assemble_stiffness_3d(*mesh_r);
    ASSERT_TRUE(K.has_value());
}

TEST(FemStiffness1D, empty_mesh_errors) {
    Mesh1D empty;
    EXPECT_FALSE(assemble_stiffness_1d(empty).has_value());
    Mesh1D nodes_only;
    nodes_only.nodes = {0.0, 1.0};
    EXPECT_FALSE(assemble_stiffness_1d(nodes_only).has_value());
}

TEST(FemStiffness1D, invalid_connectivity) {
    Mesh1D mesh;
    mesh.nodes = {0.0, 1.0};
    mesh.connectivity = {{{0, 5}}};
    EXPECT_FALSE(assemble_stiffness_1d(mesh).has_value());
}

TEST(FemStiffness1D, nonpositive_length) {
    Mesh1D mesh;
    mesh.nodes = {0.0, 1.0};
    mesh.connectivity = {{{1, 0}}};
    EXPECT_FALSE(assemble_stiffness_1d(mesh).has_value());
}

TEST(FemLoad1D, empty_mesh_errors) {
    Mesh1D empty;
    EXPECT_FALSE(assemble_load_1d(empty, [](double) { return 1.0; }).has_value());
}

TEST(FemLoad1D, nonpositive_length) {
    Mesh1D mesh;
    mesh.nodes = {0.0, 1.0};
    mesh.connectivity = {{{1, 0}}};
    EXPECT_FALSE(assemble_load_1d(mesh, [](double) { return 1.0; }).has_value());
}

TEST(FemStiffness2D, empty_mesh_errors) {
    Mesh2D empty;
    EXPECT_FALSE(assemble_stiffness_2d(empty).has_value());
    Mesh2D nodes_only;
    nodes_only.nodes = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};
    EXPECT_FALSE(assemble_stiffness_2d(nodes_only).has_value());
}

TEST(FemStiffness2D, invalid_connectivity) {
    Mesh2D mesh;
    mesh.nodes = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};
    mesh.triangles = {{{0, 1, 9}}};
    EXPECT_FALSE(assemble_stiffness_2d(mesh).has_value());
}

TEST(FemStiffness2D, degenerate_triangle) {
    Mesh2D mesh;
    mesh.nodes = {{0.0, 0.0}, {1.0, 0.0}, {2.0, 0.0}};
    mesh.triangles = {{{0, 1, 2}}};
    EXPECT_FALSE(assemble_stiffness_2d(mesh).has_value());
}

TEST(FemLoad2D, empty_mesh_errors) {
    Mesh2D empty;
    EXPECT_FALSE(assemble_load_2d(empty, [](double, double) { return 1.0; }).has_value());
}

TEST(FemLoad2D, degenerate_triangle) {
    Mesh2D mesh;
    mesh.nodes = {{0.0, 0.0}, {1.0, 0.0}, {2.0, 0.0}};
    mesh.triangles = {{{0, 1, 2}}};
    EXPECT_FALSE(assemble_load_2d(mesh, [](double, double) { return 1.0; }).has_value());
}

TEST(FemStiffness3D, empty_mesh_errors) {
    Mesh3D empty;
    EXPECT_FALSE(assemble_stiffness_3d(empty).has_value());
    Mesh3D nodes_only;
    nodes_only.nodes = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    EXPECT_FALSE(assemble_stiffness_3d(nodes_only).has_value());
}

TEST(FemStiffness3D, invalid_connectivity) {
    Mesh3D mesh;
    mesh.nodes = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    mesh.tetrahedra = {{{0, 1, 2, 9}}};
    EXPECT_FALSE(assemble_stiffness_3d(mesh).has_value());
}

TEST(FemStiffness3D, degenerate_tetrahedron) {
    Mesh3D mesh;
    mesh.nodes = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0.5, 0.5, 0}};
    mesh.tetrahedra = {{{0, 1, 2, 3}}};
    EXPECT_FALSE(assemble_stiffness_3d(mesh).has_value());
}

TEST(FemLoad3D, empty_mesh_errors) {
    Mesh3D empty;
    EXPECT_FALSE(assemble_load_3d(empty, [](double, double, double) { return 1.0; }).has_value());
}

TEST(FemLoad3D, invalid_connectivity) {
    Mesh3D mesh;
    mesh.nodes = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    mesh.tetrahedra = {{{0, 1, 2, 9}}};
    EXPECT_FALSE(assemble_load_3d(mesh, [](double, double, double) { return 1.0; }).has_value());
}

TEST(FemLoad3D, degenerate_tetrahedron) {
    Mesh3D mesh;
    mesh.nodes = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0.5, 0.5, 0}};
    mesh.tetrahedra = {{{0, 1, 2, 3}}};
    EXPECT_FALSE(assemble_load_3d(mesh, [](double, double, double) { return 1.0; }).has_value());
}

TEST(FemDirichlet, rejects_nonsquare_K) {
    ColMatrix<double> K(2, 3, 0.0);
    ColMatrix<double> f(2, 1, 0.0);
    EXPECT_FALSE(apply_dirichlet(K, f, {0}, {0.0}).has_value());
}

TEST(FemDirichlet, rejects_f_shape) {
    ColMatrix<double> K(2, 2, 0.0);
    ColMatrix<double> f(2, 2, 0.0);
    EXPECT_FALSE(apply_dirichlet(K, f, {0}, {1.0}).has_value());
}

TEST(FemDirichlet, rejects_size_mismatch) {
    ColMatrix<double> K(2, 2, 0.0);
    ColMatrix<double> f(2, 1, 0.0);
    EXPECT_FALSE(apply_dirichlet(K, f, {0, 1}, {1.0}).has_value());
}

TEST(FemDirichlet, rejects_out_of_range) {
    ColMatrix<double> K(2, 2, 0.0);
    ColMatrix<double> f(2, 1, 0.0);
    EXPECT_FALSE(apply_dirichlet(K, f, {5}, {1.0}).has_value());
}

TEST(FemSolve, rejects_nonsquare) {
    ColMatrix<double> K(2, 3, 0.0);
    ColMatrix<double> f(2, 1, 0.0);
    EXPECT_FALSE(solve_fem(K, f).has_value());
}

TEST(FemSolve, rejects_f_mismatch) {
    ColMatrix<double> K(2, 2, 0.0);
    K(0, 0) = 1.0;
    K(1, 1) = 1.0;
    ColMatrix<double> f(3, 1, 0.0);
    EXPECT_FALSE(solve_fem(K, f).has_value());
}

TEST(FemSolve, empty_system_errors) {
    ColMatrix<double> K(0, 0);
    ColMatrix<double> f(0, 1);
    EXPECT_FALSE(solve_fem(K, f).has_value());
}

TEST(FemSolve, singular_tridiagonal) {
    ColMatrix<double> K(2, 2, 0.0);
    ColMatrix<double> f(2, 1, 1.0);
    EXPECT_FALSE(solve_fem(K, f).has_value());
}

TEST(FemSolve, singular_one_by_one) {
    ColMatrix<double> K(1, 1, 0.0);
    ColMatrix<double> f(1, 1, 1.0);
    EXPECT_FALSE(solve_fem(K, f).has_value());
}

TEST(FemSolve3D, rejects_mismatch) {
    ColMatrix<double> K(2, 2, 0.0);
    ColMatrix<double> f(1, 1, 0.0);
    EXPECT_FALSE(solve_fem_3d(K, f).has_value());
}

TEST(FemSolve, singular_tridiagonal_trailing_diag) {
    ColMatrix<double> K(3, 3, 0.0);
    K(0, 0) = 1.0;
    K(1, 1) = 1.0;
    ColMatrix<double> f(3, 1, 1.0);
    EXPECT_FALSE(solve_fem(K, f).has_value());
}

TEST(FemSolve, singular_tridiagonal_mid_pivot) {
    ColMatrix<double> K(3, 3, 0.0);
    K(0, 0) = 1.0;
    K(0, 1) = 1.0;
    K(1, 0) = 1.0;
    K(1, 1) = 1.0;
    K(2, 2) = 1.0;
    ColMatrix<double> f(3, 1, 1.0);
    EXPECT_FALSE(solve_fem(K, f).has_value());
}

TEST(FemSolve, singular_dense_nondiagonal) {
    ColMatrix<double> K(3, 3, 1.0);
    ColMatrix<double> f(3, 1, 1.0);
    EXPECT_FALSE(solve_fem(K, f).has_value());
}

TEST(FemDirichlet, rejects_later_node_out_of_range) {
    ColMatrix<double> K(2, 2, 0.0);
    ColMatrix<double> f(2, 1, 0.0);
    EXPECT_FALSE(apply_dirichlet(K, f, {0, 4}, {1.0, 2.0}).has_value());
}

TEST(FemStiffness1D, first_node_out_of_range) {
    Mesh1D mesh;
    mesh.nodes = {0.0, 1.0};
    mesh.connectivity = {{{9, 1}}};
    EXPECT_FALSE(assemble_stiffness_1d(mesh).has_value());
}

TEST(FemStiffness2D, first_node_out_of_range) {
    Mesh2D mesh;
    mesh.nodes = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};
    mesh.triangles = {{{8, 1, 2}}};
    EXPECT_FALSE(assemble_stiffness_2d(mesh).has_value());
}

TEST(FemStiffness3D, first_node_out_of_range) {
    Mesh3D mesh;
    mesh.nodes = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    mesh.tetrahedra = {{{7, 1, 2, 3}}};
    EXPECT_FALSE(assemble_stiffness_3d(mesh).has_value());
}

TEST(FemLagrangeBasis, p2_degree_rejected) {
    LagrangeBasis basis;
    basis.degree = 2;
    EXPECT_FALSE(basis.evaluate(0.0).has_value());
    EXPECT_FALSE(basis.derivative(1.0).has_value());
}

TEST(FemLoad, remaining_empty_connectivity_errors) {
    Mesh1D mesh1;
    mesh1.nodes = {0.0, 1.0};
    EXPECT_FALSE(assemble_load_1d(mesh1, [](double) { return 1.0; }).has_value());

    Mesh2D mesh2;
    mesh2.nodes = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};
    EXPECT_FALSE(assemble_load_2d(mesh2, [](double, double) { return 1.0; }).has_value());

    Mesh3D mesh3;
    mesh3.nodes = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    EXPECT_FALSE(assemble_load_3d(mesh3, [](double, double, double) { return 1.0; }).has_value());
}

TEST(FemDirichlet, rejects_f_row_count_mismatch) {
    ColMatrix<double> K(2, 2, 0.0);
    ColMatrix<double> f_short(1, 1, 0.0);
    const std::vector<std::size_t> nodes{0};
    const std::vector<double> values{1.0};
    const auto r = apply_dirichlet(K, f_short, nodes, values);
    ASSERT_FALSE(r.has_value());
    EXPECT_TRUE(std::holds_alternative<DomainError>(r.error()));
}

TEST(FemSolve, non_column_rhs_and_singular_zero_matrix) {
    ColMatrix<double> K(2, 2, 0.0);
    K(0, 0) = 1.0;
    K(1, 1) = 1.0;
    ColMatrix<double> f_wide(2, 2, 1.0);
    EXPECT_FALSE(solve_fem(K, f_wide).has_value());

    ColMatrix<double> K_zero(2, 2, 0.0);
    ColMatrix<double> f(2, 1, 1.0);
    const auto u = solve_fem(K_zero, f);
    ASSERT_FALSE(u.has_value());
    EXPECT_TRUE(std::holds_alternative<SingularMatrix>(u.error()));
}

TEST(FemDirichlet, empty_nodes_noop) {
    ColMatrix<double> K(2, 2, 0.0);
    K(0, 0) = 1.0;
    K(1, 1) = 1.0;
    ColMatrix<double> f(2, 1, 3.0);
    const std::vector<std::size_t> empty_nodes;
    const std::vector<double> empty_values;
    const auto r = apply_dirichlet(K, f, empty_nodes, empty_values);
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(K(0, 0), 1.0, 1e-14);
    EXPECT_NEAR(f(0, 0), 3.0, 1e-14);
}

TEST(FemSolve, identity_one_by_one) {
    ColMatrix<double> K(1, 1, 1.0);
    ColMatrix<double> f(1, 1, 2.0);
    const auto u = solve_fem(K, f);
    ASSERT_TRUE(u.has_value());
    EXPECT_NEAR((*u)(0, 0), 2.0, 1e-14);
}

TEST(FemSolve3D, nonsquare_and_wide_rhs) {
    ColMatrix<double> K(2, 3, 0.0);
    ColMatrix<double> f_wide(2, 2, 1.0);
    EXPECT_FALSE(solve_fem_3d(K, f_wide).has_value());
    ColMatrix<double> Ksq(2, 2, 0.0);
    ColMatrix<double> f_col(3, 1, 0.0);
    EXPECT_FALSE(solve_fem_3d(Ksq, f_col).has_value());
}

TEST(FemLagrangeBasis, degree_three_rejected) {
    LagrangeBasis basis;
    basis.degree = 3;
    EXPECT_FALSE(basis.evaluate(0.5).has_value());
    EXPECT_FALSE(basis.derivative(0.5).has_value());
}

TEST(FemStiffness2D, middle_node_out_of_range) {
    Mesh2D mesh;
    mesh.nodes = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};
    mesh.triangles = {{{0, 9, 2}}};
    EXPECT_FALSE(assemble_stiffness_2d(mesh).has_value());
}

TEST(FemStiffness3D, middle_node_out_of_range) {
    Mesh3D mesh;
    mesh.nodes = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    mesh.tetrahedra = {{{0, 8, 2, 3}}};
    EXPECT_FALSE(assemble_stiffness_3d(mesh).has_value());
}

TEST(FemSolve3D, empty_and_singular_zero) {
    ColMatrix<double> K_empty(0, 0);
    ColMatrix<double> f_empty(0, 1);
    EXPECT_FALSE(solve_fem_3d(K_empty, f_empty).has_value());

    ColMatrix<double> K_zero(2, 2, 0.0);
    ColMatrix<double> f(2, 1, 1.0);
    const auto u = solve_fem_3d(K_zero, f);
    ASSERT_FALSE(u.has_value());
    EXPECT_TRUE(std::holds_alternative<SingularMatrix>(u.error()));
}

TEST(FemStiffness3D, third_node_out_of_range) {
    Mesh3D mesh;
    mesh.nodes = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    mesh.tetrahedra = {{{0, 1, 8, 3}}};
    EXPECT_FALSE(assemble_stiffness_3d(mesh).has_value());
    EXPECT_FALSE(assemble_load_3d(mesh, [](double, double, double) { return 1.0; }).has_value());
}

TEST(FemAssemble, too_few_nodes_with_elements) {
    Mesh2D mesh2;
    mesh2.nodes = {{0.0, 0.0}, {1.0, 0.0}};
    mesh2.triangles = {{{0, 1, 0}}};
    EXPECT_FALSE(assemble_stiffness_2d(mesh2).has_value());
    EXPECT_FALSE(assemble_load_2d(mesh2, [](double, double) { return 1.0; }).has_value());

    Mesh3D mesh3;
    mesh3.nodes = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    mesh3.tetrahedra = {{{0, 1, 2, 0}}};
    EXPECT_FALSE(assemble_stiffness_3d(mesh3).has_value());
    EXPECT_FALSE(assemble_load_3d(mesh3, [](double, double, double) { return 1.0; }).has_value());
}

TEST(FemLagrangeBasis, degree_zero_rejected) {
    LagrangeBasis basis;
    basis.degree = 0;
    EXPECT_FALSE(basis.evaluate(0.0).has_value());
    EXPECT_FALSE(basis.derivative(0.0).has_value());
}

TEST(FemAssemble, empty_meshes_reject_stiffness_and_load) {
    Mesh1D empty1;
    EXPECT_FALSE(assemble_stiffness_1d(empty1).has_value());
    EXPECT_FALSE(assemble_load_1d(empty1, [](double) { return 1.0; }).has_value());

    Mesh1D one_node;
    one_node.nodes = {0.0};
    EXPECT_FALSE(assemble_stiffness_1d(one_node).has_value());
    EXPECT_FALSE(assemble_load_1d(one_node, [](double) { return 1.0; }).has_value());

    Mesh2D empty2;
    EXPECT_FALSE(assemble_stiffness_2d(empty2).has_value());
    EXPECT_FALSE(assemble_load_2d(empty2, [](double, double) { return 1.0; }).has_value());

    Mesh3D empty3;
    EXPECT_FALSE(assemble_stiffness_3d(empty3).has_value());
    EXPECT_FALSE(assemble_load_3d(empty3, [](double, double, double) { return 1.0; }).has_value());
}

TEST(FemLagrangeBasis, degree_two_unused_xi_rejected) {
    LagrangeBasis basis;
    basis.degree = 2;
    EXPECT_FALSE(basis.evaluate(-0.25).has_value());
    EXPECT_FALSE(basis.evaluate(1.25).has_value());
    EXPECT_FALSE(basis.derivative(-0.25).has_value());
    EXPECT_FALSE(basis.derivative(1.25).has_value());
}

TEST(FemMesh2D, equal_axis_bounds_rejected) {
    EXPECT_FALSE(mesh2d_rectangular(1.0, 0.0, 1.0, 1.0, 2, 2).has_value());
    EXPECT_FALSE(mesh2d_rectangular(0.0, 2.0, 1.0, 2.0, 2, 2).has_value());
}

TEST(FemMesh3D, equal_axis_bounds_rejected) {
    EXPECT_FALSE(mesh3d_box(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1, 1, 1).has_value());
    EXPECT_FALSE(mesh3d_box(0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1, 1, 1).has_value());
    EXPECT_FALSE(mesh3d_box(0.0, 0.0, 2.0, 1.0, 1.0, 2.0, 1, 1, 1).has_value());
}

TEST(FemLagrangeBasis, negative_degree_rejected) {
    LagrangeBasis basis;
    basis.degree = -1;
    EXPECT_FALSE(basis.evaluate(0.5).has_value());
    EXPECT_FALSE(basis.derivative(0.5).has_value());
}

TEST(FemAssemble, clockwise_triangle_and_inverted_tet) {
    Mesh2D cw;
    cw.nodes = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}};
    cw.triangles = {{{0, 1, 2}}};
    const auto K2 = assemble_stiffness_2d(cw);
    ASSERT_TRUE(K2.has_value());
    EXPECT_TRUE(std::isfinite((*K2)(0, 0)));
    EXPECT_NEAR((*K2)(0, 1), (*K2)(1, 0), 1e-12);
    const auto f2 = assemble_load_2d(cw, [](double, double) { return 2.0; });
    ASSERT_TRUE(f2.has_value());
    EXPECT_NEAR((*f2)(0, 0) + (*f2)(1, 0) + (*f2)(2, 0), 1.0, 1e-12);

    Mesh3D inv;
    inv.nodes = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    inv.tetrahedra = {{{0, 2, 1, 3}}};
    const auto K3 = assemble_stiffness_3d(inv);
    ASSERT_TRUE(K3.has_value());
    EXPECT_GT((*K3)(0, 0), 0.0);
    const auto f3 = assemble_load_3d(inv, [](double, double, double) { return 1.0; });
    ASSERT_TRUE(f3.has_value());
    double vol = 0.0;
    for (std::size_t i = 0; i < f3->rows(); ++i) {
        vol += (*f3)(i, 0);
    }
    EXPECT_NEAR(vol, 1.0 / 6.0, 1e-10);
}

TEST(FemStiffness1D, second_element_invalid_after_valid) {
    Mesh1D mesh;
    mesh.nodes = {0.0, 1.0, 2.0};
    mesh.connectivity = {{{0, 1}, {0, 9}}};
    EXPECT_FALSE(assemble_stiffness_1d(mesh).has_value());
}

TEST(FemLoad1D, second_element_nonpositive_after_valid) {
    Mesh1D mesh;
    mesh.nodes = {0.0, 1.0, 2.0};
    mesh.connectivity = {{{0, 1}, {2, 1}}};
    EXPECT_FALSE(assemble_load_1d(mesh, [](double) { return 1.0; }).has_value());
}

TEST(FemDirichlet, one_by_one_sets_value) {
    ColMatrix<double> K(1, 1, 4.0);
    ColMatrix<double> f(1, 1, 7.0);
    const std::vector<std::size_t> nodes{0};
    const std::vector<double> values{3.0};
    const auto r = apply_dirichlet(K, f, nodes, values);
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(K(0, 0), 1.0, 1e-14);
    EXPECT_NEAR(f(0, 0), 3.0, 1e-14);
}

TEST(FemAssemble, two_cell_3d_constant_load) {
    const auto mesh_r = mesh3d_box(0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 1, 1, 2);
    ASSERT_TRUE(mesh_r.has_value());
    const auto load = assemble_load_3d(*mesh_r, [](double, double, double) { return 1.0; });
    ASSERT_TRUE(load.has_value());
    double total = 0.0;
    for (std::size_t i = 0; i < load->rows(); ++i) {
        total += (*load)(i, 0);
    }
    EXPECT_NEAR(total, 2.0, 1e-10);
}

TEST(FemStiffness2D, second_triangle_middle_node_out_of_range) {
    Mesh2D mesh;
    mesh.nodes = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};
    mesh.triangles = {{{0, 1, 2}, {0, 8, 2}}};
    EXPECT_FALSE(assemble_stiffness_2d(mesh).has_value());
}
