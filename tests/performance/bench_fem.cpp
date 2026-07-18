// MathScript FEM benchmark suite

#include <benchmark/benchmark.h>
#include <cmath>
#include <vector>

#include "ms/fem/fem.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ms;
using namespace ms::fem;

namespace {

constexpr std::size_t kAssembleElements = 1000;
constexpr std::size_t kAssembleNx = 32;
constexpr std::size_t kAssembleNy = 32;
constexpr std::size_t kAssemble3DNx = 4;
constexpr std::size_t kAssemble3DNy = 4;
constexpr std::size_t kAssemble3DNz = 4;

Mesh1D make_mesh(std::size_t n_elements) {
    return mesh1d(0.0, 1.0, n_elements);
}

Mesh2D make_mesh2d(std::size_t nx, std::size_t ny) {
    return mesh2d_rectangular(0.0, 0.0, 1.0, 1.0, nx, ny);
}

Mesh3D make_mesh3d(std::size_t nx, std::size_t ny, std::size_t nz) {
    return mesh3d_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, nx, ny, nz);
}

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

static void BM_FemAssemble1D(benchmark::State& state) {
    const auto mesh = make_mesh(kAssembleElements);
    for (auto _ : state) {
        auto K = assemble_stiffness_1d(mesh);
        benchmark::DoNotOptimize(K.data());
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(kAssembleElements));
}
BENCHMARK(BM_FemAssemble1D);

static void BM_FemSolve1D(benchmark::State& state) {
    const std::size_t n_elements = static_cast<std::size_t>(state.range(0));
    const Mesh1D mesh = make_mesh(n_elements);
    ColMatrix<double> K = assemble_stiffness_1d(mesh);
    ColMatrix<double> f = assemble_load_1d(mesh, [](double x) {
        return M_PI * M_PI * std::sin(M_PI * x);
    });
    apply_dirichlet(K, f, {0, mesh.nodes.size() - 1}, {0.0, 0.0});

    for (auto _ : state) {
        auto u = solve_fem(K, f);
        benchmark::DoNotOptimize(u);
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(mesh.nodes.size()));
}
BENCHMARK(BM_FemSolve1D)->Arg(100)->Arg(500)->Arg(1000);

static void BM_FemAssemble2D(benchmark::State& state) {
    const auto mesh = make_mesh2d(kAssembleNx, kAssembleNy);
    for (auto _ : state) {
        auto K = assemble_stiffness_2d(mesh);
        benchmark::DoNotOptimize(K.data());
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(mesh.triangles.size()));
}
BENCHMARK(BM_FemAssemble2D);

static void BM_FemSolve2D(benchmark::State& state) {
    const std::size_t nx = static_cast<std::size_t>(state.range(0));
    const std::size_t ny = static_cast<std::size_t>(state.range(0));
    const Mesh2D mesh = make_mesh2d(nx, ny);
    ColMatrix<double> K = assemble_stiffness_2d(mesh);
    ColMatrix<double> f = assemble_load_2d(mesh, [](double x, double y) {
        return 2.0 * M_PI * M_PI * std::sin(M_PI * x) * std::sin(M_PI * y);
    });
    const auto boundary = rectangular_boundary_nodes(nx, ny);
    std::vector<double> boundary_values(boundary.size(), 0.0);
    apply_dirichlet(K, f, boundary, boundary_values);

    for (auto _ : state) {
        auto u = solve_fem(K, f);
        benchmark::DoNotOptimize(u);
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(mesh.nodes.size()));
}
BENCHMARK(BM_FemSolve2D)->Arg(16)->Arg(32);

static void BM_FemAssemble3D(benchmark::State& state) {
    const auto mesh = make_mesh3d(kAssemble3DNx, kAssemble3DNy, kAssemble3DNz);
    for (auto _ : state) {
        auto K = assemble_stiffness_3d(mesh);
        benchmark::DoNotOptimize(K.data());
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(mesh.tetrahedra.size()));
}
BENCHMARK(BM_FemAssemble3D);

static void BM_FemSolve3D(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const Mesh3D mesh = make_mesh3d(n, n, n);
    ColMatrix<double> K = assemble_stiffness_3d(mesh);
    ColMatrix<double> f = assemble_load_3d(mesh, [](double x, double y, double z) {
        return 3.0 * M_PI * M_PI * std::sin(M_PI * x) * std::sin(M_PI * y) *
               std::sin(M_PI * z);
    });
    const auto boundary = box_boundary_nodes(n, n, n);
    std::vector<double> boundary_values(boundary.size(), 0.0);
    apply_dirichlet(K, f, boundary, boundary_values);

    for (auto _ : state) {
        auto u = solve_fem_3d(K, f);
        benchmark::DoNotOptimize(u);
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(mesh.nodes.size()));
}
BENCHMARK(BM_FemSolve3D)->Arg(4)->Arg(6);

BENCHMARK_MAIN();
