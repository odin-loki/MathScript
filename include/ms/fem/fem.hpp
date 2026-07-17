#pragma once

#include "ms/core/matrix.hpp"
#include "ms/error/error_types.hpp"
#include <array>
#include <cstddef>
#include <functional>
#include <vector>

namespace ms {
namespace fem {

struct Mesh1D {
    std::vector<double> nodes;
    /// Each element connects two node indices [left, right].
    std::vector<std::array<std::size_t, 2>> connectivity;
};

/// Uniform 1D mesh on [a, b] with @p n_elements elements (n_elements + 1 nodes).
Mesh1D mesh1d(double a, double b, std::size_t n_elements);

/// P1 Lagrange shape functions on the reference element [0, 1].
struct LagrangeBasis {
    int degree = 1;

    /// Values of N0(xi), N1(xi) at reference coordinate xi in [0, 1].
    std::array<double, 2> evaluate(double xi) const;

    /// Derivatives dN/dxi at xi in [0, 1].
    std::array<double, 2> derivative(double xi) const;
};

/// Return the P1 Lagrange basis on the reference element [0, 1].
LagrangeBasis lagrange_basis();

/// Assemble the global stiffness matrix for -u'' on a 1D P1 mesh.
ColMatrix<double> assemble_stiffness_1d(const Mesh1D& mesh);

/// Assemble the global load vector for integral(f * phi_i) dx (2-point Gauss).
ColMatrix<double> assemble_load_1d(
    const Mesh1D& mesh,
    const std::function<double(double)>& f);

/// Apply Dirichlet boundary conditions by modifying @p K and @p f in place.
void apply_dirichlet(
    ColMatrix<double>& K,
    ColMatrix<double>& f,
    const std::vector<std::size_t>& node_indices,
    const std::vector<double>& values);

/// Solve the FEM linear system K u = f using ms::linalg::solve.
Result<ColMatrix<double>> solve_fem(
    const ColMatrix<double>& K,
    const ColMatrix<double>& f);

} // namespace fem
} // namespace ms
