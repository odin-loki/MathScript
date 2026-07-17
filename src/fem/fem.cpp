#include "ms/fem/fem.hpp"

#include "ms/core/operations.hpp"
#include <cmath>
#include <stdexcept>

namespace ms {
namespace fem {

namespace {

constexpr double k_inv_sqrt3 = 0.5773502691896257645091488;
constexpr double k_gauss_xi[2] = {-k_inv_sqrt3, k_inv_sqrt3};
constexpr double k_gauss_w[2] = {1.0, 1.0};

void validate_dirichlet(
    const ColMatrix<double>& K,
    const ColMatrix<double>& f,
    const std::vector<std::size_t>& node_indices,
    const std::vector<double>& values) {
    if (K.rows() != K.cols()) {
        throw std::invalid_argument("assemble_stiffness: K must be square");
    }
    if (f.rows() != K.rows() || f.cols() != 1) {
        throw std::invalid_argument("apply_dirichlet: f must be a column vector matching K");
    }
    if (node_indices.size() != values.size()) {
        throw std::invalid_argument("apply_dirichlet: node_indices and values size mismatch");
    }
    for (std::size_t idx : node_indices) {
        if (idx >= K.rows()) {
            throw std::invalid_argument("apply_dirichlet: node index out of range");
        }
    }
}

} // namespace

Mesh1D mesh1d(double a, double b, std::size_t n_elements) {
    if (n_elements == 0) {
        throw std::invalid_argument("mesh1d: n_elements must be positive");
    }
    if (!(a < b)) {
        throw std::invalid_argument("mesh1d: require a < b");
    }

    Mesh1D mesh;
    const std::size_t n_nodes = n_elements + 1;
    mesh.nodes.resize(n_nodes);
    mesh.connectivity.resize(n_elements);

    const double h = (b - a) / static_cast<double>(n_elements);
    for (std::size_t i = 0; i < n_nodes; ++i) {
        mesh.nodes[i] = a + static_cast<double>(i) * h;
    }
    for (std::size_t e = 0; e < n_elements; ++e) {
        mesh.connectivity[e] = {e, e + 1};
    }
    return mesh;
}

std::array<double, 2> LagrangeBasis::evaluate(double xi) const {
    if (degree != 1) {
        throw std::invalid_argument("LagrangeBasis::evaluate: only P1 supported");
    }
    return {1.0 - xi, xi};
}

std::array<double, 2> LagrangeBasis::derivative(double xi) const {
    (void)xi;
    if (degree != 1) {
        throw std::invalid_argument("LagrangeBasis::derivative: only P1 supported");
    }
    return {-1.0, 1.0};
}

LagrangeBasis lagrange_basis() {
    return LagrangeBasis{};
}

ColMatrix<double> assemble_stiffness_1d(const Mesh1D& mesh) {
    const std::size_t n_nodes = mesh.nodes.size();
    if (n_nodes < 2 || mesh.connectivity.empty()) {
        throw std::invalid_argument("assemble_stiffness_1d: mesh is too small");
    }

    ColMatrix<double> K(n_nodes, n_nodes, 0.0);
    for (const auto& elem : mesh.connectivity) {
        const std::size_t n0 = elem[0];
        const std::size_t n1 = elem[1];
        if (n1 >= n_nodes || n0 >= n_nodes) {
            throw std::invalid_argument("assemble_stiffness_1d: invalid connectivity");
        }

        const double h = mesh.nodes[n1] - mesh.nodes[n0];
        if (h <= 0.0) {
            throw std::invalid_argument("assemble_stiffness_1d: non-positive element length");
        }

        const double k = 1.0 / h;
        K(n0, n0) += k;
        K(n0, n1) -= k;
        K(n1, n0) -= k;
        K(n1, n1) += k;
    }
    return K;
}

ColMatrix<double> assemble_load_1d(
    const Mesh1D& mesh,
    const std::function<double(double)>& f) {
    const std::size_t n_nodes = mesh.nodes.size();
    if (n_nodes < 2 || mesh.connectivity.empty()) {
        throw std::invalid_argument("assemble_load_1d: mesh is too small");
    }

    ColMatrix<double> load(n_nodes, 1, 0.0);
    const LagrangeBasis basis = lagrange_basis();

    for (const auto& elem : mesh.connectivity) {
        const std::size_t n0 = elem[0];
        const std::size_t n1 = elem[1];
        const double x0 = mesh.nodes[n0];
        const double x1 = mesh.nodes[n1];
        const double h = x1 - x0;
        if (h <= 0.0) {
            throw std::invalid_argument("assemble_load_1d: non-positive element length");
        }

        for (int q = 0; q < 2; ++q) {
            const double xi_ref = 0.5 * (k_gauss_xi[q] + 1.0);
            const double x = x0 + h * xi_ref;
            const double weight = 0.5 * h * k_gauss_w[q];
            const auto N = basis.evaluate(xi_ref);
            load(n0, 0) += weight * f(x) * N[0];
            load(n1, 0) += weight * f(x) * N[1];
        }
    }
    return load;
}

void apply_dirichlet(
    ColMatrix<double>& K,
    ColMatrix<double>& f,
    const std::vector<std::size_t>& node_indices,
    const std::vector<double>& values) {
    validate_dirichlet(K, f, node_indices, values);

    for (std::size_t k = 0; k < node_indices.size(); ++k) {
        const std::size_t i = node_indices[k];
        const double g = values[k];

        for (std::size_t j = 0; j < K.rows(); ++j) {
            if (j != i) {
                f(j, 0) -= K(j, i) * g;
            }
        }

        for (std::size_t j = 0; j < K.cols(); ++j) {
            K(i, j) = 0.0;
            K(j, i) = 0.0;
        }
        K(i, i) = 1.0;
        f(i, 0) = g;
    }
}

Result<ColMatrix<double>> solve_fem(
    const ColMatrix<double>& K,
    const ColMatrix<double>& f) {
    return solve(K, f);
}

} // namespace fem
} // namespace ms
