#include "ms/fem/fem.hpp"

#include "ms/core/operations.hpp"
#include <cmath>
#include <stdexcept>
#include <vector>

namespace ms {
namespace fem {

namespace {

constexpr double k_inv_sqrt3 = 0.5773502691896257645091488;
constexpr double k_gauss_xi[2] = {-k_inv_sqrt3, k_inv_sqrt3};
constexpr double k_gauss_w[2] = {1.0, 1.0};

constexpr double k_tri_quad_xi[3] = {1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0};
constexpr double k_tri_quad_eta[3] = {1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0};

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

constexpr double k_tridiag_singular_tol = 1e-300;

bool is_tridiagonal(const ColMatrix<double>& K) {
    if (K.rows() != K.cols()) {
        return false;
    }
    const std::size_t n = K.rows();
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            if (i + 1 < j || j + 1 < i) {
                if (std::abs(K(i, j)) > 1e-14) {
                    return false;
                }
            }
        }
    }
    return true;
}

/// Thomas algorithm for a tridiagonal system. Returns nullopt on singularity.
Result<ColMatrix<double>> solve_fem_tridiag(
    const ColMatrix<double>& K,
    const ColMatrix<double>& f) {
    const std::size_t n = K.rows();
    if (n == 0) {
        return std::unexpected(SingularMatrix{});
    }
    if (f.rows() != n || f.cols() != 1) {
        return std::unexpected(DimensionMismatch{f.rows(), f.cols(), n, 1});
    }

    std::vector<double> lower(n, 0.0);
    std::vector<double> diag(n);
    std::vector<double> upper(n, 0.0);
    std::vector<double> rhs(n);

    for (std::size_t i = 0; i < n; ++i) {
        diag[i] = K(i, i);
        rhs[i] = f(i, 0);
        if (i > 0) {
            lower[i] = K(i, i - 1);
        }
        if (i + 1 < n) {
            upper[i] = K(i, i + 1);
        }
    }

    for (std::size_t i = 1; i < n; ++i) {
        if (std::abs(diag[i - 1]) < k_tridiag_singular_tol) {
            return std::unexpected(SingularMatrix{});
        }
        const double w = lower[i] / diag[i - 1];
        diag[i] -= w * upper[i - 1];
        rhs[i] -= w * rhs[i - 1];
    }

    if (std::abs(diag[n - 1]) < k_tridiag_singular_tol) {
        return std::unexpected(SingularMatrix{});
    }

    ColMatrix<double> u(n, 1);
    u(n - 1, 0) = rhs[n - 1] / diag[n - 1];
    for (std::size_t i = n - 1; i-- > 0;) {
        u(i, 0) = (rhs[i] - upper[i] * u(i + 1, 0)) / diag[i];
    }
    return u;
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

Mesh2D mesh2d_rectangular(
    double x0, double y0, double x1, double y1, std::size_t nx, std::size_t ny) {
    if (nx == 0 || ny == 0) {
        throw std::invalid_argument("mesh2d_rectangular: nx and ny must be positive");
    }
    if (!(x0 < x1) || !(y0 < y1)) {
        throw std::invalid_argument("mesh2d_rectangular: require x0 < x1 and y0 < y1");
    }

    Mesh2D mesh;
    const std::size_t n_nodes_x = nx + 1;
    const std::size_t n_nodes_y = ny + 1;
    mesh.nodes.resize(n_nodes_x * n_nodes_y);
    mesh.triangles.resize(2 * nx * ny);

    const double hx = (x1 - x0) / static_cast<double>(nx);
    const double hy = (y1 - y0) / static_cast<double>(ny);

    for (std::size_t j = 0; j < n_nodes_y; ++j) {
        for (std::size_t i = 0; i < n_nodes_x; ++i) {
            mesh.nodes[i + j * n_nodes_x] = {
                x0 + static_cast<double>(i) * hx,
                y0 + static_cast<double>(j) * hy};
        }
    }

    auto node_index = [n_nodes_x](std::size_t i, std::size_t j) {
        return i + j * n_nodes_x;
    };

    std::size_t tri = 0;
    for (std::size_t j = 0; j < ny; ++j) {
        for (std::size_t i = 0; i < nx; ++i) {
            const std::size_t n00 = node_index(i, j);
            const std::size_t n10 = node_index(i + 1, j);
            const std::size_t n01 = node_index(i, j + 1);
            const std::size_t n11 = node_index(i + 1, j + 1);
            mesh.triangles[tri++] = {n00, n10, n01};
            mesh.triangles[tri++] = {n10, n11, n01};
        }
    }
    return mesh;
}

Mesh3D mesh3d_box(
    double x0,
    double y0,
    double z0,
    double x1,
    double y1,
    double z1,
    std::size_t nx,
    std::size_t ny,
    std::size_t nz) {
    if (nx == 0 || ny == 0 || nz == 0) {
        throw std::invalid_argument("mesh3d_box: nx, ny, and nz must be positive");
    }
    if (!(x0 < x1) || !(y0 < y1) || !(z0 < z1)) {
        throw std::invalid_argument("mesh3d_box: require x0 < x1, y0 < y1, and z0 < z1");
    }

    Mesh3D mesh;
    const std::size_t n_nodes_x = nx + 1;
    const std::size_t n_nodes_y = ny + 1;
    const std::size_t n_nodes_z = nz + 1;
    const std::size_t n_nodes_xy = n_nodes_x * n_nodes_y;
    mesh.nodes.resize(n_nodes_x * n_nodes_y * n_nodes_z);
    mesh.tetrahedra.resize(6 * nx * ny * nz);

    const double hx = (x1 - x0) / static_cast<double>(nx);
    const double hy = (y1 - y0) / static_cast<double>(ny);
    const double hz = (z1 - z0) / static_cast<double>(nz);

    for (std::size_t k = 0; k < n_nodes_z; ++k) {
        for (std::size_t j = 0; j < n_nodes_y; ++j) {
            for (std::size_t i = 0; i < n_nodes_x; ++i) {
                mesh.nodes[i + j * n_nodes_x + k * n_nodes_xy] = {
                    x0 + static_cast<double>(i) * hx,
                    y0 + static_cast<double>(j) * hy,
                    z0 + static_cast<double>(k) * hz};
            }
        }
    }

    auto node_index = [n_nodes_x, n_nodes_xy](std::size_t i, std::size_t j, std::size_t k) {
        return i + j * n_nodes_x + k * n_nodes_xy;
    };

    std::size_t tet = 0;
    for (std::size_t k = 0; k < nz; ++k) {
        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t i = 0; i < nx; ++i) {
                const std::size_t v0 = node_index(i, j, k);
                const std::size_t v1 = node_index(i + 1, j, k);
                const std::size_t v2 = node_index(i + 1, j + 1, k);
                const std::size_t v3 = node_index(i, j + 1, k);
                const std::size_t v4 = node_index(i, j, k + 1);
                const std::size_t v5 = node_index(i + 1, j, k + 1);
                const std::size_t v6 = node_index(i + 1, j + 1, k + 1);
                const std::size_t v7 = node_index(i, j + 1, k + 1);

                // Split each hexahedron into six tetrahedra via the 0-6 diagonal.
                mesh.tetrahedra[tet++] = {v0, v1, v2, v6};
                mesh.tetrahedra[tet++] = {v0, v2, v3, v6};
                mesh.tetrahedra[tet++] = {v0, v3, v7, v6};
                mesh.tetrahedra[tet++] = {v0, v7, v4, v6};
                mesh.tetrahedra[tet++] = {v0, v4, v5, v6};
                mesh.tetrahedra[tet++] = {v0, v5, v1, v6};
            }
        }
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

ColMatrix<double> assemble_stiffness_2d(const Mesh2D& mesh) {
    const std::size_t n_nodes = mesh.nodes.size();
    if (n_nodes < 3 || mesh.triangles.empty()) {
        throw std::invalid_argument("assemble_stiffness_2d: mesh is too small");
    }

    ColMatrix<double> K(n_nodes, n_nodes, 0.0);
    for (const auto& tri : mesh.triangles) {
        const std::size_t n0 = tri[0];
        const std::size_t n1 = tri[1];
        const std::size_t n2 = tri[2];
        if (n0 >= n_nodes || n1 >= n_nodes || n2 >= n_nodes) {
            throw std::invalid_argument("assemble_stiffness_2d: invalid connectivity");
        }

        const double x0 = mesh.nodes[n0][0];
        const double y0 = mesh.nodes[n0][1];
        const double x1 = mesh.nodes[n1][0];
        const double y1 = mesh.nodes[n1][1];
        const double x2 = mesh.nodes[n2][0];
        const double y2 = mesh.nodes[n2][1];

        const double area =
            0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0));
        if (std::abs(area) <= 0.0) {
            throw std::invalid_argument("assemble_stiffness_2d: degenerate triangle");
        }

        const double b[3] = {y1 - y2, y2 - y0, y0 - y1};
        const double c[3] = {x2 - x1, x0 - x2, x1 - x0};
        const std::size_t nodes[3] = {n0, n1, n2};

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                const double ke = (b[i] * b[j] + c[i] * c[j]) / (4.0 * area);
                K(nodes[i], nodes[j]) += ke;
            }
        }
    }
    return K;
}

namespace {

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

Vec3 operator-(const Vec3& a, const Vec3& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

double dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 cross(const Vec3& a, const Vec3& b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x};
}

double det3(const Vec3& a, const Vec3& b, const Vec3& c) {
    return dot(a, cross(b, c));
}

bool invert3x3(
    const Vec3& c0,
    const Vec3& c1,
    const Vec3& c2,
    double inv[3][3]) {
    const double det = det3(c0, c1, c2);
    if (std::abs(det) <= 0.0) {
        return false;
    }
    const double inv_det = 1.0 / det;

    inv[0][0] = inv_det * (c1.y * c2.z - c1.z * c2.y);
    inv[0][1] = inv_det * (c2.y * c0.z - c2.z * c0.y);
    inv[0][2] = inv_det * (c0.y * c1.z - c0.z * c1.y);
    inv[1][0] = inv_det * (c1.z * c2.x - c1.x * c2.z);
    inv[1][1] = inv_det * (c2.z * c0.x - c2.x * c0.z);
    inv[1][2] = inv_det * (c0.z * c1.x - c0.x * c1.z);
    inv[2][0] = inv_det * (c1.x * c2.y - c1.y * c2.x);
    inv[2][1] = inv_det * (c2.x * c0.y - c2.y * c0.x);
    inv[2][2] = inv_det * (c0.x * c1.y - c0.y * c1.x);
    return true;
}

Vec3 matvec_transpose(const double m[3][3], const Vec3& v) {
    return {
        m[0][0] * v.x + m[1][0] * v.y + m[2][0] * v.z,
        m[0][1] * v.x + m[1][1] * v.y + m[2][1] * v.z,
        m[0][2] * v.x + m[1][2] * v.y + m[2][2] * v.z};
}

void add_tet_stiffness(
    ColMatrix<double>& K,
    const std::array<Vec3, 4>& verts,
    const std::array<std::size_t, 4>& nodes) {
    const Vec3 e1 = verts[1] - verts[0];
    const Vec3 e2 = verts[2] - verts[0];
    const Vec3 e3 = verts[3] - verts[0];
    const double volume = det3(e1, e2, e3) / 6.0;
    if (std::abs(volume) <= 0.0) {
        throw std::invalid_argument("assemble_stiffness_3d: degenerate tetrahedron");
    }

    double j_inv[3][3];
    if (!invert3x3(e1, e2, e3, j_inv)) {
        throw std::invalid_argument("assemble_stiffness_3d: degenerate tetrahedron");
    }

    const std::array<Vec3, 4> grad_ref = {
        Vec3{-1.0, -1.0, -1.0},
        Vec3{1.0, 0.0, 0.0},
        Vec3{0.0, 1.0, 0.0},
        Vec3{0.0, 0.0, 1.0}};

    std::array<Vec3, 4> grad_phys;
    for (int i = 0; i < 4; ++i) {
        grad_phys[i] = matvec_transpose(j_inv, grad_ref[i]);
    }

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            const double ke = volume * dot(grad_phys[i], grad_phys[j]);
            K(nodes[i], nodes[j]) += ke;
        }
    }
}

} // namespace

ColMatrix<double> assemble_stiffness_3d(const Mesh3D& mesh) {
    const std::size_t n_nodes = mesh.nodes.size();
    if (n_nodes < 4 || mesh.tetrahedra.empty()) {
        throw std::invalid_argument("assemble_stiffness_3d: mesh is too small");
    }

    ColMatrix<double> K(n_nodes, n_nodes, 0.0);
    for (const auto& tet : mesh.tetrahedra) {
        const std::size_t n0 = tet[0];
        const std::size_t n1 = tet[1];
        const std::size_t n2 = tet[2];
        const std::size_t n3 = tet[3];
        if (n0 >= n_nodes || n1 >= n_nodes || n2 >= n_nodes || n3 >= n_nodes) {
            throw std::invalid_argument("assemble_stiffness_3d: invalid connectivity");
        }

        const std::array<Vec3, 4> verts = {
            Vec3{mesh.nodes[n0][0], mesh.nodes[n0][1], mesh.nodes[n0][2]},
            Vec3{mesh.nodes[n1][0], mesh.nodes[n1][1], mesh.nodes[n1][2]},
            Vec3{mesh.nodes[n2][0], mesh.nodes[n2][1], mesh.nodes[n2][2]},
            Vec3{mesh.nodes[n3][0], mesh.nodes[n3][1], mesh.nodes[n3][2]}};
        add_tet_stiffness(K, verts, tet);
    }
    return K;
}

ColMatrix<double> assemble_load_2d(
    const Mesh2D& mesh,
    const std::function<double(double, double)>& f) {
    const std::size_t n_nodes = mesh.nodes.size();
    if (n_nodes < 3 || mesh.triangles.empty()) {
        throw std::invalid_argument("assemble_load_2d: mesh is too small");
    }

    ColMatrix<double> load(n_nodes, 1, 0.0);
    for (const auto& tri : mesh.triangles) {
        const std::size_t n0 = tri[0];
        const std::size_t n1 = tri[1];
        const std::size_t n2 = tri[2];
        const double x0 = mesh.nodes[n0][0];
        const double y0 = mesh.nodes[n0][1];
        const double x1 = mesh.nodes[n1][0];
        const double y1 = mesh.nodes[n1][1];
        const double x2 = mesh.nodes[n2][0];
        const double y2 = mesh.nodes[n2][1];

        const double area =
            0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0));
        if (std::abs(area) <= 0.0) {
            throw std::invalid_argument("assemble_load_2d: degenerate triangle");
        }

        const std::size_t nodes[3] = {n0, n1, n2};
        const double weight = std::abs(area) / 3.0;

        for (int q = 0; q < 3; ++q) {
            const double xi = k_tri_quad_xi[q];
            const double eta = k_tri_quad_eta[q];
            const double zeta = 1.0 - xi - eta;
            const double x = x0 * zeta + x1 * xi + x2 * eta;
            const double y = y0 * zeta + y1 * xi + y2 * eta;
            const double fq = f(x, y);
            const double N[3] = {zeta, xi, eta};
            for (int i = 0; i < 3; ++i) {
                load(nodes[i], 0) += weight * fq * N[i];
            }
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
    if (K.rows() != K.cols()) {
        return std::unexpected(DimensionMismatch{K.rows(), K.cols()});
    }
    if (K.rows() != f.rows() || f.cols() != 1) {
        return std::unexpected(DimensionMismatch{f.rows(), f.cols(), K.rows(), 1});
    }
    if (is_tridiagonal(K)) {
        return solve_fem_tridiag(K, f);
    }
    return solve(K, f);
}

} // namespace fem
} // namespace ms
