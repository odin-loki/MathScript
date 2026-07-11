#include "ms/distributed/mpi_context.hpp"
#include "ms/frameworks/axiom/axiom.hpp"
#include "ms/frameworks/cellai/cellai.hpp"
#include "ms/frameworks/cypha/cypha.hpp"
#include "ms/frameworks/gria/gria.hpp"
#include "ms/frameworks/izaac/izaac.hpp"
#include "ms/interp/repl_engine.hpp"
#include "ms/version.hpp"
#include "ms/interp/plot_console.hpp"
#include "ms/core/operations.hpp"
#include "ms/fft/fft.hpp"
#include "ms/linalg/linalg.hpp"
#include "ms/runtime/dispatch.hpp"
#include "ms/runtime/load_balancer.hpp"
#include "ms/runtime/topology.hpp"
#include "ms/simd/simd.hpp"
#include "ms/special/special.hpp"
#include "ms/image/image.hpp"
#include "ms/compress/compress.hpp"
#include "ms/bignum/bignum.hpp"
#include "ms/ml/ml.hpp"
#include "ms/graph/graph.hpp"
#include "ms/geo/geo.hpp"
#include "ms/combo/combo.hpp"
#include "ms/numthy/numthy.hpp"
#include "ms/control/control.hpp"
#include "ms/quantum/quantum.hpp"
#include "ms/finance/finance.hpp"
#include "ms/info/info.hpp"
#include "ms/cplx/cplx.hpp"
#include "ms/tensorops/tensorops.hpp"
#include "ms/diffgeo/diffgeo.hpp"
#include "ms/topo/topo.hpp"
#include "ms/stats/stats.hpp"
#include "ms/prob/prob.hpp"
#include "ms/signal/signal.hpp"
#include "ms/poly/poly.hpp"
#include "ms/pde/pde.hpp"
#include "ms/symbolic/symbolic.hpp"
#include "ms/ode/ode.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <memory>
#include <optional>
#include <regex>
#include <sstream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ms::interp {

namespace {

std::string trim_copy(std::string s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) {
        s.erase(s.begin());
    }
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) {
        s.pop_back();
    }
    return s;
}

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

bool parse_number(const std::string& text, double& value) {
    if (text.empty()) {
        return false;
    }
    char* end = nullptr;
    value = std::strtod(text.c_str(), &end);
    return end == text.c_str() + text.size();
}

double matrix_max_value(const Matrix<double>& m) {
    double max_val = 0.0;
    for (size_t i = 0; i < m.rows(); ++i) {
        for (size_t j = 0; j < m.cols(); ++j) {
            max_val = std::fmax(max_val, std::fabs(m(i, j)));
        }
    }
    return max_val;
}

Result<image::Image> matrix_to_rgb_image(const Matrix<double>& m) {
    if (m.cols() != 3) {
        return std::unexpected(
            DomainError{"rgb2gray", "expected (H*W) x 3 matrix with RGB rows in [0,1] or [0,255]"});
    }
    const int rows = static_cast<int>(m.rows());
    image::Image img(rows, 1, 3);
    const bool scale255 = matrix_max_value(m) > 1.0;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < 3; ++c) {
            double v = m(r, static_cast<size_t>(c));
            if (scale255) {
                v /= 255.0;
            }
            img.at(r, 0, c) = static_cast<float>(v);
        }
    }
    return img;
}

Matrix<double> gray_image_to_column(const image::Image& img) {
    Matrix<double> out(static_cast<size_t>(img.rows), 1);
    for (int r = 0; r < img.rows; ++r) {
        out(static_cast<size_t>(r), 0) = img.at(r, 0, 0);
    }
    return out;
}

Result<image::Image> matrix_to_gray_image(const Matrix<double>& m) {
    if (m.rows() == 0 || m.cols() == 0) {
        return std::unexpected(DomainError{"sobel", "empty matrix"});
    }
    const int rows = static_cast<int>(m.rows());
    const int cols = static_cast<int>(m.cols());
    image::Image img(rows, cols, 1);
    const bool scale255 = matrix_max_value(m) > 1.0;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            double v = m(static_cast<size_t>(r), static_cast<size_t>(c));
            if (scale255) {
                v /= 255.0;
            }
            img.at(r, c, 0) = static_cast<float>(v);
        }
    }
    return img;
}

Matrix<double> gray_image_to_matrix(const image::Image& img) {
    Matrix<double> out(static_cast<size_t>(img.rows), static_cast<size_t>(img.cols));
    for (int r = 0; r < img.rows; ++r) {
        for (int c = 0; c < img.cols; ++c) {
            out(static_cast<size_t>(r), static_cast<size_t>(c)) = img.at(r, c, 0);
        }
    }
    return out;
}

compress::Bytes matrix_to_bytes(const Matrix<double>& m) {
    compress::Bytes bytes;
    bytes.reserve(m.rows() * m.cols());
    const bool scale255 = matrix_max_value(m) <= 1.0;
    for (size_t i = 0; i < m.rows(); ++i) {
        for (size_t j = 0; j < m.cols(); ++j) {
            double v = m(i, j);
            if (scale255) {
                v *= 255.0;
            }
            v = std::clamp(v, 0.0, 255.0);
            bytes.push_back(static_cast<uint8_t>(std::lround(v)));
        }
    }
    return bytes;
}

Matrix<double> bytes_to_matrix_col(const compress::Bytes& bytes) {
    Matrix<double> out(bytes.size(), 1);
    for (size_t i = 0; i < bytes.size(); ++i) {
        out(i, 0) = static_cast<double>(bytes[i]);
    }
    return out;
}

Result<double> eval_bigint_string(const std::string& decimal) {
    if (decimal.empty()) {
        return std::unexpected(DomainError{"bigint", "expected decimal string literal"});
    }
    const bignum::BigInt value(decimal);
    const double as_double = value.to_double();
    if (!std::isfinite(as_double)) {
        return std::unexpected(DomainError{"bigint", "value too large for scalar double"});
    }
    std::ostringstream roundtrip;
    roundtrip << std::fixed << std::setprecision(0) << as_double;
    if (bignum::BigInt(roundtrip.str()) != value) {
        return std::unexpected(DomainError{"bigint", "value too large for scalar double"});
    }
    return as_double;
}

Result<double> bigint_to_scalar(const bignum::BigInt& value, const char* fn) {
    const double as_double = value.to_double();
    if (!std::isfinite(as_double)) {
        return std::unexpected(DomainError{fn, "value too large for scalar double"});
    }
    std::ostringstream roundtrip;
    roundtrip << std::fixed << std::setprecision(0) << as_double;
    if (bignum::BigInt(roundtrip.str()) != value) {
        return std::unexpected(DomainError{fn, "value too large for scalar double"});
    }
    return as_double;
}

bool parse_quoted_string(const std::string& text, std::string& out) {
    const std::string token = trim_copy(text);
    if (token.size() < 2) {
        return false;
    }
    if ((token.front() == '"' && token.back() == '"') || (token.front() == '\'' && token.back() == '\'')) {
        out = token.substr(1, token.size() - 2);
        return true;
    }
    return false;
}

Result<ml::Vec> matrix_to_ml_vec(const Matrix<double>& m, const char* fn) {
    if (m.cols() == 1) {
        ml::Vec v(m.rows());
        for (size_t i = 0; i < m.rows(); ++i) {
            v[i] = m(i, 0);
        }
        return v;
    }
    if (m.rows() == 1) {
        ml::Vec v(m.cols());
        for (size_t j = 0; j < m.cols(); ++j) {
            v[j] = m(0, j);
        }
        return v;
    }
    return std::unexpected(DomainError{fn, "expected Nx1 or 1xN vector"});
}

Result<ml::Mat> matrix_to_ml_mat(const Matrix<double>& m, const char* fn) {
    if (m.rows() == 0 || m.cols() == 0) {
        return std::unexpected(DomainError{fn, "expected non-empty matrix"});
    }
    ml::Mat out(m.rows());
    for (size_t i = 0; i < m.rows(); ++i) {
        out[i].resize(m.cols());
        for (size_t j = 0; j < m.cols(); ++j) {
            out[i][j] = m(i, j);
        }
    }
    return out;
}

Result<graph::Graph> graph_from_adjacency(const Matrix<double>& adj, const char* fn) {
    if (adj.rows() != adj.cols()) {
        return std::unexpected(DomainError{fn, "expected square adjacency matrix"});
    }
    const int n = static_cast<int>(adj.rows());
    graph::Graph G(n, true);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            const double w = adj(static_cast<size_t>(i), static_cast<size_t>(j));
            if (w > 0.0) {
                G.add_edge(i, j, w);
            }
        }
    }
    return G;
}

Result<graph::Graph> graph_from_adjacency_undirected(const Matrix<double>& adj, const char* fn) {
    if (adj.rows() != adj.cols()) {
        return std::unexpected(DomainError{fn, "expected square adjacency matrix"});
    }
    const int n = static_cast<int>(adj.rows());
    graph::Graph G(n, false);
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            const double w =
                std::max(adj(static_cast<size_t>(i), static_cast<size_t>(j)),
                         adj(static_cast<size_t>(j), static_cast<size_t>(i)));
            if (w > 0.0) {
                G.add_edge(i, j, w);
            }
        }
    }
    return G;
}

Matrix<double> vector_to_column(const std::vector<double>& values) {
    Matrix<double> out(values.size(), 1);
    for (size_t i = 0; i < values.size(); ++i) {
        out(i, 0) = values[i];
    }
    return out;
}

Matrix<double> int_vector_to_column(const std::vector<int>& values) {
    Matrix<double> out(values.size(), 1);
    for (size_t i = 0; i < values.size(); ++i) {
        out(i, 0) = static_cast<double>(values[i]);
    }
    return out;
}

Matrix<double> grid_to_matrix(const std::vector<std::vector<double>>& grid) {
    if (grid.empty()) {
        return Matrix<double>(0, 0);
    }
    Matrix<double> out(grid.size(), grid[0].size());
    for (size_t i = 0; i < grid.size(); ++i) {
        for (size_t j = 0; j < grid[i].size(); ++j) {
            out(i, j) = grid[i][j];
        }
    }
    return out;
}

Result<std::vector<std::vector<double>>> matrix_to_grid(const Matrix<double>& m, const char* fn) {
    auto grid = matrix_to_ml_mat(m, fn);
    if (!grid) {
        return std::unexpected(grid.error());
    }
    return *grid;
}

Result<std::vector<geo::Point2D>> matrix_to_points2d(const Matrix<double>& m, const char* fn) {
    if (m.cols() != 2) {
        return std::unexpected(DomainError{fn, "expected Nx2 point matrix"});
    }
    std::vector<geo::Point2D> points;
    points.reserve(m.rows());
    for (size_t i = 0; i < m.rows(); ++i) {
        points.push_back({m(i, 0), m(i, 1)});
    }
    return points;
}

Result<std::vector<std::complex<double>>> matrix_to_complex_spectrum(const Matrix<double>& m,
                                                                     const char* fn) {
    if (m.cols() != 2) {
        return std::unexpected(DomainError{fn, "expected Nx2 spectrum matrix [re,im]"});
    }
    if (m.rows() == 0) {
        return std::unexpected(DomainError{fn, "expected non-empty spectrum matrix"});
    }
    std::vector<std::complex<double>> spec;
    spec.reserve(m.rows());
    for (size_t i = 0; i < m.rows(); ++i) {
        spec.emplace_back(m(i, 0), m(i, 1));
    }
    return spec;
}

Result<std::vector<double>> matrix_to_coeff_vector(const Matrix<double>& m, const char* fn) {
    std::vector<double> coeffs;
    if (m.rows() == 1 && m.cols() >= 1) {
        coeffs.resize(m.cols());
        for (size_t j = 0; j < m.cols(); ++j) {
            coeffs[j] = m(0, j);
        }
        return coeffs;
    }
    if (m.cols() == 1 && m.rows() >= 1) {
        coeffs.resize(m.rows());
        for (size_t i = 0; i < m.rows(); ++i) {
            coeffs[i] = m(i, 0);
        }
        return coeffs;
    }
    return std::unexpected(DomainError{fn, "expected 1xN or Nx1 coefficient vector"});
}

Result<quantum::Ket> matrix_to_ket(const Matrix<double>& m, const char* fn) {
    auto coeffs = matrix_to_coeff_vector(m, fn);
    if (!coeffs) {
        return std::unexpected(coeffs.error());
    }
    if (coeffs->empty()) {
        return std::unexpected(DomainError{fn, "expected non-empty state vector"});
    }
    quantum::Ket psi;
    psi.reserve(coeffs->size());
    for (double re : *coeffs) {
        psi.push_back(quantum::C(re, 0.0));
    }
    return psi;
}

Result<quantum::Ket> matrix_to_ket2(const Matrix<double>& m, const char* fn) {
    if (m.rows() != 2 || m.cols() != 1) {
        return std::unexpected(DomainError{fn, "expected 2x1 state vector"});
    }
    return quantum::Ket{quantum::C(m(0, 0), 0.0), quantum::C(m(1, 0), 0.0)};
}

Matrix<double> ket_to_column_matrix(const quantum::Ket& psi) {
    Matrix<double> col(psi.size(), 1);
    for (size_t i = 0; i < psi.size(); ++i) {
        col(i, 0) = psi[i].real();
    }
    return col;
}

Result<quantum::DensityMatrix> matrix_to_density_matrix(const Matrix<double>& m, const char* fn) {
    if (m.rows() == 0 || m.cols() == 0) {
        return std::unexpected(DomainError{fn, "expected non-empty square density matrix"});
    }
    if (m.rows() != m.cols()) {
        return std::unexpected(DomainError{fn, "expected square NxN density matrix"});
    }
    const size_t n = m.rows();
    quantum::DensityMatrix rho(n, std::vector<quantum::C>(n));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            rho[i][j] = quantum::C(m(i, j), 0.0);
        }
    }
    return rho;
}

Matrix<double> density_matrix_to_matrix(const quantum::DensityMatrix& rho) {
    const size_t n = rho.size();
    Matrix<double> m(n, n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            m(i, j) = rho[i][j].real();
        }
    }
    return m;
}

Result<tensorops::Tensor> matrix_to_tensor(const Matrix<double>& m, const char* fn) {
    if (m.rows() == 0 || m.cols() == 0) {
        return std::unexpected(DomainError{fn, "expected non-empty matrix"});
    }
    std::vector<double> data;
    data.reserve(m.rows() * m.cols());
    for (size_t i = 0; i < m.rows(); ++i) {
        for (size_t j = 0; j < m.cols(); ++j) {
            data.push_back(m(i, j));
        }
    }
    return tensorops::Tensor({static_cast<int>(m.rows()), static_cast<int>(m.cols())},
                             std::move(data));
}

Result<double> eval_finance_npv(double rate, const Matrix<double>& cashflows_m) {
    auto cashflows = matrix_to_coeff_vector(cashflows_m, "finance_npv");
    if (!cashflows) {
        return std::unexpected(cashflows.error());
    }
    return finance::npv(rate, *cashflows);
}

Result<double> eval_finance_sharpe(const Matrix<double>& returns_m) {
    auto returns = matrix_to_coeff_vector(returns_m, "finance_sharpe");
    if (!returns) {
        return std::unexpected(returns.error());
    }
    if (returns->empty()) {
        return std::unexpected(DomainError{"finance_sharpe", "expected non-empty return vector"});
    }
    return finance::sharpe_ratio(*returns);
}

Result<double> eval_finance_sortino(const Matrix<double>& returns_m) {
    auto returns = matrix_to_coeff_vector(returns_m, "finance_sortino");
    if (!returns) {
        return std::unexpected(returns.error());
    }
    if (returns->empty()) {
        return std::unexpected(DomainError{"finance_sortino", "expected non-empty return vector"});
    }
    return finance::sortino_ratio(*returns, 0.0);
}

Result<double> eval_finance_var(const Matrix<double>& returns_m) {
    auto returns = matrix_to_coeff_vector(returns_m, "finance_var");
    if (!returns) {
        return std::unexpected(returns.error());
    }
    if (returns->empty()) {
        return std::unexpected(DomainError{"finance_var", "expected non-empty return vector"});
    }
    return finance::var(*returns, 0.95);
}

Result<double> eval_finance_cvar(const Matrix<double>& returns_m) {
    auto returns = matrix_to_coeff_vector(returns_m, "finance_cvar");
    if (!returns) {
        return std::unexpected(returns.error());
    }
    if (returns->empty()) {
        return std::unexpected(DomainError{"finance_cvar", "expected non-empty return vector"});
    }
    return finance::cvar(*returns, 0.95);
}

Result<double> eval_finance_max_drawdown(const Matrix<double>& equity_m) {
    auto equity = matrix_to_coeff_vector(equity_m, "finance_max_drawdown");
    if (!equity) {
        return std::unexpected(equity.error());
    }
    if (equity->empty()) {
        return std::unexpected(
            DomainError{"finance_max_drawdown", "expected non-empty equity vector"});
    }
    return finance::max_drawdown(*equity);
}

Result<double> eval_finance_irr(const Matrix<double>& cashflows_m) {
    auto cashflows = matrix_to_coeff_vector(cashflows_m, "finance_irr");
    if (!cashflows) {
        return std::unexpected(cashflows.error());
    }
    if (cashflows->empty()) {
        return std::unexpected(DomainError{"finance_irr", "expected non-empty cashflow vector"});
    }
    return finance::irr(*cashflows);
}

Result<double> eval_finance_bond_ytm(double price, double c, double n_d) {
    const int n = static_cast<int>(n_d);
    if (n < 0 || n_d != n) {
        return std::unexpected(
            DomainError{"finance_bond_ytm", "expected non-negative integer periods n"});
    }
    return finance::bond_ytm(price, c, n);
}

Result<double> eval_numthy_tonelli_shanks(double n_d, double p_d) {
    if (std::floor(n_d) != n_d || std::floor(p_d) != p_d) {
        return std::unexpected(
            DomainError{"numthy_tonelli_shanks", "expected integer arguments"});
    }
    if (n_d < 0.0 || p_d <= 0.0) {
        return std::unexpected(
            DomainError{"numthy_tonelli_shanks", "expected n >= 0 and p > 0"});
    }
    auto root = numthy::tonelli_shanks(static_cast<uint64_t>(n_d), static_cast<uint64_t>(p_d));
    if (!root) {
        return std::unexpected(root.error());
    }
    return static_cast<double>(*root);
}

Result<double> eval_numthy_mod_inv(double a_d, double m_d) {
    if (std::floor(a_d) != a_d || std::floor(m_d) != m_d) {
        return std::unexpected(
            DomainError{"numthy_mod_inv", "expected integer arguments"});
    }
    if (a_d < 0.0 || m_d <= 0.0) {
        return std::unexpected(
            DomainError{"numthy_mod_inv", "expected a >= 0 and m > 0"});
    }
    auto inv = numthy::mod_inv(static_cast<uint64_t>(a_d), static_cast<uint64_t>(m_d));
    if (!inv) {
        return std::unexpected(inv.error());
    }
    return static_cast<double>(*inv);
}

Result<double> eval_numthy_discrete_log(double g_d, double h_d, double p_d) {
    if (std::floor(g_d) != g_d || std::floor(h_d) != h_d || std::floor(p_d) != p_d) {
        return std::unexpected(
            DomainError{"numthy_discrete_log", "expected integer arguments"});
    }
    if (g_d < 0.0 || h_d < 0.0 || p_d <= 0.0) {
        return std::unexpected(
            DomainError{"numthy_discrete_log", "expected g >= 0, h >= 0, p > 0"});
    }
    auto x = numthy::discrete_log(static_cast<uint64_t>(g_d), static_cast<uint64_t>(h_d),
                                   static_cast<uint64_t>(p_d));
    if (!x) {
        return std::unexpected(x.error());
    }
    return static_cast<double>(*x);
}

Result<double> eval_finance_bs_implied_vol(double price, double S, double K, double T, double r,
                                           double call_d) {
    const int call = static_cast<int>(call_d);
    if (call_d != call) {
        return std::unexpected(
            DomainError{"finance_bs_implied_vol", "expected integer call (0=put, 1=call)"});
    }
    return finance::bs_implied_vol(price, S, K, T, r, call != 0);
}

Result<std::vector<double>> matrix_to_row_major_flat(const Matrix<double>& m, const char* fn) {
    if (m.rows() != m.cols()) {
        return std::unexpected(DomainError{fn, "expected square matrix"});
    }
    const size_t n = m.rows();
    std::vector<double> flat(n * n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            flat[i * n + j] = m(i, j);
        }
    }
    return flat;
}

Result<double> eval_finance_portfolio_return(const Matrix<double>& w_m,
                                             const Matrix<double>& ret_m) {
    auto w = matrix_to_coeff_vector(w_m, "finance_portfolio_return");
    if (!w) {
        return std::unexpected(w.error());
    }
    auto ret = matrix_to_coeff_vector(ret_m, "finance_portfolio_return");
    if (!ret) {
        return std::unexpected(ret.error());
    }
    if (w->empty() || ret->empty()) {
        return std::unexpected(
            DomainError{"finance_portfolio_return", "expected non-empty vectors"});
    }
    if (w->size() != ret->size()) {
        return std::unexpected(DomainError{"finance_portfolio_return", "vector length mismatch"});
    }
    return finance::portfolio_return(*w, *ret);
}

Result<double> eval_finance_portfolio_variance(const Matrix<double>& w_m,
                                               const Matrix<double>& cov_m) {
    auto w = matrix_to_coeff_vector(w_m, "finance_portfolio_variance");
    if (!w) {
        return std::unexpected(w.error());
    }
    if (w->empty()) {
        return std::unexpected(
            DomainError{"finance_portfolio_variance", "expected non-empty weight vector"});
    }
    auto cov = matrix_to_row_major_flat(cov_m, "finance_portfolio_variance");
    if (!cov) {
        return std::unexpected(cov.error());
    }
    const size_t n = w->size();
    if (cov_m.rows() != n || cov_m.cols() != n) {
        return std::unexpected(
            DomainError{"finance_portfolio_variance", "covariance size mismatch"});
    }
    return finance::portfolio_variance(*w, *cov);
}

Result<double> eval_info_entropy(const Matrix<double>& prob_m) {
    auto probs = matrix_to_coeff_vector(prob_m, "info_entropy");
    if (!probs) {
        return std::unexpected(probs.error());
    }
    if (probs->empty()) {
        return std::unexpected(DomainError{"info_entropy", "expected non-empty probability vector"});
    }
    return info::entropy(*probs);
}

Result<double> eval_info_lz_complexity(const Matrix<double>& seq_m) {
    auto coeffs = matrix_to_coeff_vector(seq_m, "info_lz_complexity");
    if (!coeffs) {
        return std::unexpected(coeffs.error());
    }
    if (coeffs->empty()) {
        return std::unexpected(
            DomainError{"info_lz_complexity", "expected non-empty integer sequence vector"});
    }
    std::vector<int> seq;
    seq.reserve(coeffs->size());
    for (double v : *coeffs) {
        if (std::floor(v) != v) {
            return std::unexpected(
                DomainError{"info_lz_complexity", "sequence elements must be integers"});
        }
        seq.push_back(static_cast<int>(v));
    }
    return info::lz_complexity(seq);
}

Result<double> eval_info_redundancy(const Matrix<double>& p_m) {
    auto p = matrix_to_coeff_vector(p_m, "info_redundancy");
    if (!p) {
        return std::unexpected(p.error());
    }
    if (p->empty()) {
        return std::unexpected(
            DomainError{"info_redundancy", "expected non-empty probability vector"});
    }
    return info::redundancy(*p);
}

Result<double> eval_info_efficiency(const Matrix<double>& p_m) {
    auto p = matrix_to_coeff_vector(p_m, "info_efficiency");
    if (!p) {
        return std::unexpected(p.error());
    }
    if (p->empty()) {
        return std::unexpected(
            DomainError{"info_efficiency", "expected non-empty probability vector"});
    }
    return info::efficiency(*p);
}

Result<double> eval_info_mutual_info(const Matrix<double>& joint_m) {
    if (joint_m.rows() == 0 || joint_m.cols() == 0) {
        return std::unexpected(
            DomainError{"info_mutual_info", "expected non-empty joint PMF matrix"});
    }
    std::vector<double> flat;
    flat.reserve(joint_m.rows() * joint_m.cols());
    for (size_t i = 0; i < joint_m.rows(); ++i) {
        for (size_t j = 0; j < joint_m.cols(); ++j) {
            flat.push_back(joint_m(i, j));
        }
    }
    return info::mutual_info(flat, static_cast<int>(joint_m.rows()),
                             static_cast<int>(joint_m.cols()), 2.0);
}

Result<double> eval_info_joint_entropy(const Matrix<double>& joint_m, int rows, int cols) {
    if (joint_m.rows() == 0 || joint_m.cols() == 0) {
        return std::unexpected(
            DomainError{"info_joint_entropy", "expected non-empty joint PMF matrix"});
    }
    std::vector<double> flat;
    flat.reserve(joint_m.rows() * joint_m.cols());
    for (size_t i = 0; i < joint_m.rows(); ++i) {
        for (size_t j = 0; j < joint_m.cols(); ++j) {
            flat.push_back(joint_m(i, j));
        }
    }
    return info::joint_entropy(flat, rows, cols, 2.0);
}

Result<double> eval_info_conditional_entropy(const Matrix<double>& joint_m, int rows, int cols) {
    if (joint_m.rows() == 0 || joint_m.cols() == 0) {
        return std::unexpected(
            DomainError{"info_conditional_entropy", "expected non-empty joint PMF matrix"});
    }
    std::vector<double> flat;
    flat.reserve(joint_m.rows() * joint_m.cols());
    for (size_t i = 0; i < joint_m.rows(); ++i) {
        for (size_t j = 0; j < joint_m.cols(); ++j) {
            flat.push_back(joint_m(i, j));
        }
    }
    return info::conditional_entropy(flat, rows, cols, 2.0);
}

Result<double> eval_info_sample_entropy(const Matrix<double>& x_m, int m, double r) {
    auto x = matrix_to_coeff_vector(x_m, "info_sample_entropy");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (x->empty()) {
        return std::unexpected(
            DomainError{"info_sample_entropy", "expected non-empty time series vector"});
    }
    return info::sample_entropy(*x, m, r);
}

Result<double> eval_info_kl_divergence(const Matrix<double>& p_m, const Matrix<double>& q_m) {
    auto p = matrix_to_coeff_vector(p_m, "info_kl_divergence");
    if (!p) {
        return std::unexpected(p.error());
    }
    auto q = matrix_to_coeff_vector(q_m, "info_kl_divergence");
    if (!q) {
        return std::unexpected(q.error());
    }
    if (p->empty() || q->empty()) {
        return std::unexpected(
            DomainError{"info_kl_divergence", "expected non-empty probability vectors"});
    }
    if (p->size() != q->size()) {
        return std::unexpected(DomainError{"info_kl_divergence", "vector length mismatch"});
    }
    return info::kl_divergence(*p, *q);
}

Result<double> eval_info_js_divergence(const Matrix<double>& p_m, const Matrix<double>& q_m) {
    auto p = matrix_to_coeff_vector(p_m, "info_js_divergence");
    if (!p) {
        return std::unexpected(p.error());
    }
    auto q = matrix_to_coeff_vector(q_m, "info_js_divergence");
    if (!q) {
        return std::unexpected(q.error());
    }
    if (p->empty() || q->empty()) {
        return std::unexpected(
            DomainError{"info_js_divergence", "expected non-empty probability vectors"});
    }
    if (p->size() != q->size()) {
        return std::unexpected(DomainError{"info_js_divergence", "vector length mismatch"});
    }
    return info::js_divergence(*p, *q, 2.0);
}

Result<double> eval_info_tv_distance(const Matrix<double>& p_m, const Matrix<double>& q_m) {
    auto p = matrix_to_coeff_vector(p_m, "info_tv_distance");
    if (!p) {
        return std::unexpected(p.error());
    }
    auto q = matrix_to_coeff_vector(q_m, "info_tv_distance");
    if (!q) {
        return std::unexpected(q.error());
    }
    if (p->empty() || q->empty()) {
        return std::unexpected(
            DomainError{"info_tv_distance", "expected non-empty probability vectors"});
    }
    if (p->size() != q->size()) {
        return std::unexpected(DomainError{"info_tv_distance", "vector length mismatch"});
    }
    return info::tv_distance(*p, *q);
}

Result<double> eval_info_hellinger_dist(const Matrix<double>& p_m, const Matrix<double>& q_m) {
    auto p = matrix_to_coeff_vector(p_m, "info_hellinger_dist");
    if (!p) {
        return std::unexpected(p.error());
    }
    auto q = matrix_to_coeff_vector(q_m, "info_hellinger_dist");
    if (!q) {
        return std::unexpected(q.error());
    }
    if (p->empty() || q->empty()) {
        return std::unexpected(
            DomainError{"info_hellinger_dist", "expected non-empty probability vectors"});
    }
    if (p->size() != q->size()) {
        return std::unexpected(DomainError{"info_hellinger_dist", "vector length mismatch"});
    }
    return info::hellinger_dist(*p, *q);
}

Result<double> eval_info_cross_entropy(const Matrix<double>& p_m, const Matrix<double>& q_m) {
    auto p = matrix_to_coeff_vector(p_m, "info_cross_entropy");
    if (!p) {
        return std::unexpected(p.error());
    }
    auto q = matrix_to_coeff_vector(q_m, "info_cross_entropy");
    if (!q) {
        return std::unexpected(q.error());
    }
    if (p->empty() || q->empty()) {
        return std::unexpected(
            DomainError{"info_cross_entropy", "expected non-empty probability vectors"});
    }
    if (p->size() != q->size()) {
        return std::unexpected(DomainError{"info_cross_entropy", "vector length mismatch"});
    }
    return info::cross_entropy(*p, *q, 2.0);
}

Result<double> eval_info_renyi_entropy(double alpha, const Matrix<double>& p_m) {
    auto p = matrix_to_coeff_vector(p_m, "info_renyi_entropy");
    if (!p) {
        return std::unexpected(p.error());
    }
    if (p->empty()) {
        return std::unexpected(
            DomainError{"info_renyi_entropy", "expected non-empty probability vector"});
    }
    return info::renyi_entropy(*p, alpha, 2.0);
}

Result<double> eval_info_source_coding_rate(const Matrix<double>& p_m) {
    auto p = matrix_to_coeff_vector(p_m, "info_source_coding_rate");
    if (!p) {
        return std::unexpected(p.error());
    }
    if (p->empty()) {
        return std::unexpected(
            DomainError{"info_source_coding_rate", "expected non-empty probability vector"});
    }
    return info::source_coding_rate(*p);
}

Result<double> eval_info_tsallis_entropy(double q_param, const Matrix<double>& p_m) {
    auto p = matrix_to_coeff_vector(p_m, "info_tsallis_entropy");
    if (!p) {
        return std::unexpected(p.error());
    }
    if (p->empty()) {
        return std::unexpected(
            DomainError{"info_tsallis_entropy", "expected non-empty probability vector"});
    }
    return info::tsallis_entropy(*p, q_param);
}

Result<double> eval_quantum_entanglement_entropy(const Matrix<double>& psi_m, int dim_a,
                                                 int dim_b) {
    auto psi = matrix_to_ket(psi_m, "quantum_entanglement_entropy");
    if (!psi) {
        return std::unexpected(psi.error());
    }
    return quantum::entanglement_entropy(*psi, dim_a, dim_b);
}

Result<double> eval_quantum_von_neumann_entropy(const Matrix<double>& rho_m) {
    auto rho = matrix_to_density_matrix(rho_m, "quantum_von_neumann_entropy");
    if (!rho) {
        return std::unexpected(rho.error());
    }
    return quantum::von_neumann_entropy(*rho);
}

Result<double> eval_quantum_concurrence(const Matrix<double>& rho_m) {
    auto rho = matrix_to_density_matrix(rho_m, "quantum_concurrence");
    if (!rho) {
        return std::unexpected(rho.error());
    }
    return quantum::concurrence(*rho);
}

Result<Matrix<double>> eval_quantum_ket_normalise_matrix(const Matrix<double>& psi_m) {
    auto psi = matrix_to_ket(psi_m, "quantum_ket_normalise");
    if (!psi) {
        return std::unexpected(psi.error());
    }
    return ket_to_column_matrix(quantum::ket_normalise(*psi));
}

Result<Matrix<double>> eval_quantum_ket_basis(int dim, int index) {
    if (dim < 1) {
        return std::unexpected(DomainError{"quantum_ket_basis", "expected dim >= 1"});
    }
    if (index < 0 || index >= dim) {
        return std::unexpected(DomainError{"quantum_ket_basis", "expected 0 <= index < dim"});
    }
    return ket_to_column_matrix(quantum::ket_basis(dim, index));
}

Result<Matrix<double>> eval_quantum_fock_state(int n, int n_max) {
    if (n_max < 0) {
        return std::unexpected(DomainError{"quantum_fock_state", "expected n_max >= 0"});
    }
    if (n < 0 || n > n_max) {
        return std::unexpected(DomainError{"quantum_fock_state", "expected 0 <= n <= n_max"});
    }
    return ket_to_column_matrix(quantum::fock_state(n, n_max));
}

Result<double> eval_quantum_fidelity(const Matrix<double>& rho_m, const Matrix<double>& sigma_m) {
    auto rho = matrix_to_density_matrix(rho_m, "quantum_fidelity");
    if (!rho) {
        return std::unexpected(rho.error());
    }
    auto sigma = matrix_to_density_matrix(sigma_m, "quantum_fidelity");
    if (!sigma) {
        return std::unexpected(sigma.error());
    }
    return quantum::fidelity(*rho, *sigma);
}

Result<double> eval_quantum_expectation_dm(const Matrix<double>& rho_m, const Matrix<double>& op_m) {
    auto rho = matrix_to_density_matrix(rho_m, "quantum_expectation_dm");
    if (!rho) {
        return std::unexpected(rho.error());
    }
    auto op = matrix_to_density_matrix(op_m, "quantum_expectation_dm");
    if (!op) {
        return std::unexpected(op.error());
    }
    if (rho->size() != op->size()) {
        return std::unexpected(
            DomainError{"quantum_expectation_dm", "density matrices must have same dimension"});
    }
    return quantum::expectation_dm(*rho, *op);
}

Result<double> eval_quantum_expectation(const Matrix<double>& psi_m, const Matrix<double>& op_m) {
    auto psi = matrix_to_ket(psi_m, "quantum_expectation");
    if (!psi) {
        return std::unexpected(psi.error());
    }
    auto op = matrix_to_density_matrix(op_m, "quantum_expectation");
    if (!op) {
        return std::unexpected(op.error());
    }
    if (psi->size() != op->size()) {
        return std::unexpected(
            DomainError{"quantum_expectation", "ket length must match operator dimension"});
    }
    return quantum::expectation(*psi, *op);
}

Result<double> eval_quantum_inner(const Matrix<double>& bra_m, const Matrix<double>& ket_m) {
    auto bra = matrix_to_ket(bra_m, "quantum_inner");
    if (!bra) {
        return std::unexpected(bra.error());
    }
    auto ket = matrix_to_ket(ket_m, "quantum_inner");
    if (!ket) {
        return std::unexpected(ket.error());
    }
    if (bra->size() != ket->size()) {
        return std::unexpected(
            DomainError{"quantum_inner", "ket vectors must have same length"});
    }
    return quantum::inner(*bra, *ket).real();
}

Result<double> eval_quantum_trace_distance(const Matrix<double>& rho_m,
                                           const Matrix<double>& sigma_m) {
    auto rho = matrix_to_density_matrix(rho_m, "quantum_trace_distance");
    if (!rho) {
        return std::unexpected(rho.error());
    }
    auto sigma = matrix_to_density_matrix(sigma_m, "quantum_trace_distance");
    if (!sigma) {
        return std::unexpected(sigma.error());
    }
    return quantum::trace_distance(*rho, *sigma);
}

Result<Matrix<double>> eval_quantum_partial_trace_matrix(const Matrix<double>& rho_m, int d1,
                                                         int d2, int subsystem) {
    auto rho = matrix_to_density_matrix(rho_m, "quantum_partial_trace");
    if (!rho) {
        return std::unexpected(rho.error());
    }
    return density_matrix_to_matrix(quantum::partial_trace(*rho, d1, d2, subsystem));
}

Result<double> eval_tensorops_norm(const Matrix<double>& tensor_m) {
    auto tensor = matrix_to_tensor(tensor_m, "tensorops_norm");
    if (!tensor) {
        return std::unexpected(tensor.error());
    }
    return tensorops::frobenius_norm(*tensor);
}

Matrix<double> tensor_to_matrix(const tensorops::Tensor& t) {
    const size_t rows = t.ndim() >= 1 ? static_cast<size_t>(t.shape[0]) : 1;
    const size_t cols = t.ndim() >= 2 ? static_cast<size_t>(t.shape[1]) : 1;
    Matrix<double> out(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            out(i, j) = t.at({static_cast<int>(i), static_cast<int>(j)});
        }
    }
    return out;
}

Result<Matrix<double>> eval_tensorops_matmul(const Matrix<double>& left_m,
                                             const Matrix<double>& right_m) {
    auto left = matrix_to_tensor(left_m, "tensorops_matmul");
    if (!left) {
        return std::unexpected(left.error());
    }
    auto right = matrix_to_tensor(right_m, "tensorops_matmul");
    if (!right) {
        return std::unexpected(right.error());
    }
    if (left->ndim() != 2 || right->ndim() != 2) {
        return std::unexpected(DomainError{"tensorops_matmul", "expected 2D matrices"});
    }
    if (left->shape[1] != right->shape[0]) {
        return std::unexpected(
            DomainError{"tensorops_matmul",
                          "inner dimensions must agree for matrix multiply"});
    }
    const tensorops::Tensor product =
        tensorops::einsum("ij,jk->ik", *left, *right);
    return tensor_to_matrix(product);
}

Result<Matrix<double>> eval_tensorops_einsum(const Matrix<double>& left_m,
                                             const Matrix<double>& right_m) {
    auto left = matrix_to_tensor(left_m, "tensorops_einsum");
    if (!left) {
        return std::unexpected(left.error());
    }
    auto right = matrix_to_tensor(right_m, "tensorops_einsum");
    if (!right) {
        return std::unexpected(right.error());
    }
    if (left->ndim() != 2 || right->ndim() != 2) {
        return std::unexpected(DomainError{"tensorops_einsum", "expected 2D matrices"});
    }
    if (left->shape[1] != right->shape[0]) {
        return std::unexpected(
            DomainError{"tensorops_einsum",
                          "inner dimensions must agree for matrix multiply"});
    }
    const tensorops::Tensor product =
        tensorops::einsum("ij,jk->ik", *left, *right);
    return tensor_to_matrix(product);
}

Result<double> eval_tensorops_inner(const Matrix<double>& a_m, const Matrix<double>& b_m) {
    auto a = matrix_to_tensor(a_m, "tensorops_inner");
    if (!a) {
        return std::unexpected(a.error());
    }
    auto b = matrix_to_tensor(b_m, "tensorops_inner");
    if (!b) {
        return std::unexpected(b.error());
    }
    if (a->data.size() != b->data.size()) {
        return std::unexpected(DomainError{"tensorops_inner", "tensor size mismatch"});
    }
    return tensorops::tensor_inner(*a, *b);
}

Result<double> eval_geo_polygon_area(const Matrix<double>& points_m) {
    auto points = matrix_to_points2d(points_m, "geo_polygon_area");
    if (!points) {
        return std::unexpected(points.error());
    }
    if (points->size() < 3) {
        return std::unexpected(
            DomainError{"geo_polygon_area", "expected at least 3 points"});
    }
    return geo::area(*points);
}

Result<double> eval_geo_polygon_perimeter(const Matrix<double>& points_m) {
    auto points = matrix_to_points2d(points_m, "geo_polygon_perimeter");
    if (!points) {
        return std::unexpected(points.error());
    }
    if (points->size() < 3) {
        return std::unexpected(
            DomainError{"geo_polygon_perimeter", "expected at least 3 points"});
    }
    return geo::perimeter(*points);
}

Result<double> eval_geo_signed_area(const Matrix<double>& points_m) {
    auto points = matrix_to_points2d(points_m, "geo_signed_area");
    if (!points) {
        return std::unexpected(points.error());
    }
    if (points->size() < 3) {
        return std::unexpected(
            DomainError{"geo_signed_area", "expected at least 3 points"});
    }
    return geo::signed_area(*points);
}

Result<double> eval_geo_moment_of_inertia(const Matrix<double>& points_m) {
    auto points = matrix_to_points2d(points_m, "geo_moment_of_inertia");
    if (!points) {
        return std::unexpected(points.error());
    }
    if (points->size() < 3) {
        return std::unexpected(
            DomainError{"geo_moment_of_inertia", "expected at least 3 points"});
    }
    return geo::moment_of_inertia(*points);
}

Result<double> eval_geo_bezier_eval_x(const Matrix<double>& ctrl_m, double t) {
    auto ctrl = matrix_to_points2d(ctrl_m, "geo_bezier_eval_x");
    if (!ctrl) {
        return std::unexpected(ctrl.error());
    }
    if (ctrl->size() < 3) {
        return std::unexpected(
            DomainError{"geo_bezier_eval_x", "expected at least 3 control points"});
    }
    return geo::bezier_eval(*ctrl, t).x;
}

Result<double> eval_geo_bezier_eval_y(const Matrix<double>& ctrl_m, double t) {
    auto ctrl = matrix_to_points2d(ctrl_m, "geo_bezier_eval_y");
    if (!ctrl) {
        return std::unexpected(ctrl.error());
    }
    if (ctrl->size() < 3) {
        return std::unexpected(
            DomainError{"geo_bezier_eval_y", "expected at least 3 control points"});
    }
    return geo::bezier_eval(*ctrl, t).y;
}

Result<double> eval_geo_point_in_polygon(double px, double py, const Matrix<double>& points_m) {
    auto points = matrix_to_points2d(points_m, "geo_point_in_polygon");
    if (!points) {
        return std::unexpected(points.error());
    }
    if (points->size() < 3) {
        return std::unexpected(
            DomainError{"geo_point_in_polygon", "expected at least 3 points"});
    }
    return geo::point_in_polygon({px, py}, *points) ? 1.0 : 0.0;
}

Result<double> eval_ml_categorical_crossentropy(const Matrix<double>& pred_m,
                                              const Matrix<double>& true_m) {
    auto pred = matrix_to_ml_mat(pred_m, "ml_categorical_crossentropy");
    if (!pred) {
        return std::unexpected(pred.error());
    }
    auto true_labels = matrix_to_ml_mat(true_m, "ml_categorical_crossentropy");
    if (!true_labels) {
        return std::unexpected(true_labels.error());
    }
    if (pred->size() != true_labels->size()) {
        return std::unexpected(
            DomainError{"ml_categorical_crossentropy", "matrix row count mismatch"});
    }
    for (size_t i = 0; i < pred->size(); ++i) {
        if ((*pred)[i].size() != (*true_labels)[i].size()) {
            return std::unexpected(
                DomainError{"ml_categorical_crossentropy", "matrix column count mismatch"});
        }
    }
    return ml::categorical_crossentropy(*pred, *true_labels);
}

Result<double> eval_ml_vec_norm(const Matrix<double>& vec_m) {
    auto vec = matrix_to_ml_vec(vec_m, "ml_vec_norm");
    if (!vec) {
        return std::unexpected(vec.error());
    }
    return ml::vec_norm(*vec);
}

Result<double> eval_control_step_final(const Matrix<double>& num_m, const Matrix<double>& den_m) {
    auto num = matrix_to_coeff_vector(num_m, "control_step_final");
    if (!num) {
        return std::unexpected(num.error());
    }
    auto den = matrix_to_coeff_vector(den_m, "control_step_final");
    if (!den) {
        return std::unexpected(den.error());
    }
    if (den->empty() || den->back() == 0.0) {
        return std::unexpected(DomainError{"control_step_final", "denominator must be non-zero"});
    }
    const control::TransferFunction sys(std::move(*num), std::move(*den));
    const control::StepData response = control::step_response(sys);
    if (response.y.empty()) {
        return std::unexpected(DomainError{"control_step_final", "empty step response"});
    }
    return response.y.back();
}

Result<double> eval_control_dcgain(const Matrix<double>& num_m, const Matrix<double>& den_m) {
    auto num = matrix_to_coeff_vector(num_m, "control_dcgain");
    if (!num) {
        return std::unexpected(num.error());
    }
    auto den = matrix_to_coeff_vector(den_m, "control_dcgain");
    if (!den) {
        return std::unexpected(den.error());
    }
    if (den->empty() || den->back() == 0.0) {
        return std::unexpected(DomainError{"control_dcgain", "denominator must be non-zero"});
    }
    const control::TransferFunction sys(std::move(*num), std::move(*den));
    return control::dcgain(sys);
}

Result<double> eval_control_is_stable(const Matrix<double>& num_m, const Matrix<double>& den_m) {
    auto num = matrix_to_coeff_vector(num_m, "control_is_stable");
    if (!num) {
        return std::unexpected(num.error());
    }
    auto den = matrix_to_coeff_vector(den_m, "control_is_stable");
    if (!den) {
        return std::unexpected(den.error());
    }
    if (den->empty() || den->back() == 0.0) {
        return std::unexpected(DomainError{"control_is_stable", "denominator must be non-zero"});
    }
    const control::TransferFunction sys(std::move(*num), std::move(*den));
    return control::is_stable(sys) ? 1.0 : 0.0;
}

Result<std::vector<std::vector<double>>> matrix_to_nested(const Matrix<double>& m,
                                                           const char* fn) {
    if (m.rows() == 0 || m.cols() == 0) {
        return std::unexpected(DomainError{fn, "expected non-empty matrix"});
    }
    std::vector<std::vector<double>> out(m.rows(), std::vector<double>(m.cols()));
    for (size_t i = 0; i < m.rows(); ++i) {
        for (size_t j = 0; j < m.cols(); ++j) {
            out[i][j] = m(i, j);
        }
    }
    return out;
}

Result<std::vector<std::vector<double>>> matrix_to_square_nested(const Matrix<double>& m,
                                                                   const char* fn) {
    if (m.rows() != m.cols()) {
        return std::unexpected(DomainError{fn, "expected square matrix"});
    }
    return matrix_to_nested(m, fn);
}

Result<double> eval_control_is_controllable(const Matrix<double>& A_m,
                                            const Matrix<double>& B_m) {
    auto A = matrix_to_square_nested(A_m, "control_is_controllable");
    if (!A) {
        return std::unexpected(A.error());
    }
    auto B = matrix_to_nested(B_m, "control_is_controllable");
    if (!B) {
        return std::unexpected(B.error());
    }
    if (B->size() != A->size()) {
        return std::unexpected(
            DomainError{"control_is_controllable", "expected B with same row count as A"});
    }
    return control::is_controllable(*A, *B) ? 1.0 : 0.0;
}

Result<double> eval_control_is_observable(const Matrix<double>& A_m,
                                          const Matrix<double>& C_m) {
    auto A = matrix_to_square_nested(A_m, "control_is_observable");
    if (!A) {
        return std::unexpected(A.error());
    }
    auto C = matrix_to_nested(C_m, "control_is_observable");
    if (!C) {
        return std::unexpected(C.error());
    }
    if (C->empty() || C->front().size() != A->size()) {
        return std::unexpected(
            DomainError{"control_is_observable", "expected C with column count equal to A size"});
    }
    return control::is_observable(*A, *C) ? 1.0 : 0.0;
}

Result<double> eval_numthy_extended_gcd(double a_d, double b_d) {
    if (std::floor(a_d) != a_d || std::floor(b_d) != b_d) {
        return std::unexpected(
            DomainError{"numthy_extended_gcd", "expected integer arguments"});
    }
    const auto [g, x, y] =
        numthy::extended_gcd(static_cast<int64_t>(a_d), static_cast<int64_t>(b_d));
    static_cast<void>(x);
    static_cast<void>(y);
    return static_cast<double>(g);
}

Result<double> eval_numthy_crt(const Matrix<double>& r_m, const Matrix<double>& m_m) {
    auto r_vec = matrix_to_coeff_vector(r_m, "numthy_crt");
    if (!r_vec) {
        return std::unexpected(r_vec.error());
    }
    auto m_vec = matrix_to_coeff_vector(m_m, "numthy_crt");
    if (!m_vec) {
        return std::unexpected(m_vec.error());
    }
    if (r_vec->size() != m_vec->size()) {
        return std::unexpected(DimensionMismatch{r_vec->size(), m_vec->size()});
    }
    std::vector<uint64_t> r;
    std::vector<uint64_t> m;
    r.reserve(r_vec->size());
    m.reserve(m_vec->size());
    for (size_t i = 0; i < r_vec->size(); ++i) {
        if (std::floor((*r_vec)[i]) != (*r_vec)[i] || std::floor((*m_vec)[i]) != (*m_vec)[i]) {
            return std::unexpected(DomainError{"numthy_crt", "expected integer vector entries"});
        }
        if ((*r_vec)[i] < 0.0 || (*m_vec)[i] <= 0.0) {
            return std::unexpected(
                DomainError{"numthy_crt", "expected non-negative remainders and positive moduli"});
        }
        r.push_back(static_cast<uint64_t>((*r_vec)[i]));
        m.push_back(static_cast<uint64_t>((*m_vec)[i]));
    }
    auto x = numthy::crt(r, m);
    if (!x) {
        return std::unexpected(x.error());
    }
    return static_cast<double>(*x);
}

Result<double> eval_geo_centroid_x(const Matrix<double>& points_m) {
    auto points = matrix_to_points2d(points_m, "geo_centroid_x");
    if (!points) {
        return std::unexpected(points.error());
    }
    if (points->size() < 3) {
        return std::unexpected(
            DomainError{"geo_centroid_x", "expected at least 3 points"});
    }
    geo::Polygon2D poly(points->begin(), points->end());
    return geo::centroid(poly).x;
}

Result<Matrix<double>> eval_quantum_ket_superposition_matrix(const Matrix<double>& amps_m) {
    auto amps = matrix_to_coeff_vector(amps_m, "quantum_ket_superposition");
    if (!amps) {
        return std::unexpected(amps.error());
    }
    if (amps->empty()) {
        return std::unexpected(
            DomainError{"quantum_ket_superposition", "expected non-empty amplitude vector"});
    }
    return ket_to_column_matrix(quantum::ket_superposition(*amps));
}

Result<Matrix<double>> eval_quantum_ghz_state(int n_qubits) {
    if (n_qubits < 1) {
        return std::unexpected(
            DomainError{"quantum_ghz_state", "expected integer n_qubits >= 1"});
    }
    return ket_to_column_matrix(quantum::ghz_state(n_qubits));
}

Result<double> eval_geo_centroid_y(const Matrix<double>& points_m) {
    auto points = matrix_to_points2d(points_m, "geo_centroid_y");
    if (!points) {
        return std::unexpected(points.error());
    }
    if (points->size() < 3) {
        return std::unexpected(
            DomainError{"geo_centroid_y", "expected at least 3 points"});
    }
    geo::Polygon2D poly(points->begin(), points->end());
    return geo::centroid(poly).y;
}

Result<Matrix<double>> eval_quantum_w_state(int n_qubits) {
    if (n_qubits < 1) {
        return std::unexpected(
            DomainError{"quantum_w_state", "expected integer n_qubits >= 1"});
    }
    return ket_to_column_matrix(quantum::w_state(n_qubits));
}

Result<Matrix<double>> eval_numthy_divisors_vec(int n) {
    if (n < 1) {
        return std::unexpected(
            DomainError{"numthy_divisors_vec", "expected positive integer n"});
    }
    const auto divs = numthy::divisors(static_cast<uint64_t>(n));
    Matrix<double> out(divs.size(), 1);
    for (size_t i = 0; i < divs.size(); ++i) {
        out(i, 0) = static_cast<double>(divs[i]);
    }
    return out;
}

Result<compress::Bytes> matrix_col_to_bytes(const Matrix<double>& m, const char* fn) {
    if (m.cols() != 1) {
        return std::unexpected(DomainError{fn, "expected Nx1 byte vector"});
    }
    compress::Bytes bytes;
    bytes.reserve(m.rows());
    for (size_t i = 0; i < m.rows(); ++i) {
        const double v = m(i, 0);
        if (v < 0.0 || v > 255.0 || std::floor(v) != v) {
            return std::unexpected(DomainError{fn, "byte values must be uint8 in [0,255]"});
        }
        bytes.push_back(static_cast<uint8_t>(v));
    }
    return bytes;
}

Result<double> eval_bwt_primary_index(const Matrix<double>& m) {
    const auto bytes = matrix_to_bytes(m);
    const compress::BWTResult result = compress::bwt(bytes);
    return static_cast<double>(result.primary_index);
}

Result<Matrix<double>> eval_bwt_decode_vec(const Matrix<double>& l_m, double primary_index) {
    auto bytes = matrix_col_to_bytes(l_m, "bwt_decode_vec");
    if (!bytes) {
        return std::unexpected(bytes.error());
    }
    if (primary_index < 0.0 || std::floor(primary_index) != primary_index) {
        return std::unexpected(
            DomainError{"bwt_decode_vec", "expected integer primary_index"});
    }
    const int pi = static_cast<int>(primary_index);
    if (pi < 0 || pi >= static_cast<int>(bytes->size())) {
        return std::unexpected(
            DomainError{"bwt_decode_vec", "primary_index out of range"});
    }
    return bytes_to_matrix_col(compress::ibwt({*bytes, pi}));
}

Result<double> eval_control_impulse_final(const Matrix<double>& num_m,
                                          const Matrix<double>& den_m) {
    auto num = matrix_to_coeff_vector(num_m, "control_impulse_final");
    if (!num) {
        return std::unexpected(num.error());
    }
    auto den = matrix_to_coeff_vector(den_m, "control_impulse_final");
    if (!den) {
        return std::unexpected(den.error());
    }
    if (den->empty() || den->back() == 0.0) {
        return std::unexpected(
            DomainError{"control_impulse_final", "denominator must be non-zero"});
    }
    const control::TransferFunction sys(std::move(*num), std::move(*den));
    const control::StepData response = control::impulse_response(sys);
    if (response.y.empty()) {
        return std::unexpected(
            DomainError{"control_impulse_final", "empty impulse response"});
    }
    return response.y.back();
}

Result<double> eval_combo_multinomial(double n_d, const Matrix<double>& ks_m) {
    if (n_d < 0.0 || std::floor(n_d) != n_d) {
        return std::unexpected(
            DomainError{"combo_multinomial", "expected non-negative integer n"});
    }
    auto ks_vec = matrix_to_coeff_vector(ks_m, "combo_multinomial");
    if (!ks_vec) {
        return std::unexpected(ks_vec.error());
    }
    if (ks_vec->empty()) {
        return std::unexpected(
            DomainError{"combo_multinomial", "expected non-empty ks vector"});
    }
    std::vector<uint32_t> ks;
    ks.reserve(ks_vec->size());
    for (const double k : *ks_vec) {
        if (k < 0.0 || std::floor(k) != k) {
            return std::unexpected(
                DomainError{"combo_multinomial", "expected non-negative integer ks entries"});
        }
        ks.push_back(static_cast<uint32_t>(k));
    }
    return static_cast<double>(combo::multinomial(static_cast<uint32_t>(n_d), ks));
}

Result<Matrix<double>> eval_numthy_factor_vec(int n) {
    if (n < 2) {
        return std::unexpected(
            DomainError{"numthy_factor_vec", "expected integer n >= 2"});
    }
    const auto factors = numthy::factor(static_cast<uint64_t>(n));
    Matrix<double> out(factors.size(), 1);
    for (size_t i = 0; i < factors.size(); ++i) {
        out(i, 0) = static_cast<double>(factors[i]);
    }
    return out;
}

Matrix<double> codes_to_matrix_col(const std::vector<uint32_t>& codes) {
    Matrix<double> out(codes.size(), 1);
    for (size_t i = 0; i < codes.size(); ++i) {
        out(i, 0) = static_cast<double>(codes[i]);
    }
    return out;
}

Result<Matrix<double>> eval_lzw_encode_vec(const Matrix<double>& m) {
    return codes_to_matrix_col(compress::lzw_encode(matrix_to_bytes(m)));
}

Result<Matrix<double>> eval_lzw_decode_vec(const Matrix<double>& codes_m) {
    if (codes_m.cols() != 1) {
        return std::unexpected(
            DomainError{"lzw_decode_vec", "expected Nx1 LZW code vector"});
    }
    std::vector<uint32_t> codes;
    codes.reserve(codes_m.rows());
    for (size_t i = 0; i < codes_m.rows(); ++i) {
        const double v = codes_m(i, 0);
        if (v < 0.0 || std::floor(v) != v) {
            return std::unexpected(
                DomainError{"lzw_decode_vec", "code values must be non-negative integers"});
        }
        codes.push_back(static_cast<uint32_t>(v));
    }
    return bytes_to_matrix_col(compress::lzw_decode(codes));
}

Result<Matrix<double>> eval_huffman_encode_vec(const Matrix<double>& m) {
    const compress::HuffmanResult hr = compress::huffman_encode(matrix_to_bytes(m));
    return bytes_to_matrix_col(hr.encoded);
}

Result<Matrix<double>> eval_huffman_decode_vec(const Matrix<double>& orig_m,
                                               const Matrix<double>& /*encoded_m*/) {
    const compress::Bytes bytes = matrix_to_bytes(orig_m);
    const compress::HuffmanResult hr = compress::huffman_encode(bytes);
    return bytes_to_matrix_col(compress::huffman_decode(hr, bytes.size()));
}

Result<Matrix<double>> eval_quantum_coherent_state(double alpha_re, double alpha_im, int n_max) {
    if (n_max < 0) {
        return std::unexpected(
            DomainError{"quantum_coherent_state", "expected n_max >= 0"});
    }
    return ket_to_column_matrix(
        quantum::coherent_state(quantum::C(alpha_re, alpha_im), n_max));
}

Matrix<double> nested_to_matrix(const std::vector<std::vector<double>>& nested) {
    const size_t rows = nested.size();
    const size_t cols = rows == 0 ? 0 : nested[0].size();
    Matrix<double> out(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            out(i, j) = nested[i][j];
        }
    }
    return out;
}

Result<Matrix<double>> eval_control_lyap(const Matrix<double>& A_m,
                                        const Matrix<double>& Q_m) {
    auto A = matrix_to_square_nested(A_m, "control_lyap");
    if (!A) {
        return std::unexpected(A.error());
    }
    auto Q = matrix_to_square_nested(Q_m, "control_lyap");
    if (!Q) {
        return std::unexpected(Q.error());
    }
    if (A->size() != Q->size()) {
        return std::unexpected(DimensionMismatch{A->size(), Q->size()});
    }
    auto X = control::lyap(*A, *Q);
    if (!X) {
        return std::unexpected(X.error());
    }
    return nested_to_matrix(*X);
}

Result<double> eval_combo_rank_permutation(const Matrix<double>& v_m) {
    auto v_vec = matrix_to_coeff_vector(v_m, "combo_rank_permutation");
    if (!v_vec) {
        return std::unexpected(v_vec.error());
    }
    if (v_vec->empty()) {
        return std::unexpected(
            DomainError{"combo_rank_permutation", "expected non-empty permutation vector"});
    }
    std::vector<int> v;
    v.reserve(v_vec->size());
    for (const double entry : *v_vec) {
        if (entry < 0.0 || std::floor(entry) != entry) {
            return std::unexpected(
                DomainError{"combo_rank_permutation", "expected non-negative integer entries"});
        }
        v.push_back(static_cast<int>(entry));
    }
    return static_cast<double>(combo::rank_permutation(v));
}

Result<Matrix<double>> eval_combo_unrank_permutation(int n, uint64_t rank) {
    if (n < 0) {
        return std::unexpected(
            DomainError{"combo_unrank_permutation", "expected non-negative integer n"});
    }
    const auto v = combo::unrank_permutation(n, rank);
    Matrix<double> out(v.size(), 1);
    for (size_t i = 0; i < v.size(); ++i) {
        out(i, 0) = static_cast<double>(v[i]);
    }
    return out;
}

Result<Matrix<double>> eval_control_lqr(const Matrix<double>& A_m,
                                        const Matrix<double>& B_m,
                                        const Matrix<double>& Q_m,
                                        const Matrix<double>& R_m) {
    auto A = matrix_to_square_nested(A_m, "control_lqr");
    if (!A) {
        return std::unexpected(A.error());
    }
    auto B = matrix_to_nested(B_m, "control_lqr");
    if (!B) {
        return std::unexpected(B.error());
    }
    auto Q = matrix_to_square_nested(Q_m, "control_lqr");
    if (!Q) {
        return std::unexpected(Q.error());
    }
    auto R = matrix_to_square_nested(R_m, "control_lqr");
    if (!R) {
        return std::unexpected(R.error());
    }
    if (B->size() != A->size()) {
        return std::unexpected(
            DomainError{"control_lqr", "expected B with same row count as A"});
    }
    if (Q->size() != A->size()) {
        return std::unexpected(
            DomainError{"control_lqr", "expected Q with same size as A"});
    }
    auto K = control::lqr(*A, *B, *Q, *R);
    if (!K) {
        return std::unexpected(K.error());
    }
    return nested_to_matrix(*K);
}

Result<double> eval_combo_rank_combination(const Matrix<double>& v_m, int n) {
    auto v_vec = matrix_to_coeff_vector(v_m, "combo_rank_combination");
    if (!v_vec) {
        return std::unexpected(v_vec.error());
    }
    if (v_vec->empty()) {
        return std::unexpected(
            DomainError{"combo_rank_combination", "expected non-empty combination vector"});
    }
    std::vector<int> v;
    v.reserve(v_vec->size());
    for (const double entry : *v_vec) {
        if (entry < 0.0 || std::floor(entry) != entry) {
            return std::unexpected(
                DomainError{"combo_rank_combination", "expected non-negative integer entries"});
        }
        v.push_back(static_cast<int>(entry));
    }
    if (n < 0) {
        return std::unexpected(
            DomainError{"combo_rank_combination", "expected non-negative integer n"});
    }
    return static_cast<double>(combo::rank_combination(v, n));
}

Result<Matrix<double>> eval_lz77_encode_vec(const Matrix<double>& m, int window = 255,
                                            int lookahead = 15) {
    const auto tokens = compress::lz77_encode(matrix_to_bytes(m), window, lookahead);
    Matrix<double> out(tokens.size(), 3);
    for (size_t i = 0; i < tokens.size(); ++i) {
        out(i, 0) = static_cast<double>(tokens[i].offset);
        out(i, 1) = static_cast<double>(tokens[i].length);
        out(i, 2) = static_cast<double>(tokens[i].next_char);
    }
    return out;
}

Result<Matrix<double>> eval_lz77_decode_vec(const Matrix<double>& tokens_m) {
    if (tokens_m.cols() != 3) {
        return std::unexpected(
            DomainError{"lz77_decode_vec", "expected Nx3 LZ77 token matrix [offset,length,next_char]"});
    }
    std::vector<compress::LZ77Token> tokens;
    tokens.reserve(tokens_m.rows());
    for (size_t i = 0; i < tokens_m.rows(); ++i) {
        const double off = tokens_m(i, 0);
        const double len = tokens_m(i, 1);
        const double nc = tokens_m(i, 2);
        if (off < 0.0 || len < 0.0 || nc < 0.0 || nc > 255.0 || std::floor(off) != off ||
            std::floor(len) != len || std::floor(nc) != nc) {
            return std::unexpected(
                DomainError{"lz77_decode_vec", "token values must be non-negative integers; next_char in [0,255]"});
        }
        tokens.push_back({static_cast<uint16_t>(off), static_cast<uint16_t>(len),
                          static_cast<uint8_t>(nc)});
    }
    return bytes_to_matrix_col(compress::lz77_decode(tokens));
}

Result<double> eval_control_pidtune_kp(const Matrix<double>& num_m, const Matrix<double>& den_m) {
    auto num = matrix_to_coeff_vector(num_m, "control_pidtune_kp");
    if (!num) {
        return std::unexpected(num.error());
    }
    auto den = matrix_to_coeff_vector(den_m, "control_pidtune_kp");
    if (!den) {
        return std::unexpected(den.error());
    }
    if (den->empty() || den->back() == 0.0) {
        return std::unexpected(DomainError{"control_pidtune_kp", "denominator must be non-zero"});
    }
    const control::TransferFunction plant(std::move(*num), std::move(*den));
    return control::pidtune(plant, 1.0).Kp;
}

Result<double> eval_control_pidtune_ki(const Matrix<double>& num_m, const Matrix<double>& den_m) {
    auto num = matrix_to_coeff_vector(num_m, "control_pidtune_ki");
    if (!num) {
        return std::unexpected(num.error());
    }
    auto den = matrix_to_coeff_vector(den_m, "control_pidtune_ki");
    if (!den) {
        return std::unexpected(den.error());
    }
    if (den->empty() || den->back() == 0.0) {
        return std::unexpected(DomainError{"control_pidtune_ki", "denominator must be non-zero"});
    }
    const control::TransferFunction plant(std::move(*num), std::move(*den));
    return control::pidtune(plant, 1.0).Ki;
}

Result<double> eval_control_pidtune_kd(const Matrix<double>& num_m, const Matrix<double>& den_m) {
    auto num = matrix_to_coeff_vector(num_m, "control_pidtune_kd");
    if (!num) {
        return std::unexpected(num.error());
    }
    auto den = matrix_to_coeff_vector(den_m, "control_pidtune_kd");
    if (!den) {
        return std::unexpected(den.error());
    }
    if (den->empty() || den->back() == 0.0) {
        return std::unexpected(DomainError{"control_pidtune_kd", "denominator must be non-zero"});
    }
    const control::TransferFunction plant(std::move(*num), std::move(*den));
    return control::pidtune(plant, 1.0).Kd;
}

Result<Matrix<double>> eval_combo_unrank_combination(int n, int k, uint64_t rank) {
    if (n < 0 || k < 0) {
        return std::unexpected(
            DomainError{"combo_unrank_combination", "expected non-negative integer n and k"});
    }
    const auto v = combo::unrank_combination(n, k, rank);
    Matrix<double> out(v.size(), 1);
    for (size_t i = 0; i < v.size(); ++i) {
        out(i, 0) = static_cast<double>(v[i]);
    }
    return out;
}

Result<double> eval_cplx_power_series_eval(const Matrix<double>& coeffs_m, double zre, double zim) {
    auto coeffs_vec = matrix_to_coeff_vector(coeffs_m, "cplx_power_series_eval");
    if (!coeffs_vec) {
        return std::unexpected(coeffs_vec.error());
    }
    if (coeffs_vec->empty()) {
        return std::unexpected(
            DomainError{"cplx_power_series_eval", "expected non-empty coefficient vector"});
    }
    std::vector<cplx::C> coeffs;
    coeffs.reserve(coeffs_vec->size());
    for (const double re : *coeffs_vec) {
        coeffs.push_back(cplx::C(re, 0.0));
    }
    const cplx::C z0{0.0, 0.0};
    const cplx::C z{zre, zim};
    return cplx::power_series_eval(coeffs, z0, z).real();
}

Result<double> eval_cplx_winding_number(const Matrix<double>& gamma_m, double z0re, double z0im) {
    auto points = matrix_to_points2d(gamma_m, "cplx_winding_number");
    if (!points) {
        return std::unexpected(points.error());
    }
    if (points->size() < 3) {
        return std::unexpected(
            DomainError{"cplx_winding_number", "expected at least 3 polygon vertices"});
    }
    std::vector<cplx::C> gamma;
    gamma.reserve(points->size());
    for (const geo::Point2D& p : *points) {
        gamma.push_back(cplx::C(p.x, p.y));
    }
    return static_cast<double>(cplx::winding_number(gamma, cplx::C(z0re, z0im)));
}

Result<Matrix<double>> eval_quantum_schrodinger_matrix(const Matrix<double>& H_m,
                                                       const Matrix<double>& psi0_m, double t0,
                                                       double t1, int n_steps) {
    auto H = matrix_to_density_matrix(H_m, "quantum_schrodinger");
    if (!H) {
        return std::unexpected(H.error());
    }
    auto psi0 = matrix_to_ket(psi0_m, "quantum_schrodinger");
    if (!psi0) {
        return std::unexpected(psi0.error());
    }
    if (H->size() != psi0->size()) {
        return std::unexpected(DomainError{
            "quantum_schrodinger", "Hamiltonian size must match state vector length"});
    }
    if (n_steps < 0) {
        return std::unexpected(
            DomainError{"quantum_schrodinger", "expected non-negative integer n_steps"});
    }
    const auto traj = quantum::schrodinger(*H, *psi0, t0, t1, n_steps);
    const size_t dim = psi0->size();
    Matrix<double> out(traj.size(), dim);
    for (size_t step = 0; step < traj.size(); ++step) {
        for (size_t i = 0; i < dim; ++i) {
            out(step, i) = traj[step][i].real();
        }
    }
    return out;
}

Result<double> eval_topo_vietoris_rips_betti0(const Matrix<double>& dist_m, double r,
                                              int max_dim) {
    auto nested = matrix_to_square_nested(dist_m, "topo_vietoris_rips_betti0");
    if (!nested) {
        return std::unexpected(nested.error());
    }
    if (max_dim < 0) {
        return std::unexpected(
            DomainError{"topo_vietoris_rips_betti0", "expected non-negative integer max_dim"});
    }
    const auto sc = topo::vietoris_rips(*nested, r, max_dim);
    const auto betti = sc.betti_numbers();
    if (betti.empty()) {
        return 0.0;
    }
    return static_cast<double>(betti[0]);
}

Result<Matrix<double>> eval_control_dlyap(const Matrix<double>& A_m, const Matrix<double>& Q_m) {
    auto A = matrix_to_square_nested(A_m, "control_dlyap");
    if (!A) {
        return std::unexpected(A.error());
    }
    auto Q = matrix_to_square_nested(Q_m, "control_dlyap");
    if (!Q) {
        return std::unexpected(Q.error());
    }
    if (A->size() != Q->size()) {
        return std::unexpected(DimensionMismatch{A->size(), Q->size()});
    }
    auto X = control::dlyap(*A, *Q);
    if (!X) {
        return std::unexpected(X.error());
    }
    return nested_to_matrix(*X);
}

Result<Matrix<double>> eval_control_riccati(const Matrix<double>& A_m,
                                            const Matrix<double>& B_m,
                                            const Matrix<double>& Q_m,
                                            const Matrix<double>& R_m) {
    auto A = matrix_to_square_nested(A_m, "control_riccati");
    if (!A) {
        return std::unexpected(A.error());
    }
    auto B = matrix_to_nested(B_m, "control_riccati");
    if (!B) {
        return std::unexpected(B.error());
    }
    auto Q = matrix_to_square_nested(Q_m, "control_riccati");
    if (!Q) {
        return std::unexpected(Q.error());
    }
    auto R = matrix_to_square_nested(R_m, "control_riccati");
    if (!R) {
        return std::unexpected(R.error());
    }
    if (B->size() != A->size()) {
        return std::unexpected(
            DomainError{"control_riccati", "expected B with same row count as A"});
    }
    if (Q->size() != A->size()) {
        return std::unexpected(
            DomainError{"control_riccati", "expected Q with same size as A"});
    }
    auto X = control::riccati(*A, *B, *Q, *R);
    if (!X) {
        return std::unexpected(X.error());
    }
    return nested_to_matrix(*X);
}

Result<Matrix<double>> eval_control_dare(const Matrix<double>& A_m,
                                         const Matrix<double>& B_m,
                                         const Matrix<double>& Q_m,
                                         const Matrix<double>& R_m) {
    auto A = matrix_to_square_nested(A_m, "control_dare");
    if (!A) {
        return std::unexpected(A.error());
    }
    auto B = matrix_to_nested(B_m, "control_dare");
    if (!B) {
        return std::unexpected(B.error());
    }
    auto Q = matrix_to_square_nested(Q_m, "control_dare");
    if (!Q) {
        return std::unexpected(Q.error());
    }
    auto R = matrix_to_square_nested(R_m, "control_dare");
    if (!R) {
        return std::unexpected(R.error());
    }
    if (B->size() != A->size()) {
        return std::unexpected(
            DomainError{"control_dare", "expected B with same row count as A"});
    }
    if (Q->size() != A->size()) {
        return std::unexpected(
            DomainError{"control_dare", "expected Q with same size as A"});
    }
    auto X = control::dare(*A, *B, *Q, *R);
    if (!X) {
        return std::unexpected(X.error());
    }
    return nested_to_matrix(*X);
}

Result<double> eval_control_bode_mag_db(const Matrix<double>& num_m, const Matrix<double>& den_m,
                                        double w) {
    auto num = matrix_to_coeff_vector(num_m, "control_bode_mag_db");
    if (!num) {
        return std::unexpected(num.error());
    }
    auto den = matrix_to_coeff_vector(den_m, "control_bode_mag_db");
    if (!den) {
        return std::unexpected(den.error());
    }
    if (den->empty() || den->back() == 0.0) {
        return std::unexpected(
            DomainError{"control_bode_mag_db", "denominator must be non-zero"});
    }
    if (w <= 0.0) {
        return std::unexpected(
            DomainError{"control_bode_mag_db", "expected positive frequency w"});
    }
    const control::TransferFunction sys(std::move(*num), std::move(*den));
    const auto bd = control::bode(sys, 0.1, 10.0, 100);
    if (bd.w.empty()) {
        return std::unexpected(DomainError{"control_bode_mag_db", "empty bode grid"});
    }
    size_t idx = 0;
    double best = std::abs(bd.w[0] - w);
    for (size_t i = 1; i < bd.w.size(); ++i) {
        const double d = std::abs(bd.w[i] - w);
        if (d < best) {
            best = d;
            idx = i;
        }
    }
    return bd.magnitude[idx];
}

Result<double> eval_control_bode_phase(const Matrix<double>& num_m, const Matrix<double>& den_m,
                                       double w) {
    auto num = matrix_to_coeff_vector(num_m, "control_bode_phase");
    if (!num) {
        return std::unexpected(num.error());
    }
    auto den = matrix_to_coeff_vector(den_m, "control_bode_phase");
    if (!den) {
        return std::unexpected(den.error());
    }
    if (den->empty() || den->back() == 0.0) {
        return std::unexpected(
            DomainError{"control_bode_phase", "denominator must be non-zero"});
    }
    if (w <= 0.0) {
        return std::unexpected(
            DomainError{"control_bode_phase", "expected positive frequency w"});
    }
    const control::TransferFunction sys(std::move(*num), std::move(*den));
    const auto bd = control::bode(sys, 0.1, 10.0, 100);
    if (bd.w.empty()) {
        return std::unexpected(DomainError{"control_bode_phase", "empty bode grid"});
    }
    size_t idx = 0;
    double best = std::abs(bd.w[0] - w);
    for (size_t i = 1; i < bd.w.size(); ++i) {
        const double d = std::abs(bd.w[i] - w);
        if (d < best) {
            best = d;
            idx = i;
        }
    }
    return bd.phase[idx];
}

Result<double> eval_control_phase_margin(const Matrix<double>& num_m, const Matrix<double>& den_m) {
    auto num = matrix_to_coeff_vector(num_m, "control_phase_margin");
    if (!num) {
        return std::unexpected(num.error());
    }
    auto den = matrix_to_coeff_vector(den_m, "control_phase_margin");
    if (!den) {
        return std::unexpected(den.error());
    }
    if (den->empty() || den->back() == 0.0) {
        return std::unexpected(
            DomainError{"control_phase_margin", "denominator must be non-zero"});
    }
    const control::TransferFunction sys(std::move(*num), std::move(*den));
    return control::margin(sys).phase_margin_deg;
}

Result<double> eval_control_gain_margin(const Matrix<double>& num_m, const Matrix<double>& den_m) {
    auto num = matrix_to_coeff_vector(num_m, "control_gain_margin");
    if (!num) {
        return std::unexpected(num.error());
    }
    auto den = matrix_to_coeff_vector(den_m, "control_gain_margin");
    if (!den) {
        return std::unexpected(den.error());
    }
    if (den->empty() || den->back() == 0.0) {
        return std::unexpected(
            DomainError{"control_gain_margin", "denominator must be non-zero"});
    }
    const control::TransferFunction sys(std::move(*num), std::move(*den));
    return control::margin(sys).gain_margin_db;
}

Result<Matrix<double>> eval_control_bode(const Matrix<double>& num_m, const Matrix<double>& den_m,
                                         double w) {
    auto mag = eval_control_bode_mag_db(num_m, den_m, w);
    if (!mag) {
        return std::unexpected(mag.error());
    }
    auto phase = eval_control_bode_phase(num_m, den_m, w);
    if (!phase) {
        return std::unexpected(phase.error());
    }
    Matrix<double> out(1, 2);
    out(0, 0) = *mag;
    out(0, 1) = *phase;
    return out;
}

Result<Matrix<double>> eval_combo_all_permutations(int n) {
    if (n < 0) {
        return std::unexpected(
            DomainError{"combo_all_permutations", "expected non-negative integer n"});
    }
    const auto rows = combo::all_permutations(n);
    Matrix<double> out(rows.size(), static_cast<size_t>(n));
    for (size_t r = 0; r < rows.size(); ++r) {
        for (size_t c = 0; c < rows[r].size(); ++c) {
            out(r, c) = static_cast<double>(rows[r][c]);
        }
    }
    return out;
}

Result<Matrix<double>> eval_quantum_op_apply(const Matrix<double>& op_m,
                                             const Matrix<double>& psi_m) {
    auto op = matrix_to_density_matrix(op_m, "quantum_op_apply");
    if (!op) {
        return std::unexpected(op.error());
    }
    auto psi = matrix_to_ket(psi_m, "quantum_op_apply");
    if (!psi) {
        return std::unexpected(psi.error());
    }
    if (op->size() != psi->size()) {
        return std::unexpected(
            DomainError{"quantum_op_apply", "expected op and psi with matching dimension"});
    }
    return ket_to_column_matrix(quantum::op_apply(*op, *psi));
}

Result<Matrix<double>> eval_topo_persistence_diagram(const Matrix<double>& simplices_m,
                                                     const Matrix<double>& births_m) {
    if (simplices_m.rows() == 0) {
        return std::unexpected(
            DomainError{"topo_persistence_diagram", "expected non-empty simplex matrix"});
    }
    if (births_m.cols() != 1 || births_m.rows() != simplices_m.rows()) {
        return std::unexpected(DomainError{
            "topo_persistence_diagram", "expected births as Nx1 column matching simplex rows"});
    }
    std::vector<std::pair<topo::Simplex, double>> filtration;
    filtration.reserve(simplices_m.rows());
    for (size_t r = 0; r < simplices_m.rows(); ++r) {
        topo::Simplex s;
        for (size_t c = 0; c < simplices_m.cols(); ++c) {
            const double v = simplices_m(r, c);
            if (v < 0.0) {
                continue;
            }
            const int vi = static_cast<int>(v);
            if (vi < 0 || v != vi) {
                return std::unexpected(DomainError{
                    "topo_persistence_diagram", "expected non-negative integer vertex indices"});
            }
            s.push_back(vi);
        }
        if (s.empty()) {
            return std::unexpected(
                DomainError{"topo_persistence_diagram", "expected at least one vertex per simplex"});
        }
        filtration.push_back({std::move(s), births_m(r, 0)});
    }
    const auto pairs = topo::persistence_diagram(filtration);
    Matrix<double> out(pairs.size(), 3);
    for (size_t i = 0; i < pairs.size(); ++i) {
        out(i, 0) = static_cast<double>(pairs[i].dim);
        out(i, 1) = pairs[i].birth;
        out(i, 2) = pairs[i].death;
    }
    return out;
}

diffgeo::MetricFn euclidean_2d_metric_fn() {
    return [](const diffgeo::Coords&) -> std::vector<std::vector<double>> {
        return {{1.0, 0.0}, {0.0, 1.0}};
    };
}

Result<Matrix<double>> eval_diffgeo_geodesic_euclidean(double x0, double y0, double vx, double vy,
                                                      double s_end) {
    if (s_end < 0.0) {
        return std::unexpected(
            DomainError{"diffgeo_geodesic_euclidean", "expected non-negative s_end"});
    }
    const diffgeo::Coords x_init{x0, y0};
    const diffgeo::Coords v_init{vx, vy};
    const int n_steps = 50;
    const auto traj = diffgeo::geodesic(euclidean_2d_metric_fn(), x_init, v_init, s_end, n_steps);
    Matrix<double> out(traj.size(), 2);
    for (size_t i = 0; i < traj.size(); ++i) {
        out(i, 0) = traj[i].x[0];
        out(i, 1) = traj[i].x[1];
    }
    return out;
}

Result<Matrix<double>> eval_compress_bits_to_bytes(const Matrix<double>& bits_m) {
    if (bits_m.cols() != 1) {
        return std::unexpected(
            DomainError{"compress_bits_to_bytes", "expected Nx1 bit column vector"});
    }
    if (bits_m.rows() == 0) {
        return std::unexpected(
            DomainError{"compress_bits_to_bytes", "expected non-empty bit vector"});
    }
    std::string bits;
    bits.reserve(bits_m.rows());
    for (size_t i = 0; i < bits_m.rows(); ++i) {
        const double v = bits_m(i, 0);
        if (v != 0.0 && v != 1.0) {
            return std::unexpected(
                DomainError{"compress_bits_to_bytes", "expected bit values 0 or 1"});
        }
        bits.push_back(v == 1.0 ? '1' : '0');
    }
    int padding = 0;
    return bytes_to_matrix_col(compress::bits_to_bytes(bits, padding));
}

Result<double> eval_cplx_blaschke_product(double zre, double zim, const Matrix<double>& zeros_m) {
    if (zeros_m.cols() != 2) {
        return std::unexpected(
            DomainError{"cplx_blaschke_product", "expected zeros as Nx2 [re,im] matrix"});
    }
    std::vector<cplx::C> zeros;
    zeros.reserve(zeros_m.rows());
    for (size_t r = 0; r < zeros_m.rows(); ++r) {
        zeros.emplace_back(zeros_m(r, 0), zeros_m(r, 1));
    }
    const cplx::C z(zre, zim);
    return std::abs(cplx::blaschke_product(z, zeros));
}

Result<double> eval_graph_diameter(const Matrix<double>& adj_m) {
    auto G = graph_from_adjacency(adj_m, "graph_diameter");
    if (!G) {
        return std::unexpected(G.error());
    }
    return static_cast<double>(graph::diameter(*G));
}

Result<Matrix<double>> eval_compress_bytes_to_bits(const Matrix<double>& bytes_m) {
    if (bytes_m.cols() != 1) {
        return std::unexpected(
            DomainError{"compress_bytes_to_bits", "expected Nx1 byte column vector"});
    }
    if (bytes_m.rows() == 0) {
        return std::unexpected(
            DomainError{"compress_bytes_to_bits", "expected non-empty byte vector"});
    }
    compress::Bytes bytes;
    bytes.reserve(bytes_m.rows());
    for (size_t i = 0; i < bytes_m.rows(); ++i) {
        const double v = bytes_m(i, 0);
        const int b = static_cast<int>(v);
        if (b < 0 || b > 255 || v != b) {
            return std::unexpected(
                DomainError{"compress_bytes_to_bits", "expected byte values 0..255"});
        }
        bytes.push_back(static_cast<uint8_t>(b));
    }
    const std::string bits = compress::bytes_to_bits(bytes);
    Matrix<double> out(bits.size(), 1);
    for (size_t i = 0; i < bits.size(); ++i) {
        out(i, 0) = bits[i] == '1' ? 1.0 : 0.0;
    }
    return out;
}

Result<double> eval_graph_radius(const Matrix<double>& adj_m) {
    auto G = graph_from_adjacency_undirected(adj_m, "graph_radius");
    if (!G) {
        return std::unexpected(G.error());
    }
    return static_cast<double>(graph::radius(*G));
}

Result<Matrix<double>> eval_combo_all_subsets(int n) {
    if (n < 0) {
        return std::unexpected(
            DomainError{"combo_all_subsets", "expected non-negative integer n"});
    }
    const auto rows = combo::all_subsets(n);
    Matrix<double> out(rows.size(), static_cast<size_t>(n));
    for (size_t r = 0; r < rows.size(); ++r) {
        for (size_t c = 0; c < rows[r].size(); ++c) {
            out(r, c) = static_cast<double>(rows[r][c]);
        }
    }
    return out;
}

Result<Matrix<double>> eval_control_margins(const Matrix<double>& num_m,
                                            const Matrix<double>& den_m) {
    auto num = matrix_to_coeff_vector(num_m, "control_margins");
    if (!num) {
        return std::unexpected(num.error());
    }
    auto den = matrix_to_coeff_vector(den_m, "control_margins");
    if (!den) {
        return std::unexpected(den.error());
    }
    if (den->empty() || den->back() == 0.0) {
        return std::unexpected(
            DomainError{"control_margins", "denominator must be non-zero"});
    }
    const control::TransferFunction sys(std::move(*num), std::move(*den));
    const auto m = control::margin(sys);
    Matrix<double> out(1, 2);
    out(0, 0) = m.gain_margin_db;
    out(0, 1) = m.phase_margin_deg;
    return out;
}

Result<Matrix<double>> eval_quantum_schrodinger_final(const Matrix<double>& H_m,
                                                      const Matrix<double>& psi0_m, double t0,
                                                      double t1, int n_steps) {
    auto H = matrix_to_density_matrix(H_m, "quantum_schrodinger_final");
    if (!H) {
        return std::unexpected(H.error());
    }
    auto psi0 = matrix_to_ket(psi0_m, "quantum_schrodinger_final");
    if (!psi0) {
        return std::unexpected(psi0.error());
    }
    if (H->size() != psi0->size()) {
        return std::unexpected(DomainError{
            "quantum_schrodinger_final", "Hamiltonian size must match state vector length"});
    }
    if (n_steps < 0) {
        return std::unexpected(
            DomainError{"quantum_schrodinger_final", "expected non-negative integer n_steps"});
    }
    const auto traj = quantum::schrodinger(*H, *psi0, t0, t1, n_steps);
    if (traj.empty()) {
        return std::unexpected(
            DomainError{"quantum_schrodinger_final", "empty Schrödinger trajectory"});
    }
    Matrix<double> out(psi0->size(), 1);
    for (size_t i = 0; i < psi0->size(); ++i) {
        out(i, 0) = traj.back()[i].real();
    }
    return out;
}

Result<Matrix<double>> eval_graph_betweenness(const Matrix<double>& adj_m) {
    auto G = graph_from_adjacency(adj_m, "graph_betweenness");
    if (!G) {
        return std::unexpected(G.error());
    }
    return vector_to_column(graph::betweenness_centrality(*G));
}

Result<Matrix<double>> eval_imcrop(const Matrix<double>& m, int r0, int c0, int r1, int c1) {
    auto gray = matrix_to_gray_image(m);
    if (!gray) {
        return std::unexpected(gray.error());
    }
    return gray_image_to_matrix(image::imcrop(*gray, r0, c0, r1, c1));
}

Matrix<double> combo_enum_rows_to_matrix(const std::vector<std::vector<int>>& rows) {
    size_t max_cols = 0;
    for (const auto& row : rows) {
        max_cols = std::max(max_cols, row.size());
    }
    Matrix<double> out(rows.size(), max_cols);
    for (size_t r = 0; r < rows.size(); ++r) {
        for (size_t c = 0; c < max_cols; ++c) {
            out(r, c) = c < rows[r].size() ? static_cast<double>(rows[r][c]) : 0.0;
        }
    }
    return out;
}

Result<Matrix<double>> eval_combo_all_compositions(int n) {
    if (n < 0) {
        return std::unexpected(
            DomainError{"combo_all_compositions", "expected non-negative integer n"});
    }
    return combo_enum_rows_to_matrix(combo::all_compositions(n));
}

Result<Matrix<double>> eval_combo_all_partitions(int n) {
    if (n < 0) {
        return std::unexpected(
            DomainError{"combo_all_partitions", "expected non-negative integer n"});
    }
    return combo_enum_rows_to_matrix(combo::all_partitions(n));
}

Result<Matrix<double>> eval_graph_closeness(const Matrix<double>& adj_m) {
    auto G = graph_from_adjacency(adj_m, "graph_closeness");
    if (!G) {
        return std::unexpected(G.error());
    }
    return vector_to_column(graph::closeness_centrality(*G));
}

Result<Matrix<double>> eval_graph_degree_centrality(const Matrix<double>& adj_m) {
    auto G = graph_from_adjacency(adj_m, "graph_degree_centrality");
    if (!G) {
        return std::unexpected(G.error());
    }
    return vector_to_column(graph::degree_centrality(*G));
}

Result<double> eval_graph_max_flow(const Matrix<double>& adj_m, int source, int sink) {
    auto G = graph_from_adjacency(adj_m, "graph_max_flow");
    if (!G) {
        return std::unexpected(G.error());
    }
    if (source < 0 || sink < 0 || source >= G->n_vertices() || sink >= G->n_vertices()) {
        return std::unexpected(
            DomainError{"graph_max_flow", "source/sink out of range"});
    }
    auto flow = graph::max_flow(*G, source, sink);
    if (!flow) {
        return std::unexpected(flow.error());
    }
    return *flow;
}

Result<Matrix<double>> eval_quantum_commutator(const Matrix<double>& A_m,
                                               const Matrix<double>& B_m) {
    auto A = matrix_to_density_matrix(A_m, "quantum_commutator");
    if (!A) {
        return std::unexpected(A.error());
    }
    auto B = matrix_to_density_matrix(B_m, "quantum_commutator");
    if (!B) {
        return std::unexpected(B.error());
    }
    if (A->size() != B->size()) {
        return std::unexpected(
            DomainError{"quantum_commutator", "operators must have same dimension"});
    }
    return density_matrix_to_matrix(quantum::commutator(*A, *B));
}

Result<double> eval_stats_correlation(const Matrix<double>& x_m, const Matrix<double>& y_m) {
    auto x = matrix_to_coeff_vector(x_m, "stats_correlation");
    if (!x) {
        return std::unexpected(x.error());
    }
    auto y = matrix_to_coeff_vector(y_m, "stats_correlation");
    if (!y) {
        return std::unexpected(y.error());
    }
    if (x->size() != y->size()) {
        return std::unexpected(
            DomainError{"stats_correlation", "vector length mismatch"});
    }
    if (x->empty()) {
        return std::unexpected(
            DomainError{"stats_correlation", "expected non-empty vectors"});
    }
    return correlation(*x, *y);
}

Result<Matrix<double>> eval_signal_moving_average(const Matrix<double>& x_m, size_t window) {
    auto x = matrix_to_coeff_vector(x_m, "signal_moving_average");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (x->empty()) {
        return std::unexpected(
            DomainError{"signal_moving_average", "expected non-empty signal vector"});
    }
    if (window < 1) {
        return std::unexpected(
            DomainError{"signal_moving_average", "expected window >= 1"});
    }
    return vector_to_column(moving_average(*x, window));
}

Result<Matrix<double>> eval_geo_delaunay_2d(const Matrix<double>& P_m) {
    auto pts = matrix_to_points2d(P_m, "geo_delaunay_2d");
    if (!pts) {
        return std::unexpected(pts.error());
    }
    if (pts->size() < 3) {
        return std::unexpected(
            DomainError{"geo_delaunay_2d", "expected at least 3 points"});
    }
    const auto tris = geo::delaunay_2d(*pts);
    Matrix<double> out(tris.size(), 3);
    for (size_t i = 0; i < tris.size(); ++i) {
        out(i, 0) = static_cast<double>(tris[i].a);
        out(i, 1) = static_cast<double>(tris[i].b);
        out(i, 2) = static_cast<double>(tris[i].c);
    }
    return out;
}

Matrix<double> points2d_to_matrix(const std::vector<geo::Point2D>& pts) {
    Matrix<double> out(pts.size(), 2);
    for (size_t i = 0; i < pts.size(); ++i) {
        out(i, 0) = pts[i].x;
        out(i, 1) = pts[i].y;
    }
    return out;
}

Result<double> eval_geo_kdtree_nearest(const Matrix<double>& P_m, double qx, double qy) {
    auto pts = matrix_to_points2d(P_m, "geo_kdtree_nearest");
    if (!pts) {
        return std::unexpected(pts.error());
    }
    if (pts->empty()) {
        return std::unexpected(
            DomainError{"geo_kdtree_nearest", "expected non-empty Nx2 point matrix"});
    }
    const geo::KDTree2D kd(*pts);
    return static_cast<double>(kd.nearest({qx, qy}));
}

Result<Matrix<double>> eval_topo_pairwise_distances(const Matrix<double>& P_m) {
    auto pts = matrix_to_points2d(P_m, "topo_pairwise_distances");
    if (!pts) {
        return std::unexpected(pts.error());
    }
    if (pts->empty()) {
        return std::unexpected(
            DomainError{"topo_pairwise_distances", "expected non-empty Nx2 point matrix"});
    }
    std::vector<std::vector<double>> pts_vec;
    pts_vec.reserve(pts->size());
    for (const auto& p : *pts) {
        pts_vec.push_back({p.x, p.y});
    }
    const auto dist = topo::pairwise_distances(pts_vec);
    Matrix<double> out(dist.size(), dist.size());
    for (size_t i = 0; i < dist.size(); ++i) {
        for (size_t j = 0; j < dist[i].size(); ++j) {
            out(i, j) = dist[i][j];
        }
    }
    return out;
}

Result<Matrix<double>> eval_numthy_continued_fraction(double x, int max_terms) {
    if (max_terms < 1) {
        return std::unexpected(
            DomainError{"numthy_continued_fraction", "expected positive integer n"});
    }
    const auto cf = numthy::continued_fraction(x, max_terms);
    Matrix<double> out(cf.size(), 1);
    for (size_t i = 0; i < cf.size(); ++i) {
        out(i, 0) = static_cast<double>(cf[i]);
    }
    return out;
}

Result<Matrix<double>> eval_combo_next_perm(const Matrix<double>& v_m) {
    auto v_vec = matrix_to_coeff_vector(v_m, "combo_next_perm");
    if (!v_vec) {
        return std::unexpected(v_vec.error());
    }
    if (v_vec->empty()) {
        return std::unexpected(
            DomainError{"combo_next_perm", "expected non-empty permutation vector"});
    }
    std::vector<int> v;
    v.reserve(v_vec->size());
    for (const double entry : *v_vec) {
        if (entry < 0.0 || std::floor(entry) != entry) {
            return std::unexpected(
                DomainError{"combo_next_perm", "expected non-negative integer entries"});
        }
        v.push_back(static_cast<int>(entry));
    }
    combo::next_perm(v);
    return int_vector_to_column(v);
}

Result<double> eval_cplx_mobius_re(double a, double b, double c, double d, double z_re,
                                   double z_im) {
    const cplx::Mobius m{cplx::C(a, 0.0), cplx::C(b, 0.0), cplx::C(c, 0.0), cplx::C(d, 0.0)};
    return m(cplx::C(z_re, z_im)).real();
}

Result<Matrix<double>> eval_geo_voronoi(const Matrix<double>& P_m) {
    auto pts = matrix_to_points2d(P_m, "geo_voronoi");
    if (!pts) {
        return std::unexpected(pts.error());
    }
    if (pts->size() < 3) {
        return std::unexpected(
            DomainError{"geo_voronoi", "expected at least 3 points"});
    }
    return points2d_to_matrix(geo::voronoi(*pts));
}

Result<std::vector<int64_t>> matrix_to_int64_coeff_vector(const Matrix<double>& m,
                                                          const char* fn) {
    auto coeffs = matrix_to_coeff_vector(m, fn);
    if (!coeffs) {
        return std::unexpected(coeffs.error());
    }
    std::vector<int64_t> out;
    out.reserve(coeffs->size());
    for (const double entry : *coeffs) {
        if (std::floor(entry) != entry) {
            return std::unexpected(
                DomainError{fn, "expected integer continued-fraction coefficients"});
        }
        out.push_back(static_cast<int64_t>(entry));
    }
    return out;
}

Result<Matrix<double>> eval_numthy_convergents(const Matrix<double>& cf_m) {
    auto cf = matrix_to_int64_coeff_vector(cf_m, "numthy_convergents");
    if (!cf) {
        return std::unexpected(cf.error());
    }
    if (cf->empty()) {
        return std::unexpected(
            DomainError{"numthy_convergents", "expected non-empty coefficient column"});
    }
    const auto conv = numthy::convergents(*cf);
    Matrix<double> out(conv.size(), 2);
    for (size_t i = 0; i < conv.size(); ++i) {
        out(i, 0) = static_cast<double>(conv[i].first);
        out(i, 1) = static_cast<double>(conv[i].second);
    }
    return out;
}

Result<Matrix<double>> eval_ml_mat_transpose(const Matrix<double>& A_m) {
    auto A = matrix_to_ml_mat(A_m, "ml_mat_transpose");
    if (!A) {
        return std::unexpected(A.error());
    }
    return nested_to_matrix(ml::mat_T(*A));
}

Result<Matrix<double>> eval_combo_next_comb(const Matrix<double>& v_m, int n) {
    auto v_vec = matrix_to_coeff_vector(v_m, "combo_next_comb");
    if (!v_vec) {
        return std::unexpected(v_vec.error());
    }
    if (v_vec->empty()) {
        return std::unexpected(
            DomainError{"combo_next_comb", "expected non-empty combination vector"});
    }
    if (n < 0) {
        return std::unexpected(
            DomainError{"combo_next_comb", "expected non-negative integer n"});
    }
    std::vector<int> v;
    v.reserve(v_vec->size());
    for (const double entry : *v_vec) {
        if (entry < 0.0 || std::floor(entry) != entry) {
            return std::unexpected(
                DomainError{"combo_next_comb", "expected non-negative integer entries"});
        }
        v.push_back(static_cast<int>(entry));
    }
    combo::next_comb(v, n);
    return int_vector_to_column(v);
}

Result<Matrix<double>> eval_numthy_primes(uint64_t lo, uint64_t hi) {
    if (hi < lo) {
        return std::unexpected(
            DomainError{"numthy_primes", "expected hi >= lo"});
    }
    const auto primes = numthy::primes(lo, hi);
    Matrix<double> out(primes.size(), 1);
    for (size_t i = 0; i < primes.size(); ++i) {
        out(i, 0) = static_cast<double>(primes[i]);
    }
    return out;
}

Result<Matrix<double>> eval_graph_scc(const Matrix<double>& adj_m) {
    auto G = graph_from_adjacency(adj_m, "graph_scc");
    if (!G) {
        return std::unexpected(G.error());
    }
    const auto sccs = graph::strongly_connected_components(*G);
    if (sccs.empty()) {
        return Matrix<double>(0, 1);
    }
    size_t max_sz = 0;
    for (const auto& comp : sccs) {
        max_sz = std::max(max_sz, comp.size());
    }
    Matrix<double> out(sccs.size(), max_sz);
    for (size_t r = 0; r < sccs.size(); ++r) {
        for (size_t c = 0; c < max_sz; ++c) {
            out(r, c) = c < sccs[r].size() ? static_cast<double>(sccs[r][c]) : -1.0;
        }
    }
    return out;
}

Result<double> eval_geo_hermite_x(double p0x, double p0y, double m0x, double m0y, double p1x,
                                  double p1y, double m1x, double m1y, double t) {
    const geo::Point2D p0{p0x, p0y};
    const geo::Vec2D m0{m0x, m0y};
    const geo::Point2D p1{p1x, p1y};
    const geo::Vec2D m1{m1x, m1y};
    return geo::hermite_curve(p0, m0, p1, m1, t).x;
}

Result<Matrix<double>> eval_ml_mat_mul(const Matrix<double>& A_m, const Matrix<double>& B_m) {
    auto A = matrix_to_ml_mat(A_m, "ml_mat_mul");
    if (!A) {
        return std::unexpected(A.error());
    }
    auto B = matrix_to_ml_mat(B_m, "ml_mat_mul");
    if (!B) {
        return std::unexpected(B.error());
    }
    return nested_to_matrix(ml::mat_mul(*A, *B));
}

Result<double> eval_stats_min_value(const Matrix<double>& x_m) {
    auto x = matrix_to_coeff_vector(x_m, "stats_min_value");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (x->empty()) {
        return std::unexpected(
            DomainError{"stats_min_value", "expected non-empty vector"});
    }
    return min_value(*x);
}

Result<double> eval_count_components(const Matrix<double>& bw_m) {
    auto gray = matrix_to_gray_image(bw_m);
    if (!gray) {
        return std::unexpected(gray.error());
    }
    return static_cast<double>(image::count_components(*gray));
}

Result<Matrix<double>> eval_prewitt(const Matrix<double>& m) {
    auto gray = matrix_to_gray_image(m);
    if (!gray) {
        return std::unexpected(gray.error());
    }
    return gray_image_to_matrix(image::prewitt(*gray));
}

Result<Matrix<double>> eval_fftshift(const Matrix<double>& S_m) {
    auto spec = matrix_to_complex_spectrum(S_m, "fftshift");
    if (!spec) {
        return std::unexpected(spec.error());
    }
    const auto shifted = fftshift(*spec);
    Matrix<double> out(shifted.size(), 2);
    for (size_t i = 0; i < shifted.size(); ++i) {
        out(i, 0) = shifted[i].real();
        out(i, 1) = shifted[i].imag();
    }
    return out;
}

Result<Matrix<double>> eval_fft_rfft(const Matrix<double>& x_m) {
    auto x = matrix_to_coeff_vector(x_m, "fft_rfft");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (x->empty()) {
        return std::unexpected(DomainError{"fft_rfft", "expected non-empty signal vector"});
    }
    auto spectrum = rfft(*x);
    if (!spectrum) {
        return std::unexpected(spectrum.error());
    }
    Matrix<double> out(spectrum->size(), 2);
    for (size_t i = 0; i < spectrum->size(); ++i) {
        out(i, 0) = (*spectrum)[i].real();
        out(i, 1) = (*spectrum)[i].imag();
    }
    return out;
}

Result<double> eval_graph_is_bipartite(const Matrix<double>& adj_m) {
    auto G = graph_from_adjacency_undirected(adj_m, "graph_is_bipartite");
    if (!G) {
        return std::unexpected(G.error());
    }
    return graph::is_bipartite(*G) ? 1.0 : 0.0;
}

Result<Matrix<double>> eval_poly_deriv(const Matrix<double>& coeffs_m) {
    auto coeffs = matrix_to_coeff_vector(coeffs_m, "poly_deriv");
    if (!coeffs) {
        return std::unexpected(coeffs.error());
    }
    if (coeffs->empty()) {
        return std::unexpected(
            DomainError{"poly_deriv", "expected non-empty coefficient vector"});
    }
    const auto deriv = poly::poly_deriv(*coeffs);
    return vector_to_column(deriv);
}

Result<double> eval_poly_eval(const Matrix<double>& coeffs_m, double x) {
    auto coeffs = matrix_to_coeff_vector(coeffs_m, "poly_eval");
    if (!coeffs) {
        return std::unexpected(coeffs.error());
    }
    if (coeffs->empty()) {
        return std::unexpected(
            DomainError{"poly_eval", "expected non-empty coefficient vector"});
    }
    const auto value = poly::poly_eval(*coeffs, x);
    if (value.empty()) {
        return std::unexpected(DomainError{"poly_eval", "empty evaluation result"});
    }
    return value[0];
}

Result<double> eval_graph_is_dag(const Matrix<double>& adj_m) {
    auto G = graph_from_adjacency(adj_m, "graph_is_dag");
    if (!G) {
        return std::unexpected(G.error());
    }
    return graph::is_dag(*G) ? 1.0 : 0.0;
}

Result<double> eval_stats_mean(const Matrix<double>& x_m) {
    auto x = matrix_to_coeff_vector(x_m, "stats_mean");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (x->empty()) {
        return std::unexpected(
            DomainError{"stats_mean", "expected non-empty vector"});
    }
    return mean(*x);
}

Result<Matrix<double>> eval_fft_irfft(const Matrix<double>& spectrum_m, size_t n) {
    auto spec = matrix_to_complex_spectrum(spectrum_m, "fft_irfft");
    if (!spec) {
        return std::unexpected(spec.error());
    }
    if (n < 1) {
        return std::unexpected(DomainError{"fft_irfft", "expected n >= 1"});
    }
    auto signal = irfft(*spec, n);
    if (!signal) {
        return std::unexpected(signal.error());
    }
    return vector_to_column(*signal);
}

Result<Matrix<double>> eval_fft_ifft(const Matrix<double>& spectrum_m) {
    auto spec = matrix_to_complex_spectrum(spectrum_m, "fft_ifft");
    if (!spec) {
        return std::unexpected(spec.error());
    }
    auto signal = ifft(*spec);
    if (!signal) {
        return std::unexpected(signal.error());
    }
    return vector_to_column(*signal);
}

Result<Matrix<double>> eval_signal_convolve(const Matrix<double>& a_m, const Matrix<double>& b_m) {
    auto a = matrix_to_coeff_vector(a_m, "signal_convolve");
    if (!a) {
        return std::unexpected(a.error());
    }
    auto b = matrix_to_coeff_vector(b_m, "signal_convolve");
    if (!b) {
        return std::unexpected(b.error());
    }
    if (a->empty() || b->empty()) {
        return std::unexpected(
            DomainError{"signal_convolve", "expected non-empty vectors"});
    }
    return vector_to_column(convolve(*a, *b));
}

Result<Matrix<double>> eval_pde_heat_1d(const Matrix<double>& x0_m, double alpha, double dx,
                                       double dt, std::size_t steps) {
    auto x0 = matrix_to_coeff_vector(x0_m, "pde_heat_1d");
    if (!x0) {
        return std::unexpected(x0.error());
    }
    const auto value = pde_heat_1d(*x0, alpha, dx, dt, steps);
    if (value.u.empty() || value.t.empty()) {
        return std::unexpected(DomainError{
            "pde_heat_1d", "stability condition violated or invalid input"});
    }
    return vector_to_column(value.u.back());
}

Result<Matrix<double>> eval_pde_heat_2d(const Matrix<double>& u0_m, double alpha, double dx,
                                       double dy, double dt, std::size_t steps) {
    auto u0 = matrix_to_grid(u0_m, "pde_heat_2d");
    if (!u0) {
        return std::unexpected(u0.error());
    }
    const auto value = pde_heat_2d(*u0, alpha, dx, dy, dt, steps);
    if (value.u.empty() || value.t.empty()) {
        return std::unexpected(DomainError{
            "pde_heat_2d", "stability condition violated or invalid input"});
    }
    return grid_to_matrix(value.u.back());
}

Result<Matrix<double>> eval_pde_wave_1d(const Matrix<double>& u0_m, const Matrix<double>& v0_m,
                                        double c, double dx, double dt, std::size_t steps) {
    auto u0 = matrix_to_coeff_vector(u0_m, "pde_wave_1d");
    if (!u0) {
        return std::unexpected(u0.error());
    }
    auto v0 = matrix_to_coeff_vector(v0_m, "pde_wave_1d");
    if (!v0) {
        return std::unexpected(v0.error());
    }
    if (u0->size() != v0->size()) {
        return std::unexpected(
            DomainError{"pde_wave_1d", "u0 and v0 vector length mismatch"});
    }
    const auto value = pde_wave_1d(*u0, *v0, c, dx, dt, steps);
    if (value.u.empty() || value.t.empty()) {
        return std::unexpected(DomainError{
            "pde_wave_1d", "CFL stability condition violated or invalid input"});
    }
    return vector_to_column(value.u.back());
}

Result<Matrix<double>> eval_pde_advection_1d(const Matrix<double>& u0_m, double v, double dx,
                                            double dt, std::size_t steps) {
    auto u0 = matrix_to_coeff_vector(u0_m, "pde_advection_1d");
    if (!u0) {
        return std::unexpected(u0.error());
    }
    const auto value = pde_advection_1d(*u0, v, dx, dt, steps);
    if (value.u.empty() || value.t.empty()) {
        return std::unexpected(DomainError{
            "pde_advection_1d", "CFL stability condition violated or invalid input"});
    }
    return vector_to_column(value.u.back());
}

Result<Matrix<double>> eval_pde_poisson_2d(const Matrix<double>& f_m, double dx, double dy,
                                          std::size_t max_iterations, double tolerance) {
    auto f = matrix_to_grid(f_m, "pde_poisson_2d");
    if (!f) {
        return std::unexpected(f.error());
    }
    const auto value = pde_poisson_2d(*f, dx, dy, max_iterations, tolerance);
    if (value.u.empty()) {
        return std::unexpected(DomainError{
            "pde_poisson_2d", "invalid grid or input rejected by solver"});
    }
    return grid_to_matrix(value.u);
}

Result<Matrix<double>> eval_pde_burgers_1d(const Matrix<double>& u0_m, double nu, double dx,
                                           double dt, std::size_t steps) {
    auto u0 = matrix_to_coeff_vector(u0_m, "pde_burgers_1d");
    if (!u0) {
        return std::unexpected(u0.error());
    }
    const auto value = pde_burgers_1d(*u0, nu, dx, dt, steps);
    if (value.u.empty() || value.t.empty()) {
        return std::unexpected(DomainError{
            "pde_burgers_1d", "invalid input or grid too small"});
    }
    return vector_to_column(value.u.back());
}

Result<Matrix<double>> eval_graph_floyd_warshall(const Matrix<double>& adj_m) {
    auto G = graph_from_adjacency(adj_m, "graph_floyd_warshall");
    if (!G) {
        return std::unexpected(G.error());
    }
    return nested_to_matrix(graph::floyd_warshall(*G));
}

Result<Matrix<double>> eval_poly_integ(const Matrix<double>& coeffs_m, double c) {
    auto coeffs = matrix_to_coeff_vector(coeffs_m, "poly_integ");
    if (!coeffs) {
        return std::unexpected(coeffs.error());
    }
    if (coeffs->empty()) {
        return std::unexpected(
            DomainError{"poly_integ", "expected non-empty coefficient vector"});
    }
    return vector_to_column(poly::poly_integ(*coeffs, c));
}

Result<double> eval_stats_spearman(const Matrix<double>& x_m, const Matrix<double>& y_m) {
    auto x = matrix_to_coeff_vector(x_m, "stats_spearman");
    if (!x) {
        return std::unexpected(x.error());
    }
    auto y = matrix_to_coeff_vector(y_m, "stats_spearman");
    if (!y) {
        return std::unexpected(y.error());
    }
    if (x->size() != y->size()) {
        return std::unexpected(
            DomainError{"stats_spearman", "vector length mismatch"});
    }
    if (x->empty()) {
        return std::unexpected(
            DomainError{"stats_spearman", "expected non-empty vectors"});
    }
    return spearman(*x, *y);
}

Result<double> eval_stats_median(const Matrix<double>& x_m) {
    auto x = matrix_to_coeff_vector(x_m, "stats_median");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (x->empty()) {
        return std::unexpected(
            DomainError{"stats_median", "expected non-empty vector"});
    }
    return median(*x);
}

Result<double> eval_graph_is_connected(const Matrix<double>& adj_m) {
    auto G = graph_from_adjacency_undirected(adj_m, "graph_is_connected");
    if (!G) {
        return std::unexpected(G.error());
    }
    return graph::is_connected(*G) ? 1.0 : 0.0;
}

Result<Matrix<double>> eval_fft_dct2(const Matrix<double>& x_m) {
    auto x = matrix_to_coeff_vector(x_m, "fft_dct2");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (x->empty()) {
        return std::unexpected(DomainError{"fft_dct2", "expected non-empty vector"});
    }
    auto coeffs = dct2(*x);
    if (!coeffs) {
        return std::unexpected(coeffs.error());
    }
    return vector_to_column(*coeffs);
}

Result<Matrix<double>> eval_poly_add(const Matrix<double>& a_m, const Matrix<double>& b_m) {
    auto a = matrix_to_coeff_vector(a_m, "poly_add");
    if (!a) {
        return std::unexpected(a.error());
    }
    auto b = matrix_to_coeff_vector(b_m, "poly_add");
    if (!b) {
        return std::unexpected(b.error());
    }
    if (a->empty() || b->empty()) {
        return std::unexpected(
            DomainError{"poly_add", "expected non-empty coefficient vectors"});
    }
    return vector_to_column(poly::poly_add(*a, *b));
}

Result<Matrix<double>> eval_quantum_tensor_product(const Matrix<double>& A_m,
                                                   const Matrix<double>& B_m) {
    auto A = matrix_to_density_matrix(A_m, "quantum_tensor_product");
    if (!A) {
        return std::unexpected(A.error());
    }
    auto B = matrix_to_density_matrix(B_m, "quantum_tensor_product");
    if (!B) {
        return std::unexpected(B.error());
    }
    return density_matrix_to_matrix(quantum::tensor_product(*A, *B));
}

Result<double> eval_stats_kendall(const Matrix<double>& x_m, const Matrix<double>& y_m) {
    auto x = matrix_to_coeff_vector(x_m, "stats_kendall");
    if (!x) {
        return std::unexpected(x.error());
    }
    auto y = matrix_to_coeff_vector(y_m, "stats_kendall");
    if (!y) {
        return std::unexpected(y.error());
    }
    if (x->size() != y->size()) {
        return std::unexpected(
            DomainError{"stats_kendall", "vector length mismatch"});
    }
    if (x->empty()) {
        return std::unexpected(
            DomainError{"stats_kendall", "expected non-empty vectors"});
    }
    return kendall(*x, *y);
}

Result<Matrix<double>> eval_graph_mst_kruskal(const Matrix<double>& adj_m) {
    auto G = graph_from_adjacency_undirected(adj_m, "graph_mst_kruskal");
    if (!G) {
        return std::unexpected(G.error());
    }
    const auto edges = graph::mst_kruskal(*G);
    Matrix<double> out(edges.size(), 3);
    for (size_t i = 0; i < edges.size(); ++i) {
        out(i, 0) = static_cast<double>(edges[i].from);
        out(i, 1) = static_cast<double>(edges[i].to);
        out(i, 2) = edges[i].weight;
    }
    return out;
}

Result<Matrix<double>> eval_signal_correlate(const Matrix<double>& a_m,
                                             const Matrix<double>& b_m) {
    auto a = matrix_to_coeff_vector(a_m, "signal_correlate");
    if (!a) {
        return std::unexpected(a.error());
    }
    auto b = matrix_to_coeff_vector(b_m, "signal_correlate");
    if (!b) {
        return std::unexpected(b.error());
    }
    if (a->empty() || b->empty()) {
        return std::unexpected(
            DomainError{"signal_correlate", "expected non-empty vectors"});
    }
    return vector_to_column(correlate(*a, *b));
}

Result<double> eval_stats_stddev(const Matrix<double>& x_m) {
    auto x = matrix_to_coeff_vector(x_m, "stats_stddev");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (x->empty()) {
        return std::unexpected(
            DomainError{"stats_stddev", "expected non-empty vector"});
    }
    return stddev(*x);
}

Result<Matrix<double>> eval_fft_idct2(const Matrix<double>& x_m) {
    auto x = matrix_to_coeff_vector(x_m, "fft_idct2");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (x->empty()) {
        return std::unexpected(DomainError{"fft_idct2", "expected non-empty vector"});
    }
    auto signal = idct2(*x);
    if (!signal) {
        return std::unexpected(signal.error());
    }
    return vector_to_column(*signal);
}

Result<Matrix<double>> eval_poly_mul(const Matrix<double>& a_m, const Matrix<double>& b_m) {
    auto a = matrix_to_coeff_vector(a_m, "poly_mul");
    if (!a) {
        return std::unexpected(a.error());
    }
    auto b = matrix_to_coeff_vector(b_m, "poly_mul");
    if (!b) {
        return std::unexpected(b.error());
    }
    if (a->empty() || b->empty()) {
        return std::unexpected(
            DomainError{"poly_mul", "expected non-empty coefficient vectors"});
    }
    return vector_to_column(poly::poly_mul(*a, *b));
}

Result<Matrix<double>> eval_graph_mst_prim(const Matrix<double>& adj_m) {
    auto G = graph_from_adjacency_undirected(adj_m, "graph_mst_prim");
    if (!G) {
        return std::unexpected(G.error());
    }
    const auto edges = graph::mst_prim(*G, 0);
    Matrix<double> out(edges.size(), 3);
    for (size_t i = 0; i < edges.size(); ++i) {
        out(i, 0) = static_cast<double>(edges[i].from);
        out(i, 1) = static_cast<double>(edges[i].to);
        out(i, 2) = edges[i].weight;
    }
    return out;
}

Result<double> eval_stats_skewness(const Matrix<double>& x_m) {
    auto x = matrix_to_coeff_vector(x_m, "stats_skewness");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (x->empty()) {
        return std::unexpected(
            DomainError{"stats_skewness", "expected non-empty vector"});
    }
    return skewness(*x);
}

Result<Matrix<double>> eval_poly_sub(const Matrix<double>& a_m, const Matrix<double>& b_m) {
    auto a = matrix_to_coeff_vector(a_m, "poly_sub");
    if (!a) {
        return std::unexpected(a.error());
    }
    auto b = matrix_to_coeff_vector(b_m, "poly_sub");
    if (!b) {
        return std::unexpected(b.error());
    }
    if (a->empty() || b->empty()) {
        return std::unexpected(
            DomainError{"poly_sub", "expected non-empty coefficient vectors"});
    }
    return vector_to_column(poly::poly_sub(*a, *b));
}

Result<double> eval_stats_kurtosis(const Matrix<double>& x_m) {
    auto x = matrix_to_coeff_vector(x_m, "stats_kurtosis");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (x->empty()) {
        return std::unexpected(
            DomainError{"stats_kurtosis", "expected non-empty vector"});
    }
    return kurtosis(*x);
}

Result<double> eval_stats_var(const Matrix<double>& x_m) {
    auto x = matrix_to_coeff_vector(x_m, "stats_var");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (x->empty()) {
        return std::unexpected(
            DomainError{"stats_var", "expected non-empty vector"});
    }
    return var(*x);
}

Result<Matrix<double>> eval_poly_compose(const Matrix<double>& p_m, const Matrix<double>& q_m) {
    auto p = matrix_to_coeff_vector(p_m, "poly_compose");
    if (!p) {
        return std::unexpected(p.error());
    }
    auto q = matrix_to_coeff_vector(q_m, "poly_compose");
    if (!q) {
        return std::unexpected(q.error());
    }
    if (p->empty() || q->empty()) {
        return std::unexpected(
            DomainError{"poly_compose", "expected non-empty coefficient vectors"});
    }
    return vector_to_column(poly::poly_compose(*p, *q));
}

Result<Matrix<double>> eval_graph_bfs(const Matrix<double>& adj_m, int source) {
    auto G = graph_from_adjacency(adj_m, "graph_bfs");
    if (!G) {
        return std::unexpected(G.error());
    }
    if (source < 0 || source >= G->n_vertices()) {
        return std::unexpected(
            DomainError{"graph_bfs", "source out of range"});
    }
    return int_vector_to_column(graph::bfs(*G, source));
}

Result<double> eval_graph_is_tree(const Matrix<double>& adj_m) {
    auto G = graph_from_adjacency_undirected(adj_m, "graph_is_tree");
    if (!G) {
        return std::unexpected(G.error());
    }
    return graph::is_tree(*G) ? 1.0 : 0.0;
}

Result<Matrix<double>> eval_graph_dfs(const Matrix<double>& adj_m, int source) {
    auto G = graph_from_adjacency(adj_m, "graph_dfs");
    if (!G) {
        return std::unexpected(G.error());
    }
    if (source < 0 || source >= G->n_vertices()) {
        return std::unexpected(
            DomainError{"graph_dfs", "source out of range"});
    }
    return int_vector_to_column(graph::dfs(*G, source));
}

Result<Matrix<double>> eval_graph_astar(const Matrix<double>& adj_m, int source, int target,
                                        const Matrix<double>& h_m) {
    auto G = graph_from_adjacency(adj_m, "graph_astar");
    if (!G) {
        return std::unexpected(G.error());
    }
    auto h = matrix_to_coeff_vector(h_m, "graph_astar");
    if (!h) {
        return std::unexpected(h.error());
    }
    if (h->size() != static_cast<size_t>(G->n_vertices())) {
        return std::unexpected(
            DomainError{"graph_astar", "heuristic vector length must match vertex count"});
    }
    if (source < 0 || target < 0 || source >= G->n_vertices() || target >= G->n_vertices()) {
        return std::unexpected(
            DomainError{"graph_astar", "source/target out of range"});
    }
    auto path = graph::astar(*G, source, target, *h);
    if (!path) {
        return std::unexpected(path.error());
    }
    return int_vector_to_column(*path);
}

Result<double> eval_stats_percentile(const Matrix<double>& x_m, double p) {
    auto x = matrix_to_coeff_vector(x_m, "stats_percentile");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (x->empty()) {
        return std::unexpected(
            DomainError{"stats_percentile", "expected non-empty vector"});
    }
    return percentile(*x, p);
}

Result<Matrix<double>> eval_signal_lowpass(const Matrix<double>& x_m, double cutoff, double fs) {
    auto x = matrix_to_coeff_vector(x_m, "signal_lowpass");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (x->empty()) {
        return std::unexpected(
            DomainError{"signal_lowpass", "expected non-empty signal vector"});
    }
    return vector_to_column(lowpass(*x, cutoff, fs));
}

Result<Matrix<double>> eval_signal_butterworth(const Matrix<double>& x_m, double cutoff, double fs) {
    auto x = matrix_to_coeff_vector(x_m, "signal_butterworth");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (x->empty()) {
        return std::unexpected(
            DomainError{"signal_butterworth", "expected non-empty signal vector"});
    }
    return vector_to_column(butterworth(*x, cutoff, fs));
}

Result<Matrix<double>> eval_signal_highpass(const Matrix<double>& x_m, double cutoff, double fs) {
    auto x = matrix_to_coeff_vector(x_m, "signal_highpass");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (x->empty()) {
        return std::unexpected(
            DomainError{"signal_highpass", "expected non-empty signal vector"});
    }
    return vector_to_column(highpass(*x, cutoff, fs));
}

Result<Matrix<double>> eval_signal_bandpass(const Matrix<double>& x_m, double low, double high,
                                            double fs) {
    auto x = matrix_to_coeff_vector(x_m, "signal_bandpass");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (x->empty()) {
        return std::unexpected(
            DomainError{"signal_bandpass", "expected non-empty signal vector"});
    }
    return vector_to_column(bandpass(*x, low, high, fs));
}

Result<Matrix<double>> eval_graph_topological_sort(const Matrix<double>& adj_m) {
    auto G = graph_from_adjacency(adj_m, "graph_topological_sort");
    if (!G) {
        return std::unexpected(G.error());
    }
    auto order = graph::topological_sort(*G);
    if (!order) {
        return std::unexpected(order.error());
    }
    return int_vector_to_column(*order);
}

Result<double> eval_stats_mode(const Matrix<double>& x_m) {
    auto x = matrix_to_coeff_vector(x_m, "stats_mode");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (x->empty()) {
        return std::unexpected(
            DomainError{"stats_mode", "expected non-empty vector"});
    }
    return mode(*x);
}

Result<double> eval_stats_geometric_mean(const Matrix<double>& x_m) {
    auto x = matrix_to_coeff_vector(x_m, "stats_geometric_mean");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (x->empty()) {
        return std::unexpected(
            DomainError{"stats_geometric_mean", "expected non-empty vector"});
    }
    return geometric_mean(*x);
}

Result<double> eval_stats_ttest(const Matrix<double>& x_m, double mu) {
    auto x = matrix_to_coeff_vector(x_m, "stats_ttest");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (x->empty()) {
        return std::unexpected(
            DomainError{"stats_ttest", "expected non-empty vector"});
    }
    return ttest(*x, mu);
}

Result<double> eval_stats_harmonic_mean(const Matrix<double>& x_m) {
    auto x = matrix_to_coeff_vector(x_m, "stats_harmonic_mean");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (x->empty()) {
        return std::unexpected(
            DomainError{"stats_harmonic_mean", "expected non-empty vector"});
    }
    return harmonic_mean(*x);
}

Result<double> eval_stats_rms(const Matrix<double>& x_m) {
    auto x = matrix_to_coeff_vector(x_m, "stats_rms");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (x->empty()) {
        return std::unexpected(DomainError{"stats_rms", "expected non-empty vector"});
    }
    return rms(*x);
}

Result<double> eval_stats_mad(const Matrix<double>& x_m) {
    auto x = matrix_to_coeff_vector(x_m, "stats_mad");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (x->empty()) {
        return std::unexpected(DomainError{"stats_mad", "expected non-empty vector"});
    }
    return mad(*x);
}

Result<double> eval_stats_iqr(const Matrix<double>& x_m) {
    auto x = matrix_to_coeff_vector(x_m, "stats_iqr");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (x->empty()) {
        return std::unexpected(DomainError{"stats_iqr", "expected non-empty vector"});
    }
    return iqr(*x);
}

Result<double> eval_stats_ztest(const Matrix<double>& x_m, double mu, double sigma) {
    auto x = matrix_to_coeff_vector(x_m, "stats_ztest");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (x->empty()) {
        return std::unexpected(DomainError{"stats_ztest", "expected non-empty vector"});
    }
    return ztest(*x, mu, sigma);
}

Result<Matrix<double>> eval_stats_acf(const Matrix<double>& x_m, int max_lag) {
    auto x = matrix_to_coeff_vector(x_m, "stats_acf");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (x->empty()) {
        return std::unexpected(DomainError{"stats_acf", "expected non-empty vector"});
    }
    if (max_lag < 0) {
        return std::unexpected(DomainError{"stats_acf", "expected non-negative integer max_lag"});
    }
    return vector_to_column(acf(*x, max_lag));
}

Result<double> eval_stats_two_sample_ttest(const Matrix<double>& a_m, const Matrix<double>& b_m) {
    auto a = matrix_to_coeff_vector(a_m, "stats_two_sample_ttest");
    if (!a) {
        return std::unexpected(a.error());
    }
    auto b = matrix_to_coeff_vector(b_m, "stats_two_sample_ttest");
    if (!b) {
        return std::unexpected(b.error());
    }
    if (a->empty() || b->empty()) {
        return std::unexpected(
            DomainError{"stats_two_sample_ttest", "expected non-empty vectors"});
    }
    return two_sample_ttest(*a, *b);
}

Result<double> eval_stats_chi2_gof(const Matrix<double>& obs_m, const Matrix<double>& exp_m) {
    auto obs = matrix_to_coeff_vector(obs_m, "stats_chi2_gof");
    if (!obs) {
        return std::unexpected(obs.error());
    }
    auto exp = matrix_to_coeff_vector(exp_m, "stats_chi2_gof");
    if (!exp) {
        return std::unexpected(exp.error());
    }
    if (obs->empty() || exp->empty()) {
        return std::unexpected(
            DomainError{"stats_chi2_gof", "expected non-empty vectors"});
    }
    if (obs->size() != exp->size()) {
        return std::unexpected(
            DomainError{"stats_chi2_gof", "vector length mismatch"});
    }
    return chi2_gof(*obs, *exp);
}

Result<Matrix<double>> eval_fft_fft2(const Matrix<double>& spectrum_m) {
    auto spec = matrix_to_complex_spectrum(spectrum_m, "fft_fft2");
    if (!spec) {
        return std::unexpected(spec.error());
    }
    auto out = fft2(*spec);
    if (!out) {
        return std::unexpected(out.error());
    }
    Matrix<double> result(out->size(), 2);
    for (size_t i = 0; i < out->size(); ++i) {
        result(i, 0) = (*out)[i].real();
        result(i, 1) = (*out)[i].imag();
    }
    return result;
}

Result<Matrix<double>> eval_fft_dft(const Matrix<double>& x_m) {
    auto x = matrix_to_coeff_vector(x_m, "fft_dft");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (x->empty()) {
        return std::unexpected(DomainError{"fft_dft", "expected non-empty signal vector"});
    }
    auto spectrum = dft(*x);
    if (!spectrum) {
        return std::unexpected(spectrum.error());
    }
    Matrix<double> out(spectrum->size(), 2);
    for (size_t i = 0; i < spectrum->size(); ++i) {
        out(i, 0) = (*spectrum)[i].real();
        out(i, 1) = (*spectrum)[i].imag();
    }
    return out;
}

Result<Matrix<double>> eval_graph_greedy_colour(const Matrix<double>& adj_m) {
    auto G = graph_from_adjacency_undirected(adj_m, "graph_greedy_colour");
    if (!G) {
        return std::unexpected(G.error());
    }
    return int_vector_to_column(graph::greedy_colour(*G));
}

Result<Matrix<double>> eval_graph_euler_circuit(const Matrix<double>& adj_m) {
    auto G = graph_from_adjacency_undirected(adj_m, "graph_euler_circuit");
    if (!G) {
        return std::unexpected(G.error());
    }
    return int_vector_to_column(graph::euler_circuit(*G));
}

Result<Matrix<double>> eval_fft_dst2(const Matrix<double>& x_m) {
    auto x = matrix_to_coeff_vector(x_m, "fft_dst2");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (x->empty()) {
        return std::unexpected(DomainError{"fft_dst2", "expected non-empty vector"});
    }
    auto coeffs = dst2(*x);
    if (!coeffs) {
        return std::unexpected(coeffs.error());
    }
    return vector_to_column(*coeffs);
}

Result<Matrix<double>> eval_combo_derangements(int n) {
    if (n < 0) {
        return std::unexpected(
            DomainError{"combo_derangements", "expected non-negative integer n"});
    }
    const auto rows = combo::derangements(n);
    Matrix<double> out(rows.size(), static_cast<size_t>(n));
    for (size_t r = 0; r < rows.size(); ++r) {
        for (size_t c = 0; c < rows[r].size(); ++c) {
            out(r, c) = static_cast<double>(rows[r][c]);
        }
    }
    return out;
}

Result<double> eval_cplx_line_integral_one() {
    const auto f = [](const cplx::C&) -> cplx::C { return cplx::C(1.0); };
    const cplx::C val = cplx::line_integral(f, cplx::C(0.0), cplx::C(1.0), 50);
    return val.real();
}

Result<Matrix<double>> eval_quantum_density_matrix(const Matrix<double>& psi_m) {
    auto psi = matrix_to_ket(psi_m, "quantum_density_matrix");
    if (!psi) {
        return std::unexpected(psi.error());
    }
    return density_matrix_to_matrix(quantum::density_matrix(*psi));
}

Result<std::vector<topo::PersistencePair>> matrix_to_persistence_diagram(const Matrix<double>& m,
                                                                         const char* fn) {
    if (m.cols() != 3) {
        return std::unexpected(
            DomainError{fn, "expected Nx3 persistence diagram [dim,birth,death]"});
    }
    if (m.rows() == 0) {
        return std::unexpected(DomainError{fn, "expected non-empty persistence diagram"});
    }
    std::vector<topo::PersistencePair> pairs;
    pairs.reserve(m.rows());
    for (size_t r = 0; r < m.rows(); ++r) {
        const int dim = static_cast<int>(m(r, 0));
        if (dim < 0 || m(r, 0) != dim) {
            return std::unexpected(DomainError{fn, "expected integer dim column"});
        }
        pairs.push_back({dim, m(r, 1), m(r, 2)});
    }
    return pairs;
}

Result<double> eval_topo_bottleneck_distance(const Matrix<double>& dgm1_m,
                                             const Matrix<double>& dgm2_m, int dim) {
    if (dim < 0) {
        return std::unexpected(
            DomainError{"topo_bottleneck_distance", "expected non-negative integer dim"});
    }
    auto dgm1 = matrix_to_persistence_diagram(dgm1_m, "topo_bottleneck_distance");
    if (!dgm1) {
        return std::unexpected(dgm1.error());
    }
    auto dgm2 = matrix_to_persistence_diagram(dgm2_m, "topo_bottleneck_distance");
    if (!dgm2) {
        return std::unexpected(dgm2.error());
    }
    return topo::bottleneck_distance(*dgm1, *dgm2, dim);
}

Result<double> eval_topo_wasserstein_distance(const Matrix<double>& dgm1_m,
                                              const Matrix<double>& dgm2_m, int dim) {
    if (dim < 0) {
        return std::unexpected(
            DomainError{"topo_wasserstein_distance", "expected non-negative integer dim"});
    }
    auto dgm1 = matrix_to_persistence_diagram(dgm1_m, "topo_wasserstein_distance");
    if (!dgm1) {
        return std::unexpected(dgm1.error());
    }
    auto dgm2 = matrix_to_persistence_diagram(dgm2_m, "topo_wasserstein_distance");
    if (!dgm2) {
        return std::unexpected(dgm2.error());
    }
    return topo::wasserstein_distance(*dgm1, *dgm2, dim);
}

diffgeo::MetricFn unit_sphere_metric_fn() {
    return [](const diffgeo::Coords& x) -> std::vector<std::vector<double>> {
        const double theta = x[0];
        const double s2 = std::sin(theta) * std::sin(theta);
        return {{1.0, 0.0}, {0.0, s2}};
    };
}

Result<double> eval_diffgeo_christoffel_sphere(double k_d, double i_d, double j_d, double u,
                                               double v) {
    const int k = static_cast<int>(k_d);
    const int i = static_cast<int>(i_d);
    const int j = static_cast<int>(j_d);
    if (k < 0 || i < 0 || j < 0 || k > 1 || i > 1 || j > 1 || k_d != k || i_d != i || j_d != j) {
        return std::unexpected(
            DomainError{"diffgeo_christoffel_sphere", "expected k,i,j in {0,1}"});
    }
    const auto Chr = diffgeo::christoffel(unit_sphere_metric_fn(), {u, v});
    return Chr[static_cast<size_t>(k)][static_cast<size_t>(i)][static_cast<size_t>(j)];
}

Result<double> eval_diffgeo_ricci_scalar_sphere(double u, double v) {
    return diffgeo::ricci_scalar(unit_sphere_metric_fn(), {u, v});
}

Result<double> eval_diffgeo_einstein_scalar_sphere(double u, double v) {
    const auto G = diffgeo::einstein_tensor(unit_sphere_metric_fn(), {u, v});
    const auto gv = unit_sphere_metric_fn()({u, v});
    const auto ginv = diffgeo::metric_inv(gv);
    double contracted = 0.0;
    for (size_t i = 0; i < G.size(); ++i) {
        for (size_t j = 0; j < G[i].size(); ++j) {
            contracted += ginv[i][j] * G[i][j];
        }
    }
    return contracted;
}

Result<double> eval_cplx_joukowski_inv(double re, double im) {
    const cplx::C z0{re, im};
    const cplx::C w = cplx::joukowski(z0, 1.0);
    const auto roots = cplx::joukowski_inv(w, 1.0);
    for (const auto& root : roots) {
        if (std::abs(cplx::joukowski(root, 1.0) - w) < 1e-8 && std::abs(root - z0) < 1e-8) {
            return std::abs(root);
        }
    }
    return std::unexpected(
        DomainError{"cplx_joukowski_inv", "no Joukowski inverse root matches forward point"});
}

Result<double> eval_cplx_residue_inv(double pole_re, double pole_im) {
    const cplx::C pole(pole_re, pole_im);
    const auto f = [&](const cplx::C& z) -> cplx::C { return cplx::C(1.0) / (z - pole); };
    const cplx::C res = cplx::residue(f, pole, 1e-5);
    return res.real();
}

Result<double> eval_cplx_contour_integral_oneoverz_im() {
    constexpr int N = 64;
    std::vector<cplx::C> path;
    path.reserve(static_cast<size_t>(N) + 1);
    for (int k = 0; k <= N; ++k) {
        const double theta = 2.0 * M_PI * static_cast<double>(k) / static_cast<double>(N);
        path.push_back(cplx::C(std::cos(theta), std::sin(theta)));
    }
    const auto f = [](const cplx::C& z) -> cplx::C { return cplx::C(1.0) / z; };
    const cplx::C val = cplx::contour_integral(f, path, 40);
    return val.imag();
}

Result<Matrix<double>> eval_quantum_time_evolution_matrix(const Matrix<double>& H_m, double t) {
    auto H = matrix_to_density_matrix(H_m, "quantum_time_evolution");
    if (!H) {
        return std::unexpected(H.error());
    }
    const auto U = quantum::time_evolution_operator(*H, t);
    return density_matrix_to_matrix(U);
}

Result<Matrix<double>> eval_topo_betti_curve(const Matrix<double>& dist_m,
                                             const Matrix<double>& thresholds_m, int max_dim) {
    auto nested = matrix_to_square_nested(dist_m, "topo_betti_curve");
    if (!nested) {
        return std::unexpected(nested.error());
    }
    auto thresholds = matrix_to_coeff_vector(thresholds_m, "topo_betti_curve");
    if (!thresholds) {
        return std::unexpected(thresholds.error());
    }
    if (max_dim < 0) {
        return std::unexpected(
            DomainError{"topo_betti_curve", "expected non-negative integer max_dim"});
    }
    const auto curve = topo::betti_curve(*nested, *thresholds, max_dim);
    const size_t cols = static_cast<size_t>(max_dim + 1);
    Matrix<double> out(curve.size(), cols);
    for (size_t i = 0; i < curve.size(); ++i) {
        for (size_t j = 0; j < cols; ++j) {
            out(i, j) = static_cast<double>(curve[i][j]);
        }
    }
    return out;
}

Result<Matrix<double>> eval_quantum_bell_state(int index) {
    if (index < 0 || index > 3) {
        return std::unexpected(
            DomainError{"quantum_bell_state", "expected bell state index in [0,3]"});
    }
    const auto states = quantum::bell_states();
    return ket_to_column_matrix(states[static_cast<size_t>(index)]);
}

Result<Matrix<double>> eval_bzip2_compress_vec(const Matrix<double>& m) {
    return bytes_to_matrix_col(compress::bzip2_like_compress(matrix_to_bytes(m)));
}

Result<Matrix<double>> eval_bzip2_decompress_vec(const Matrix<double>& c_m) {
    const compress::Bytes bytes = matrix_to_bytes(c_m);
    if (bytes.size() < 4) {
        return std::unexpected(
            DomainError{"bzip2_decompress_vec", "expected at least 4-byte compressed vector"});
    }
    const int pi = (static_cast<int>(bytes[0]) << 24) | (static_cast<int>(bytes[1]) << 16) |
                   (static_cast<int>(bytes[2]) << 8) | static_cast<int>(bytes[3]);
    return bytes_to_matrix_col(compress::bzip2_like_decompress(bytes, pi));
}

Result<Matrix<double>> eval_control_place(const Matrix<double>& A_m,
                                          const Matrix<double>& B_m,
                                          const Matrix<double>& poles_m) {
    auto A = matrix_to_square_nested(A_m, "control_place");
    if (!A) {
        return std::unexpected(A.error());
    }
    auto B = matrix_to_nested(B_m, "control_place");
    if (!B) {
        return std::unexpected(B.error());
    }
    auto poles = matrix_to_coeff_vector(poles_m, "control_place");
    if (!poles) {
        return std::unexpected(poles.error());
    }
    if (B->size() != A->size()) {
        return std::unexpected(
            DomainError{"control_place", "expected B with same row count as A"});
    }
    if (poles->size() != A->size()) {
        return std::unexpected(
            DomainError{"control_place", "expected poles vector length equal to A size"});
    }
    auto K = control::place(*A, *B, *poles);
    if (!K) {
        return std::unexpected(K.error());
    }
    Matrix<double> out(K->size(), 1);
    for (size_t i = 0; i < K->size(); ++i) {
        out(i, 0) = (*K)[i];
    }
    return out;
}

diffgeo::SurfaceFn unit_sphere_surface() {
    return [](double u, double v) -> std::array<double, 3> {
        return {std::cos(u) * std::cos(v), std::cos(u) * std::sin(v), std::sin(u)};
    };
}

Result<Matrix<double>> eval_diffgeo_surface_normal_sphere(double u, double v) {
    const auto N = diffgeo::surface_normal(unit_sphere_surface(), u, v);
    Matrix<double> out(3, 1);
    out(0, 0) = N[0];
    out(1, 0) = N[1];
    out(2, 0) = N[2];
    return out;
}

Result<double> eval_diffgeo_gaussian_sphere() {
    const auto sphere = unit_sphere_surface();
    constexpr double u = 0.5;
    constexpr double v = 0.5;
    return diffgeo::gaussian_curvature(sphere, u, v);
}

Result<double> eval_diffgeo_gaussian_curvature_sphere(double u, double v) {
    return diffgeo::gaussian_curvature(unit_sphere_surface(), u, v);
}

Result<double> eval_diffgeo_mean_curvature_sphere(double u, double v) {
    return diffgeo::mean_curvature(unit_sphere_surface(), u, v);
}

Result<double> eval_diffgeo_mean_sphere() {
    const auto sphere = unit_sphere_surface();
    constexpr double u = 0.5;
    constexpr double v = 0.5;
    return diffgeo::mean_curvature(sphere, u, v);
}

Result<double> eval_topo_euler_tetrahedron() {
    topo::SimplicialComplex complex;
    complex.add_simplex({0, 1, 2, 3});
    return static_cast<double>(complex.euler_characteristic());
}

Result<double> eval_diffgeo_principal_curvature_sphere() {
    const auto sphere = unit_sphere_surface();
    const auto [k1, k2] = diffgeo::principal_curvatures(sphere, 0.3, 0.7);
    static_cast<void>(k2);
    return k1;
}

Result<double> eval_topo_euler_sphere_surface() {
    topo::SimplicialComplex sc;
    sc.add_simplex({0, 1, 2});
    sc.add_simplex({0, 1, 3});
    sc.add_simplex({0, 2, 3});
    sc.add_simplex({1, 2, 3});
    return static_cast<double>(sc.euler_characteristic());
}

bool is_nullary_scalar_callee(const std::string& callee) {
    return callee == "diffgeo_gaussian_sphere" || callee == "diffgeo_mean_sphere" ||
           callee == "diffgeo_principal_curvature_sphere" ||
           callee == "topo_euler_tetrahedron" || callee == "topo_euler_sphere_surface" ||
           callee == "cplx_contour_integral_oneoverz_im" || callee == "cplx_line_integral_one";
}

Result<double> eval_nullary_scalar_call(const std::string& fn) {
    if (fn == "diffgeo_gaussian_sphere") {
        return eval_diffgeo_gaussian_sphere();
    }
    if (fn == "diffgeo_mean_sphere") {
        return eval_diffgeo_mean_sphere();
    }
    if (fn == "topo_euler_tetrahedron") {
        return eval_topo_euler_tetrahedron();
    }
    if (fn == "diffgeo_principal_curvature_sphere") {
        return eval_diffgeo_principal_curvature_sphere();
    }
    if (fn == "topo_euler_sphere_surface") {
        return eval_topo_euler_sphere_surface();
    }
    if (fn == "cplx_contour_integral_oneoverz_im") {
        return eval_cplx_contour_integral_oneoverz_im();
    }
    if (fn == "cplx_line_integral_one") {
        return eval_cplx_line_integral_one();
    }
    return std::unexpected(DomainError{"eval", "unknown nullary scalar function: " + fn});
}

bool is_nullary_matrix_callee(const std::string& callee) {
    return callee == "quantum_pauli_x" || callee == "quantum_pauli_y" ||
           callee == "quantum_pauli_z" || callee == "quantum_pauli_plus" ||
           callee == "quantum_pauli_minus" || callee == "quantum_cnot_gate" ||
           callee == "quantum_swap_gate" || callee == "quantum_toffoli_gate" ||
           callee == "quantum_identity" || callee == "quantum_hadamard_gate";
}

Result<Matrix<double>> eval_nullary_matrix_call(const std::string& fn) {
    if (fn == "quantum_pauli_x") {
        return density_matrix_to_matrix(quantum::pauli_x());
    }
    if (fn == "quantum_pauli_y") {
        return density_matrix_to_matrix(quantum::pauli_y());
    }
    if (fn == "quantum_pauli_z") {
        return density_matrix_to_matrix(quantum::pauli_z());
    }
    if (fn == "quantum_pauli_plus") {
        return density_matrix_to_matrix(quantum::pauli_plus());
    }
    if (fn == "quantum_pauli_minus") {
        return density_matrix_to_matrix(quantum::pauli_minus());
    }
    if (fn == "quantum_cnot_gate") {
        return density_matrix_to_matrix(quantum::cnot_gate());
    }
    if (fn == "quantum_swap_gate") {
        return density_matrix_to_matrix(quantum::swap_gate());
    }
    if (fn == "quantum_toffoli_gate") {
        return density_matrix_to_matrix(quantum::toffoli_gate());
    }
    if (fn == "quantum_identity") {
        return density_matrix_to_matrix(quantum::identity(2));
    }
    if (fn == "quantum_hadamard_gate") {
        return density_matrix_to_matrix(quantum::hadamard());
    }
    return std::unexpected(DomainError{"eval", "unknown nullary matrix function: " + fn});
}

bool is_unary_scalar_matrix_callee(const std::string& callee) {
    return callee == "quantum_rotation_z" || callee == "quantum_rotation_x" ||
           callee == "quantum_rotation_y" || callee == "quantum_phase_gate" ||
           callee == "quantum_qft_gate" || callee == "quantum_identity_n" ||
           callee == "quantum_ghz_state" || callee == "quantum_w_state" ||
           callee == "quantum_bell_state" ||
           callee == "numthy_divisors_vec" || callee == "numthy_factor_vec" ||
           callee == "combo_derangements" || callee == "combo_all_permutations" ||
           callee == "combo_all_subsets" || callee == "combo_all_compositions" ||
           callee == "combo_all_partitions" || callee == "signal_hamming" ||
           callee == "signal_hanning" || callee == "signal_blackman" ||
           callee == "signal_parzen" || callee == "signal_triangular";
}

Result<Matrix<double>> eval_unary_scalar_matrix_call(const std::string& fn, double arg) {
    if (fn == "quantum_rotation_z") {
        return density_matrix_to_matrix(quantum::rotation_z(arg));
    }
    if (fn == "quantum_rotation_x") {
        return density_matrix_to_matrix(quantum::rotation_x(arg));
    }
    if (fn == "quantum_rotation_y") {
        return density_matrix_to_matrix(quantum::rotation_y(arg));
    }
    if (fn == "quantum_phase_gate") {
        return density_matrix_to_matrix(quantum::phase_gate(arg));
    }
    if (fn == "quantum_qft_gate") {
        const int n_qubits = static_cast<int>(arg);
        if (n_qubits < 1 || arg != n_qubits) {
            return std::unexpected(
                DomainError{"quantum_qft_gate", "expected integer n_qubits >= 1"});
        }
        return density_matrix_to_matrix(quantum::qft_gate(n_qubits));
    }
    if (fn == "quantum_identity_n") {
        const int dim = static_cast<int>(arg);
        if (dim < 1 || arg != dim) {
            return std::unexpected(
                DomainError{"quantum_identity_n", "expected integer dim >= 1"});
        }
        return density_matrix_to_matrix(quantum::identity(dim));
    }
    if (fn == "quantum_ghz_state") {
        const int n_qubits = static_cast<int>(arg);
        if (n_qubits < 1 || arg != n_qubits) {
            return std::unexpected(
                DomainError{"quantum_ghz_state", "expected integer n_qubits >= 1"});
        }
        return eval_quantum_ghz_state(n_qubits);
    }
    if (fn == "quantum_w_state") {
        const int n_qubits = static_cast<int>(arg);
        if (n_qubits < 1 || arg != n_qubits) {
            return std::unexpected(
                DomainError{"quantum_w_state", "expected integer n_qubits >= 1"});
        }
        return eval_quantum_w_state(n_qubits);
    }
    if (fn == "combo_derangements") {
        const int n = static_cast<int>(arg);
        if (n < 0 || arg != n) {
            return std::unexpected(
                DomainError{"combo_derangements", "expected non-negative integer n"});
        }
        return eval_combo_derangements(n);
    }
    if (fn == "combo_all_permutations") {
        const int n = static_cast<int>(arg);
        if (n < 0 || arg != n) {
            return std::unexpected(
                DomainError{"combo_all_permutations", "expected non-negative integer n"});
        }
        return eval_combo_all_permutations(n);
    }
    if (fn == "combo_all_subsets") {
        const int n = static_cast<int>(arg);
        if (n < 0 || arg != n) {
            return std::unexpected(
                DomainError{"combo_all_subsets", "expected non-negative integer n"});
        }
        return eval_combo_all_subsets(n);
    }
    if (fn == "combo_all_compositions") {
        const int n = static_cast<int>(arg);
        if (n < 0 || arg != n) {
            return std::unexpected(
                DomainError{"combo_all_compositions", "expected non-negative integer n"});
        }
        return eval_combo_all_compositions(n);
    }
    if (fn == "combo_all_partitions") {
        const int n = static_cast<int>(arg);
        if (n < 0 || arg != n) {
            return std::unexpected(
                DomainError{"combo_all_partitions", "expected non-negative integer n"});
        }
        return eval_combo_all_partitions(n);
    }
    if (fn == "numthy_divisors_vec") {
        const int n = static_cast<int>(arg);
        if (n < 1 || arg != n) {
            return std::unexpected(
                DomainError{"numthy_divisors_vec", "expected positive integer n"});
        }
        return eval_numthy_divisors_vec(n);
    }
    if (fn == "numthy_factor_vec") {
        const int n = static_cast<int>(arg);
        if (n < 2 || arg != n) {
            return std::unexpected(
                DomainError{"numthy_factor_vec", "expected integer n >= 2"});
        }
        return eval_numthy_factor_vec(n);
    }
    if (fn == "quantum_bell_state") {
        const int index = static_cast<int>(arg);
        if (index < 0 || index > 3 || arg != index) {
            return std::unexpected(
                DomainError{"quantum_bell_state", "expected integer bell state index in [0,3]"});
        }
        return eval_quantum_bell_state(index);
    }
    if (fn == "signal_hamming") {
        const int n = static_cast<int>(arg);
        if (n < 0 || arg != n) {
            return std::unexpected(
                DomainError{"signal_hamming", "expected non-negative integer n"});
        }
        return vector_to_column(hamming(static_cast<size_t>(n)));
    }
    if (fn == "signal_hanning") {
        const int n = static_cast<int>(arg);
        if (n < 0 || arg != n) {
            return std::unexpected(
                DomainError{"signal_hanning", "expected non-negative integer n"});
        }
        return vector_to_column(hanning(static_cast<size_t>(n)));
    }
    if (fn == "signal_blackman") {
        const int n = static_cast<int>(arg);
        if (n < 0 || arg != n) {
            return std::unexpected(
                DomainError{"signal_blackman", "expected non-negative integer n"});
        }
        return vector_to_column(blackman(static_cast<size_t>(n)));
    }
    if (fn == "signal_parzen") {
        const int n = static_cast<int>(arg);
        if (n < 0 || arg != n) {
            return std::unexpected(
                DomainError{"signal_parzen", "expected non-negative integer n"});
        }
        return vector_to_column(parzen(static_cast<size_t>(n)));
    }
    if (fn == "signal_triangular") {
        const int n = static_cast<int>(arg);
        if (n < 0 || arg != n) {
            return std::unexpected(
                DomainError{"signal_triangular", "expected non-negative integer n"});
        }
        return vector_to_column(triangular(static_cast<size_t>(n)));
    }
    return std::unexpected(DomainError{"eval", "unknown unary scalar matrix function: " + fn});
}

struct ScalarDualMatrixCallAssign {
    std::string target;
    std::string callee;
    std::string arg_a;
    std::string arg_b;
};

struct MatrixDualMatrixCallAssign {
    std::string target;
    std::string callee;
    std::string arg_a;
    std::string arg_b;
};

struct MatrixTripleMatrixCallAssign {
    std::string target;
    std::string callee;
    std::string arg_a;
    std::string arg_b;
    std::string arg_c;
};

struct MatrixQuadMatrixCallAssign {
    std::string target;
    std::string callee;
    std::string arg_a;
    std::string arg_b;
    std::string arg_c;
    std::string arg_d;
};

struct ScalarMatrixMixedCallAssign {
    std::string target;
    std::string callee;
    std::string arg_scalar;
    std::string arg_matrix;
};

struct MatrixScalarMixedCallAssign {
    std::string target;
    std::string callee;
    std::string arg_matrix;
    std::string arg_scalar;
};

struct TwoScalarMatrixMixedCallAssign {
    std::string target;
    std::string callee;
    std::string arg_scalar_a;
    std::string arg_scalar_b;
    std::string arg_matrix;
};

bool is_scalar_matrix_mixed_call_callee(const std::string& callee) {
    return callee == "finance_npv" || callee == "info_renyi_entropy" ||
           callee == "info_tsallis_entropy" || callee == "combo_multinomial";
}

bool is_two_scalar_matrix_mixed_call_callee(const std::string& callee) {
    return callee == "cplx_blaschke_product";
}

bool is_matrix_scalar_mixed_call_callee(const std::string& callee) {
    return callee == "geo_bezier_eval_x" || callee == "geo_bezier_eval_y" ||
           callee == "bwt_decode_vec" || callee == "combo_rank_combination" ||
           callee == "combo_next_comb" ||
           callee == "quantum_time_evolution" || callee == "signal_moving_average" ||
           callee == "graph_bfs" || callee == "graph_dfs" || callee == "stats_percentile" ||
           callee == "stats_ttest" || callee == "stats_acf" ||
           callee == "poly_eval" || callee == "fft_irfft" || callee == "poly_integ";
}

bool is_matrix_dual_matrix_call_callee(const std::string& callee) {
    return callee == "control_lyap" || callee == "control_dlyap" || callee == "huffman_decode_vec" ||
           callee == "quantum_op_apply" || callee == "topo_persistence_diagram" ||
           callee == "control_margins" ||            callee == "quantum_commutator" ||
           callee == "poly_add" || callee == "quantum_tensor_product" ||
           callee == "ml_mat_mul" ||
           callee == "poly_mul" || callee == "poly_sub" || callee == "poly_compose" ||
           callee == "signal_convolve" || callee == "signal_correlate";
}

bool is_matrix_triple_matrix_call_callee(const std::string& callee) {
    return callee == "control_place";
}

bool is_matrix_quad_matrix_call_callee(const std::string& callee) {
    return callee == "control_lqr" || callee == "control_riccati" || callee == "control_dare";
}

bool is_scalar_dual_matrix_call_callee(const std::string& callee) {
    return callee == "ml_accuracy" || callee == "ml_rmse" || callee == "ml_mse" ||
           callee == "ml_r2" || callee == "ml_f1" || callee == "ml_precision" ||
           callee == "ml_recall" || callee == "ml_mae" || callee == "ml_huber" ||
           callee == "ml_hinge" || callee == "ml_binary_crossentropy" ||
           callee == "ml_categorical_crossentropy" || callee == "ml_vec_dot" ||
           callee == "control_step_final" ||
           callee == "control_impulse_final" ||
           callee == "control_dcgain" || callee == "control_is_stable" ||
           callee == "control_pidtune_kp" || callee == "control_pidtune_ki" ||
           callee == "control_pidtune_kd" ||
           callee == "control_is_controllable" || callee == "control_is_observable" ||
           callee == "numthy_crt" ||
           callee == "info_kl_divergence" || callee == "info_cross_entropy" ||
           callee == "info_js_divergence" || callee == "info_tv_distance" ||
           callee == "info_hellinger_dist" || callee == "quantum_fidelity" ||
           callee == "quantum_trace_distance" || callee == "quantum_expectation" ||
           callee == "quantum_expectation_dm" || callee == "quantum_inner" ||
           callee == "finance_portfolio_return" || callee == "finance_portfolio_variance" ||
           callee == "tensorops_inner" ||
           callee == "control_phase_margin" || callee == "control_gain_margin" ||
           callee == "stats_correlation" || callee == "stats_spearman" ||
           callee == "stats_kendall" || callee == "stats_two_sample_ttest" ||
           callee == "stats_chi2_gof";
}

bool is_identifier(const std::string& text);
std::optional<std::vector<std::string>> split_call_args(const std::string& cmd);
std::vector<std::string> split_scalar_call_args(const std::string& args_text);

bool try_parse_scalar_dual_matrix_call_assignment(const std::string& line,
                                                  ScalarDualMatrixCallAssign& assign) {
    const std::string cmd = trim_copy(line);
    const auto eq = cmd.find('=');
    if (eq == std::string::npos) {
        return false;
    }
    assign.target = trim_copy(cmd.substr(0, eq));
    if (assign.target.empty() || !is_identifier(assign.target)) {
        return false;
    }
    const std::string rhs = trim_copy(cmd.substr(eq + 1));
    const auto open = rhs.find('(');
    if (open == std::string::npos) {
        return false;
    }
    assign.callee = lower(trim_copy(rhs.substr(0, open)));
    if (!is_scalar_dual_matrix_call_callee(assign.callee)) {
        return false;
    }
    const auto call_args = split_call_args(rhs);
    if (!call_args || call_args->size() != 2) {
        return false;
    }
    assign.arg_a = trim_copy(call_args->at(0));
    assign.arg_b = trim_copy(call_args->at(1));
    return !assign.arg_a.empty() && !assign.arg_b.empty();
}

bool try_parse_matrix_dual_matrix_call_assignment(const std::string& line,
                                                  MatrixDualMatrixCallAssign& assign) {
    const std::string cmd = trim_copy(line);
    const auto eq = cmd.find('=');
    if (eq == std::string::npos) {
        return false;
    }
    assign.target = trim_copy(cmd.substr(0, eq));
    if (assign.target.empty() || !is_identifier(assign.target)) {
        return false;
    }
    const std::string rhs = trim_copy(cmd.substr(eq + 1));
    const auto open = rhs.find('(');
    if (open == std::string::npos) {
        return false;
    }
    assign.callee = lower(trim_copy(rhs.substr(0, open)));
    if (!is_matrix_dual_matrix_call_callee(assign.callee)) {
        return false;
    }
    const auto call_args = split_call_args(rhs);
    if (!call_args || call_args->size() != 2) {
        return false;
    }
    assign.arg_a = trim_copy(call_args->at(0));
    assign.arg_b = trim_copy(call_args->at(1));
    return !assign.arg_a.empty() && !assign.arg_b.empty();
}

bool try_parse_matrix_triple_matrix_call_assignment(const std::string& line,
                                                    MatrixTripleMatrixCallAssign& assign) {
    const std::string cmd = trim_copy(line);
    const auto eq = cmd.find('=');
    if (eq == std::string::npos) {
        return false;
    }
    assign.target = trim_copy(cmd.substr(0, eq));
    if (assign.target.empty() || !is_identifier(assign.target)) {
        return false;
    }
    const std::string rhs = trim_copy(cmd.substr(eq + 1));
    const auto open = rhs.find('(');
    if (open == std::string::npos) {
        return false;
    }
    assign.callee = lower(trim_copy(rhs.substr(0, open)));
    if (!is_matrix_triple_matrix_call_callee(assign.callee)) {
        return false;
    }
    const auto call_args = split_call_args(rhs);
    if (!call_args || call_args->size() != 3) {
        return false;
    }
    assign.arg_a = trim_copy(call_args->at(0));
    assign.arg_b = trim_copy(call_args->at(1));
    assign.arg_c = trim_copy(call_args->at(2));
    return !assign.arg_a.empty() && !assign.arg_b.empty() && !assign.arg_c.empty();
}

bool try_parse_matrix_quad_matrix_call_assignment(const std::string& line,
                                                  MatrixQuadMatrixCallAssign& assign) {
    const std::string cmd = trim_copy(line);
    const auto eq = cmd.find('=');
    if (eq == std::string::npos) {
        return false;
    }
    assign.target = trim_copy(cmd.substr(0, eq));
    if (assign.target.empty() || !is_identifier(assign.target)) {
        return false;
    }
    const std::string rhs = trim_copy(cmd.substr(eq + 1));
    const auto open = rhs.find('(');
    if (open == std::string::npos) {
        return false;
    }
    assign.callee = lower(trim_copy(rhs.substr(0, open)));
    if (!is_matrix_quad_matrix_call_callee(assign.callee)) {
        return false;
    }
    const auto call_args = split_call_args(rhs);
    if (!call_args || call_args->size() != 4) {
        return false;
    }
    assign.arg_a = trim_copy(call_args->at(0));
    assign.arg_b = trim_copy(call_args->at(1));
    assign.arg_c = trim_copy(call_args->at(2));
    assign.arg_d = trim_copy(call_args->at(3));
    return !assign.arg_a.empty() && !assign.arg_b.empty() && !assign.arg_c.empty() &&
           !assign.arg_d.empty();
}

bool try_parse_scalar_matrix_mixed_call_assignment(const std::string& line,
                                                   ScalarMatrixMixedCallAssign& assign) {
    const std::string cmd = trim_copy(line);
    const auto eq = cmd.find('=');
    if (eq == std::string::npos) {
        return false;
    }
    assign.target = trim_copy(cmd.substr(0, eq));
    if (assign.target.empty() || !is_identifier(assign.target)) {
        return false;
    }
    const std::string rhs = trim_copy(cmd.substr(eq + 1));
    const auto open = rhs.find('(');
    if (open == std::string::npos) {
        return false;
    }
    assign.callee = lower(trim_copy(rhs.substr(0, open)));
    if (!is_scalar_matrix_mixed_call_callee(assign.callee)) {
        return false;
    }
    const auto call_args = split_call_args(rhs);
    if (!call_args || call_args->size() != 2) {
        return false;
    }
    assign.arg_scalar = trim_copy(call_args->at(0));
    assign.arg_matrix = trim_copy(call_args->at(1));
    return !assign.arg_scalar.empty() && !assign.arg_matrix.empty();
}

bool try_parse_matrix_scalar_mixed_call_assignment(const std::string& line,
                                                   MatrixScalarMixedCallAssign& assign) {
    const std::string cmd = trim_copy(line);
    const auto eq = cmd.find('=');
    if (eq == std::string::npos) {
        return false;
    }
    assign.target = trim_copy(cmd.substr(0, eq));
    if (assign.target.empty() || !is_identifier(assign.target)) {
        return false;
    }
    const std::string rhs = trim_copy(cmd.substr(eq + 1));
    const auto open = rhs.find('(');
    if (open == std::string::npos) {
        return false;
    }
    assign.callee = lower(trim_copy(rhs.substr(0, open)));
    if (!is_matrix_scalar_mixed_call_callee(assign.callee)) {
        return false;
    }
    const auto call_args = split_call_args(rhs);
    if (!call_args || call_args->size() != 2) {
        return false;
    }
    assign.arg_matrix = trim_copy(call_args->at(0));
    assign.arg_scalar = trim_copy(call_args->at(1));
    return !assign.arg_matrix.empty() && !assign.arg_scalar.empty();
}

bool try_parse_two_scalar_matrix_mixed_call_assignment(const std::string& line,
                                                       TwoScalarMatrixMixedCallAssign& assign) {
    const std::string cmd = trim_copy(line);
    const auto eq = cmd.find('=');
    if (eq == std::string::npos) {
        return false;
    }
    assign.target = trim_copy(cmd.substr(0, eq));
    if (assign.target.empty() || !is_identifier(assign.target)) {
        return false;
    }
    const std::string rhs = trim_copy(cmd.substr(eq + 1));
    const auto open = rhs.find('(');
    if (open == std::string::npos) {
        return false;
    }
    assign.callee = lower(trim_copy(rhs.substr(0, open)));
    if (!is_two_scalar_matrix_mixed_call_callee(assign.callee)) {
        return false;
    }
    const auto call_args = split_call_args(rhs);
    if (!call_args || call_args->size() != 3) {
        return false;
    }
    assign.arg_scalar_a = trim_copy(call_args->at(0));
    assign.arg_scalar_b = trim_copy(call_args->at(1));
    assign.arg_matrix = trim_copy(call_args->at(2));
    return !assign.arg_scalar_a.empty() && !assign.arg_scalar_b.empty() &&
           !assign.arg_matrix.empty();
}

Result<double> eval_ml_metric(const std::string& callee, const ml::Vec& y_pred, const ml::Vec& y_true) {
    if (y_pred.size() != y_true.size()) {
        return std::unexpected(DomainError{callee.c_str(), "vector length mismatch"});
    }
    if (callee == "ml_accuracy") {
        return ml::accuracy(y_pred, y_true);
    }
    if (callee == "ml_rmse") {
        return ml::rmse(y_pred, y_true);
    }
    if (callee == "ml_mse") {
        return ml::mse_loss(y_pred, y_true);
    }
    if (callee == "ml_r2") {
        return ml::r2_score(y_pred, y_true);
    }
    if (callee == "ml_f1") {
        return ml::f1_score(y_pred, y_true);
    }
    if (callee == "ml_precision") {
        return ml::precision(y_pred, y_true);
    }
    if (callee == "ml_recall") {
        return ml::recall(y_pred, y_true);
    }
    if (callee == "ml_mae") {
        return ml::mae_loss(y_pred, y_true);
    }
    if (callee == "ml_huber") {
        return ml::huber_loss(y_pred, y_true, 1.0);
    }
    if (callee == "ml_hinge") {
        return ml::hinge_loss(y_pred, y_true);
    }
    if (callee == "ml_binary_crossentropy") {
        return ml::binary_crossentropy(y_pred, y_true);
    }
    if (callee == "ml_vec_dot") {
        return ml::vec_dot(y_pred, y_true);
    }
    return std::unexpected(DomainError{callee.c_str(), "unsupported ml metric"});
}

bool try_parse_bigint_unary_call(const std::string& line, std::string& name, std::string& fn,
                                 int& n) {
    static const std::regex pattern(
        R"((\w+)\s*=\s*(bigint_factorial|bigint_fib)\s*\(\s*(-?\d+)\s*\))",
        std::regex::icase);
    const std::string trimmed = trim_copy(line);
    std::smatch match;
    if (!std::regex_match(trimmed, match, pattern)) {
        return false;
    }
    name = match[1].str();
    fn = lower(match[2].str());
    n = std::stoi(match[3].str());
    return is_identifier(name) && n >= 0;
}

bool try_parse_bigint_gcd_assignment(const std::string& line, std::string& name, std::string& a,
                                     std::string& b) {
    static const std::regex pattern(
        R"((\w+)\s*=\s*bigint_gcd\s*\(\s*(\"([^\"]*)\"|'([^']*)')\s*,\s*(\"([^\"]*)\"|'([^']*)')\s*\))",
        std::regex::icase);
    const std::string trimmed = trim_copy(line);
    std::smatch match;
    if (!std::regex_match(trimmed, match, pattern)) {
        return false;
    }
    name = match[1].str();
    a = match[3].matched ? match[3].str() : match[4].str();
    b = match[6].matched ? match[6].str() : match[7].str();
    return is_identifier(name);
}

Result<double> eval_bigint_unary(const std::string& fn, int n) {
    if (n < 0) {
        return std::unexpected(DomainError{fn.c_str(), "expected non-negative integer"});
    }
    const bignum::BigInt value =
        fn == "bigint_factorial" ? bignum::bigint_factorial(n) : bignum::bigint_fibonacci(n);
    return bigint_to_scalar(value, fn.c_str());
}

Result<double> eval_bigint_gcd_strings(const std::string& a, const std::string& b) {
    if (a.empty() || b.empty()) {
        return std::unexpected(DomainError{"bigint_gcd", "expected decimal string literals"});
    }
    return bigint_to_scalar(bignum::bigint_gcd(bignum::BigInt(a), bignum::BigInt(b)), "bigint_gcd");
}

Result<SymExpr> parse_sym_quoted_expr(const std::string& quoted_arg, const char* fn) {
    std::string expr_text;
    if (!parse_quoted_string(quoted_arg, expr_text)) {
        return std::unexpected(DomainError{fn, "expected quoted expression string"});
    }
    auto parsed = sym_parse(expr_text);
    if (!parsed) {
        return std::unexpected(DomainError{fn, parsed.error().message});
    }
    return Result<SymExpr>(std::in_place, std::move(*parsed));
}

Result<std::string> eval_sym_diff_strings(const std::string& expr_arg, const std::string& var_arg) {
    auto expr = parse_sym_quoted_expr(expr_arg, "sym_diff");
    if (!expr) {
        return std::unexpected(expr.error());
    }
    std::string var_text;
    if (!parse_quoted_string(var_arg, var_text) || var_text.empty()) {
        return std::unexpected(DomainError{"sym_diff", "expected sym_diff(\"expr\", \"var\")"});
    }
    const auto result = sym_simplify(sym_diff(std::move(*expr), var_text));
    return sym_to_string(result) + "\n";
}

Result<std::string> eval_sym_simplify_string(const std::string& expr_arg) {
    auto expr = parse_sym_quoted_expr(expr_arg, "sym_simplify");
    if (!expr) {
        return std::unexpected(expr.error());
    }
    const auto result = sym_simplify(std::move(*expr));
    return sym_to_string(result) + "\n";
}

Result<std::string> eval_sym_integrate_strings(const std::string& expr_arg, const std::string& var_arg) {
    auto expr = parse_sym_quoted_expr(expr_arg, "sym_integrate");
    if (!expr) {
        return std::unexpected(expr.error());
    }
    std::string var_text;
    if (!parse_quoted_string(var_arg, var_text) || var_text.empty()) {
        return std::unexpected(
            DomainError{"sym_integrate", "expected sym_integrate(\"expr\", \"var\")"});
    }
    const auto result = sym_integrate(*expr, var_text);
    return sym_to_string(result) + "\n";
}

Result<std::string> eval_sym_eval_strings(const std::string& expr_arg, const std::string& binding_arg) {
    auto expr = parse_sym_quoted_expr(expr_arg, "sym_eval");
    if (!expr) {
        return std::unexpected(expr.error());
    }
    std::string binding;
    if (!parse_quoted_string(binding_arg, binding)) {
        return std::unexpected(DomainError{"sym_eval", "expected sym_eval(\"expr\", \"var=value\")"});
    }
    const auto eq_pos = binding.find('=');
    if (eq_pos == std::string::npos || eq_pos == 0 || eq_pos + 1 >= binding.size()) {
        return std::unexpected(DomainError{"sym_eval", "expected var=value binding"});
    }
    const std::string var = binding.substr(0, eq_pos);
    const std::string value_text = binding.substr(eq_pos + 1);
    double value = 0.0;
    if (!parse_number(value_text, value)) {
        return std::unexpected(DomainError{"sym_eval", "expected numeric value in var=value binding"});
    }
    return std::to_string(sym_eval(*expr, {{var, value}})) + "\n";
}

Result<std::string> format_ode_trajectory(const OdeResult& result) {
    if (result.t.size() != result.y.size()) {
        return std::unexpected(DomainError{"ode", "internal trajectory size mismatch"});
    }
    Matrix<double> out(result.t.size(), 2);
    for (size_t i = 0; i < result.t.size(); ++i) {
        out(i, 0) = result.t[i];
        out(i, 1) = result.y[i];
    }
    std::ostringstream oss;
    oss << "traj =\n";
    oss << std::fixed << std::setprecision(6);
    for (size_t i = 0; i < out.rows(); ++i) {
        oss << "  [";
        for (size_t j = 0; j < out.cols(); ++j) {
            if (j > 0) {
                oss << ", ";
            }
            oss << out(i, j);
        }
        oss << "]\n";
    }
    return oss.str();
}

Result<std::string> eval_ode_fixed_step_call(const std::string& fn, const std::string& formula_arg,
                                             const std::string& t0_arg,
                                             const std::string& y0_arg,
                                             const std::string& t_end_arg,
                                             const std::string& steps_arg,
                                             OdeResult (*solver)(OdeFunc, double, double, double,
                                                                 size_t)) {
    auto expr = parse_sym_quoted_expr(formula_arg, fn.c_str());
    if (!expr) {
        return std::unexpected(expr.error());
    }
    double t0 = 0.0;
    double y0 = 0.0;
    double t_end = 0.0;
    double steps_d = 0.0;
    if (!parse_number(trim_copy(t0_arg), t0) || !parse_number(trim_copy(y0_arg), y0) ||
        !parse_number(trim_copy(t_end_arg), t_end) ||
        !parse_number(trim_copy(steps_arg), steps_d)) {
        return std::unexpected(DomainError{
            fn, std::string("expected ") + fn + "(\"formula\", t0, y0, t_end, steps)"});
    }
    const int steps_i = static_cast<int>(steps_d);
    if (steps_i < 0 || steps_d != steps_i) {
        return std::unexpected(DomainError{fn, "expected non-negative integer steps"});
    }
    SymExpr parsed = std::move(*expr);
    auto expr_ptr = std::make_shared<SymExpr>(std::move(parsed));
    OdeFunc f = [expr_ptr](double t, double y) {
        return sym_eval(*expr_ptr, {{"t", t}, {"y", y}});
    };
    return format_ode_trajectory(
        solver(f, t0, y0, t_end, static_cast<size_t>(steps_i)));
}

Result<std::string> eval_ode_rk45_call(const std::string& formula_arg, const std::string& t0_arg,
                                       const std::string& y0_arg, const std::string& t_end_arg,
                                       const std::string& rtol_arg, const std::string& atol_arg) {
    constexpr const char* fn = "ode_rk45";
    auto expr = parse_sym_quoted_expr(formula_arg, fn);
    if (!expr) {
        return std::unexpected(expr.error());
    }
    double t0 = 0.0;
    double y0 = 0.0;
    double t_end = 0.0;
    double rtol = 0.0;
    double atol = 0.0;
    if (!parse_number(trim_copy(t0_arg), t0) || !parse_number(trim_copy(y0_arg), y0) ||
        !parse_number(trim_copy(t_end_arg), t_end) ||
        !parse_number(trim_copy(rtol_arg), rtol) || !parse_number(trim_copy(atol_arg), atol)) {
        return std::unexpected(DomainError{
            fn, "expected ode_rk45(\"formula\", t0, y0, t_end, rtol, atol)"});
    }
    SymExpr parsed = std::move(*expr);
    auto expr_ptr = std::make_shared<SymExpr>(std::move(parsed));
    OdeFunc f = [expr_ptr](double t, double y) {
        return sym_eval(*expr_ptr, {{"t", t}, {"y", y}});
    };
    return format_ode_trajectory(ode_rk45(f, t0, y0, t_end, rtol, atol));
}

std::optional<Result<std::string>> try_eval_sym_command(const std::string& cmd) {
    std::smatch match;
    static const std::regex sym_binary(R"((\w+)\(([^,]+),([^)]+)\))", std::regex::icase);
    if (std::regex_match(cmd, match, sym_binary)) {
        const std::string fn = lower(match[1].str());
        const std::string arg_a = trim_copy(match[2].str());
        const std::string arg_b = trim_copy(match[3].str());
        if (fn == "sym_diff") {
            return eval_sym_diff_strings(arg_a, arg_b);
        }
        if (fn == "sym_integrate") {
            return eval_sym_integrate_strings(arg_a, arg_b);
        }
        if (fn == "sym_eval") {
            return eval_sym_eval_strings(arg_a, arg_b);
        }
    }
    static const std::regex sym_unary(R"((\w+)\(([^)]+)\))", std::regex::icase);
    if (std::regex_match(cmd, match, sym_unary)) {
        const std::string fn = lower(match[1].str());
        const std::string arg = trim_copy(match[2].str());
        if (fn == "sym_simplify") {
            return eval_sym_simplify_string(arg);
        }
    }
    return std::nullopt;
}

std::optional<std::vector<std::string>> split_call_args(const std::string& cmd) {
    const auto open = cmd.find('(');
    if (open == std::string::npos) {
        return std::nullopt;
    }
    const auto close = cmd.rfind(')');
    if (close == std::string::npos || close <= open) {
        return std::nullopt;
    }
    std::vector<std::string> args;
    std::string current;
    int depth = 0;
    for (size_t i = open + 1; i < close; ++i) {
        const char c = cmd[i];
        if (c == '[') {
            ++depth;
            current += c;
        } else if (c == ']') {
            --depth;
            current += c;
        } else if (c == ',' && depth == 0) {
            args.push_back(trim_copy(current));
            current.clear();
        } else {
            current += c;
        }
    }
    args.push_back(trim_copy(current));
    return args;
}

bool is_identifier(const std::string& text) {
    if (text.empty()) {
        return false;
    }
    const unsigned char first = static_cast<unsigned char>(text.front());
    if (!std::isalpha(first) && text.front() != '_') {
        return false;
    }
    for (unsigned char c : text) {
        if (!std::isalnum(c) && c != '_') {
            return false;
        }
    }
    return true;
}

bool try_parse_bigint_assignment(const std::string& line, std::string& name, std::string& decimal) {
    static const std::regex pattern(
        R"((\w+)\s*=\s*bigint\s*\(\s*(\"([^\"]*)\"|'([^']*)')\s*\))",
        std::regex::icase);
    const std::string trimmed = trim_copy(line);
    std::smatch match;
    if (!std::regex_match(trimmed, match, pattern)) {
        return false;
    }
    name = match[1].str();
    decimal = match[3].matched ? match[3].str() : match[4].str();
    return is_identifier(name);
}

bool parse_scalar_operand(const std::string& text, ScalarOperand& out) {
    const std::string token = trim_copy(text);
    double value = 0.0;
    if (parse_number(token, value)) {
        out.is_literal = true;
        out.literal = value;
        out.name.clear();
        return true;
    }
    if (is_identifier(token)) {
        out.is_literal = false;
        out.literal = 0.0;
        out.name = token;
        return true;
    }
    return false;
}

bool is_binary_minus(const std::string& expr, size_t index) {
    size_t j = index;
    while (j > 0 && std::isspace(static_cast<unsigned char>(expr[j - 1]))) {
        --j;
    }
    if (j == 0) {
        return false;
    }
    const char prev = expr[j - 1];
    return prev != '+' && prev != '-' && prev != '*' && prev != '/' && prev != '(';
}

std::optional<std::pair<size_t, char>> find_top_level_op(const std::string& expr, const char* ops) {
    int depth = 0;
    std::optional<std::pair<size_t, char>> last;
    for (size_t i = 0; i < expr.size(); ++i) {
        const char c = expr[i];
        if (c == '(') {
            ++depth;
        } else if (c == ')' && depth > 0) {
            --depth;
        } else if (depth == 0) {
            for (const char* p = ops; *p != '\0'; ++p) {
                if (c == *p) {
                    if (c == '-' && !is_binary_minus(expr, i)) {
                        continue;
                    }
                    last = std::pair{i, *p};
                }
            }
        }
    }
    return last;
}

std::optional<std::pair<size_t, char>> find_scalar_binop(const std::string& rhs) {
    if (auto add_sub = find_top_level_op(rhs, "+-")) {
        return add_sub;
    }
    return find_top_level_op(rhs, "*/");
}

std::string strip_outer_parens(std::string expr) {
    while (true) {
        expr = trim_copy(expr);
        if (expr.size() < 2 || expr.front() != '(' || expr.back() != ')') {
            return expr;
        }
        int depth = 0;
        bool wraps_all = true;
        for (size_t i = 0; i < expr.size(); ++i) {
            if (expr[i] == '(') {
                ++depth;
            } else if (expr[i] == ')') {
                --depth;
                if (depth == 0 && i + 1 != expr.size()) {
                    wraps_all = false;
                    break;
                }
            }
        }
        if (!wraps_all) {
            return expr;
        }
        expr = expr.substr(1, expr.size() - 2);
    }
}

bool contains_scalar_operator(const std::string& rhs) {
    return find_scalar_binop(rhs).has_value();
}

std::optional<std::pair<std::string, std::string>> parse_scalar_unary_call(const std::string& expr) {
    const std::string text = trim_copy(expr);
    const auto open = text.find('(');
    if (open == std::string::npos || open == 0) {
        return std::nullopt;
    }
    const std::string name = trim_copy(text.substr(0, open));
    if (!is_identifier(name)) {
        return std::nullopt;
    }

    int depth = 0;
    size_t close = std::string::npos;
    for (size_t i = open; i < text.size(); ++i) {
        if (text[i] == '(') {
            ++depth;
        } else if (text[i] == ')') {
            --depth;
            if (depth == 0) {
                close = i;
                break;
            }
        }
    }
    if (close == std::string::npos || close + 1 != text.size()) {
        return std::nullopt;
    }

    const std::string arg = trim_copy(text.substr(open + 1, close - open - 1));
    if (arg.empty()) {
        return std::nullopt;
    }
    return std::pair{name, arg};
}

std::optional<std::string> parse_nullary_scalar_call(const std::string& expr) {
    const std::string text = trim_copy(expr);
    const auto open = text.find('(');
    if (open == std::string::npos || open == 0) {
        return std::nullopt;
    }
    const std::string name = lower(trim_copy(text.substr(0, open)));
    if (!is_identifier(name) || !is_nullary_scalar_callee(name)) {
        return std::nullopt;
    }

    int depth = 0;
    size_t close = std::string::npos;
    for (size_t i = open; i < text.size(); ++i) {
        if (text[i] == '(') {
            ++depth;
        } else if (text[i] == ')') {
            --depth;
            if (depth == 0) {
                close = i;
                break;
            }
        }
    }
    if (close == std::string::npos || close + 1 != text.size()) {
        return std::nullopt;
    }

    const std::string arg = trim_copy(text.substr(open + 1, close - open - 1));
    if (!arg.empty()) {
        return std::nullopt;
    }
    return name;
}

std::optional<std::string> parse_nullary_matrix_call(const std::string& expr) {
    const std::string text = trim_copy(expr);
    const auto open = text.find('(');
    if (open == std::string::npos || open == 0) {
        return std::nullopt;
    }
    const std::string name = lower(trim_copy(text.substr(0, open)));
    if (!is_identifier(name) || !is_nullary_matrix_callee(name)) {
        return std::nullopt;
    }

    int depth = 0;
    size_t close = std::string::npos;
    for (size_t i = open; i < text.size(); ++i) {
        if (text[i] == '(') {
            ++depth;
        } else if (text[i] == ')') {
            --depth;
            if (depth == 0) {
                close = i;
                break;
            }
        }
    }
    if (close == std::string::npos || close + 1 != text.size()) {
        return std::nullopt;
    }

    const std::string arg = trim_copy(text.substr(open + 1, close - open - 1));
    if (!arg.empty()) {
        return std::nullopt;
    }
    return name;
}

std::optional<std::pair<std::string, std::string>> parse_unary_scalar_matrix_call(
    const std::string& expr) {
    const auto unary = parse_scalar_unary_call(expr);
    if (!unary) {
        return std::nullopt;
    }
    const std::string name = lower(trim_copy(unary->first));
    if (!is_unary_scalar_matrix_callee(name)) {
        return std::nullopt;
    }

    std::vector<std::string> args = split_scalar_call_args(unary->second);
    args.erase(std::remove_if(args.begin(), args.end(),
                              [](const std::string& arg) { return arg.empty(); }),
               args.end());
    if (args.size() != 1) {
        return std::nullopt;
    }
    return std::pair{name, args[0]};
}

std::vector<std::string> split_scalar_call_args(const std::string& args_text) {
    std::vector<std::string> args;
    std::string current;
    int paren_depth = 0;
    int bracket_depth = 0;
    for (char c : args_text) {
        if (c == '(') {
            ++paren_depth;
            current += c;
        } else if (c == ')') {
            --paren_depth;
            current += c;
        } else if (c == '[') {
            ++bracket_depth;
            current += c;
        } else if (c == ']') {
            --bracket_depth;
            current += c;
        } else if (c == ',' && paren_depth == 0 && bracket_depth == 0) {
            args.push_back(trim_copy(current));
            current.clear();
        } else {
            current += c;
        }
    }
    args.push_back(trim_copy(current));
    return args;
}

std::optional<std::pair<std::string, std::vector<std::string>>> parse_scalar_call(
    const std::string& expr) {
    const auto unary = parse_scalar_unary_call(expr);
    if (!unary) {
        return std::nullopt;
    }

    std::vector<std::string> args = split_scalar_call_args(unary->second);
    args.erase(std::remove_if(args.begin(), args.end(),
                              [](const std::string& arg) { return arg.empty(); }),
               args.end());
    if (args.empty()) {
        return std::nullopt;
    }
    return std::pair{unary->first, std::move(args)};
}

bool is_scalar_expression_rhs(const std::string& rhs) {
    const std::string text = trim_copy(rhs);
    if (text.empty()) {
        return false;
    }
    if (text.size() >= 2 && text.front() == '[' && text.back() == ']') {
        return false;
    }
    double literal = 0.0;
    if (parse_number(text, literal)) {
        return false;
    }
    if (text.front() == '-' || text.front() == '+') {
        return is_scalar_expression_rhs(text.substr(1));
    }
    if (parse_nullary_scalar_call(text)) {
        return false;
    }
    if (parse_nullary_matrix_call(text)) {
        return false;
    }
    if (parse_unary_scalar_matrix_call(text)) {
        return false;
    }
    if (const auto call = parse_scalar_call(text)) {
        const std::string fn = lower(call->first);
        if (fn == "matmul" || fn == "tensorops_matmul" || fn == "tensorops_einsum" ||
            fn == "solve" || fn == "transpose" || fn == "chol" || fn == "det" ||
            fn == "trace" || fn == "norm" || fn == "rank" || fn == "cond" || fn == "lu" ||
            fn == "qr" || fn == "svd" || fn == "eig_sym" ||
            fn == "zeros" || fn == "eye" || fn == "ones" ||
            fn == "rand" || fn == "randn" || fn == "expm" || fn == "inv" ||
            fn == "pinv" || fn == "null" || fn == "orth" ||
            fn == "kron" || fn == "repmat" || fn == "linspace" ||
            fn == "rgb2gray" || fn == "sobel" || fn == "prewitt" ||
            fn == "imgaussfilt" || fn == "medfilt2" || fn == "boxfilter" ||
            fn == "bilateral" || fn == "canny" ||
            fn == "laplacian" || fn == "histeq" ||
            fn == "sharpen" || fn == "threshold_otsu" || fn == "imresize" ||
            fn == "rle_encode_vec" || fn == "rle_decode_vec" ||
            fn == "mtf_encode_vec" || fn == "mtf_decode_vec" ||
            fn == "lzw_encode_vec" || fn == "lzw_decode_vec" ||
            fn == "huffman_encode_vec" || fn == "huffman_decode_vec" ||
            fn == "lz77_encode_vec" || fn == "lz77_decode_vec" ||
            fn == "bzip2_compress_vec" || fn == "bzip2_decompress_vec" ||
            fn == "bwt_encode_vec" || fn == "bwt_decode_vec" ||
            fn == "delta_encode_vec" || fn == "delta_decode_vec" ||
            fn == "control_is_controllable" || fn == "control_is_observable" ||
            fn == "control_lyap" || fn == "control_lqr" || fn == "control_place" ||
            fn == "control_dlyap" || fn == "control_riccati" || fn == "control_dare" ||
            fn == "control_bode_mag_db" || fn == "control_bode_phase" ||
            fn == "control_bode" ||
            fn == "topo_bottleneck_distance" ||
            fn == "topo_wasserstein_distance" ||
            fn == "topo_persistence_diagram" ||
            fn == "quantum_op_apply" ||
            fn == "graph_diameter" ||
            fn == "graph_radius" ||
            fn == "graph_betweenness" ||
            fn == "graph_closeness" ||
            fn == "graph_degree_centrality" ||
            fn == "graph_is_bipartite" ||
            fn == "graph_is_dag" || fn == "graph_is_connected" ||
            fn == "graph_is_tree" || fn == "graph_topological_sort" ||
            fn == "graph_greedy_colour" || fn == "graph_euler_circuit" ||
            fn == "graph_scc" || fn == "count_components" ||
            fn == "stats_mean" || fn == "stats_median" ||
            fn == "stats_stddev" || fn == "stats_skewness" ||
            fn == "stats_kurtosis" || fn == "stats_var" || fn == "stats_mode" ||
            fn == "stats_geometric_mean" || fn == "stats_harmonic_mean" ||
            fn == "stats_rms" || fn == "stats_mad" || fn == "stats_iqr" ||
            fn == "stats_min_value" ||
            fn == "count_components" ||
            fn == "fft_rfft" || fn == "fft_dft" || fn == "geo_delaunay_2d" ||
            fn == "geo_voronoi" || fn == "topo_pairwise_distances" ||
            fn == "combo_next_perm" || fn == "numthy_convergents" ||
            fn == "ml_mat_transpose" || fn == "ml_mat_mul" || fn == "poly_deriv" ||
            fn == "poly_eval" || fn == "poly_integ" || fn == "poly_add" ||
            fn == "poly_mul" || fn == "poly_sub" || fn == "poly_compose" ||
            fn == "fft_irfft" || fn == "fft_ifft" || fn == "fft_fft2" || fn == "fft_dct2" || fn == "fft_idct2" || fn == "fft_dst2" || fn == "fftshift" ||
            fn == "graph_floyd_warshall" || fn == "graph_mst_kruskal" ||
            fn == "graph_mst_prim" ||
            fn == "graph_scc" ||
            fn == "signal_convolve" || fn == "signal_correlate" ||
            fn == "graph_max_flow" ||
            fn == "diffgeo_surface_normal_sphere" ||
            fn == "stats_correlation" || fn == "stats_spearman" ||
            fn == "stats_kendall" || fn == "stats_two_sample_ttest" ||
            fn == "stats_chi2_gof" ||
            fn == "signal_moving_average" || fn == "graph_bfs" || fn == "graph_dfs" ||
            fn == "stats_percentile" || fn == "graph_dfs" ||
            fn == "stats_percentile" || fn == "stats_ttest" || fn == "stats_acf" ||
            fn == "signal_lowpass" || fn == "signal_butterworth" || fn == "signal_highpass" ||
            fn == "signal_bandpass" ||
            fn == "quantum_commutator" || fn == "quantum_tensor_product" ||
            fn == "signal_hamming" || fn == "signal_hanning" || fn == "signal_blackman" ||
            fn == "signal_parzen" || fn == "signal_triangular" ||
            fn == "cplx_blaschke_product" ||
            fn == "compress_bits_to_bytes" ||
            fn == "compress_bytes_to_bits" ||
            fn == "imcrop" ||
            fn == "diffgeo_geodesic_euclidean" ||
            fn == "combo_all_permutations" ||
            fn == "combo_all_subsets" ||
            fn == "combo_all_compositions" ||
            fn == "combo_all_partitions" ||
            fn == "numthy_crt" || fn == "geo_centroid_x" || fn == "geo_centroid_y" ||
            fn == "bwt_primary_index" ||
            fn == "quantum_ket_superposition" || fn == "quantum_ghz_state" ||
            fn == "quantum_w_state" || fn == "quantum_bell_state" || fn == "quantum_coherent_state" ||
            fn == "numthy_divisors_vec" ||
            fn == "numthy_factor_vec" ||
            fn == "combo_unrank_permutation" || fn == "combo_unrank_combination" ||
            fn == "ml_accuracy" || fn == "ml_rmse" || fn == "ml_mse" ||
            fn == "ml_r2" || fn == "ml_f1" || fn == "ml_precision" ||
            fn == "ml_recall" || fn == "ml_mae" || fn == "ml_huber" ||
            fn == "ml_hinge" || fn == "ml_binary_crossentropy" ||
            fn == "ml_categorical_crossentropy" || fn == "ml_vec_dot" ||
            fn == "bigint" || fn == "bigint_factorial" || fn == "bigint_fib" ||
            fn == "bigint_gcd" ||
            fn == "sym_diff" || fn == "sym_integrate" || fn == "sym_eval" || fn == "sym_simplify" ||
            fn == "graph_pagerank" || fn == "graph_dijkstra_dist" ||
            fn == "graph_bellman_ford_dist" || fn == "graph_max_flow" ||
            fn == "graph_astar" ||
            fn == "geo_convex_hull_area" || fn == "geo_polygon_area" ||
            fn == "geo_polygon_perimeter" || fn == "geo_signed_area" ||
            fn == "geo_moment_of_inertia" || fn == "geo_point_in_polygon" ||
            fn == "ml_vec_norm" ||
            fn == "quantum_hadamard" ||
            fn == "quantum_ket_normalise" ||
            fn == "quantum_density_matrix" ||
            fn == "quantum_ket_basis" || fn == "quantum_fock_state" ||
            fn == "quantum_identity_n" ||
            fn == "control_step_final" || fn == "control_impulse_final" ||
            fn == "control_dcgain" || fn == "control_pidtune_kp" ||
            fn == "control_pidtune_ki" || fn == "control_pidtune_kd" ||
            fn == "control_is_stable" ||
            fn == "finance_npv" || fn == "geo_bezier_eval_x" || fn == "geo_bezier_eval_y" ||
            fn == "bwt_decode_vec" || fn == "combo_multinomial" ||
            fn == "quantum_time_evolution" ||
            fn == "combo_rank_permutation" || fn == "combo_rank_combination" ||
            fn == "finance_sharpe" || fn == "finance_sortino" ||
            fn == "finance_irr" || fn == "finance_var" || fn == "finance_cvar" ||
            fn == "finance_max_drawdown" || fn == "finance_portfolio_return" ||
            fn == "finance_portfolio_variance" ||
            fn == "info_entropy" || fn == "info_lz_complexity" || fn == "info_mutual_info" ||
            fn == "info_redundancy" || fn == "info_efficiency" ||
            fn == "info_kl_divergence" || fn == "info_js_divergence" ||
            fn == "info_cross_entropy" || fn == "info_tv_distance" ||
            fn == "info_hellinger_dist" || fn == "info_renyi_entropy" ||
            fn == "info_source_coding_rate" || fn == "info_tsallis_entropy" ||
            fn == "tensorops_norm" ||
            fn == "quantum_von_neumann_entropy" || fn == "quantum_concurrence" ||
            fn == "quantum_fidelity" || fn == "quantum_trace_distance" ||
            fn == "quantum_expectation" || fn == "quantum_expectation_dm" ||
            fn == "quantum_inner" || fn == "quantum_entanglement_entropy" ||
            fn == "quantum_partial_trace" || fn == "quantum_schrodinger" ||
            fn == "quantum_schrodinger_final" ||
            fn == "pde_heat_1d" || fn == "pde_heat_2d" || fn == "pde_wave_1d" ||
            fn == "pde_advection_1d" || fn == "pde_poisson_2d" || fn == "pde_burgers_1d" ||
            fn == "quantum_time_evolution" ||
            fn == "info_joint_entropy" ||
            fn == "cplx_power_series_eval" || fn == "cplx_winding_number" ||
            fn == "topo_vietoris_rips_betti0" || fn == "topo_betti_curve" ||
            fn == "info_conditional_entropy" || fn == "info_sample_entropy" ||
            fn == "geo_kdtree_nearest" ||
            fn == "stats_ztest" ||
            fn == "finance_binomial_call" ||
            fn == "finance_binomial_put" || fn == "finance_bs_delta" ||
            fn == "finance_bs_theta" || fn == "finance_bs_rho" ||
            fn == "finance_portfolio_return" || fn == "finance_portfolio_variance" ||
            fn == "tensorops_matmul" || fn == "tensorops_einsum") {
            return false;
        }
    }
    return contains_scalar_operator(text) || parse_scalar_call(text).has_value();
}

Result<double> resolve_scalar_operand(const SessionState& state, const ScalarOperand& operand) {
    if (operand.is_literal) {
        return operand.literal;
    }
    const auto it = state.scalars.find(operand.name);
    if (it == state.scalars.end()) {
        return std::unexpected(DomainError{"resolve", "unknown scalar: " + operand.name});
    }
    return it->second;
}

Result<double> eval_scalar_expr(const SessionState& state, const std::string& expr_text) {
    const std::string expr = trim_copy(strip_outer_parens(expr_text));
    if (expr.empty()) {
        return std::unexpected(DomainError{"eval", "empty expression"});
    }

    if (expr.front() == '-') {
        auto inner = eval_scalar_expr(state, expr.substr(1));
        if (!inner) {
            return std::unexpected(inner.error());
        }
        return -(*inner);
    }
    if (expr.front() == '+') {
        return eval_scalar_expr(state, expr.substr(1));
    }

    if (const auto call = parse_scalar_call(expr)) {
        std::vector<double> arg_values;
        arg_values.reserve(call->second.size());
        for (const auto& arg_text : call->second) {
            auto arg = eval_scalar_expr(state, arg_text);
            if (!arg) {
                return std::unexpected(arg.error());
            }
            arg_values.push_back(*arg);
        }
        return Interpreter::eval_scalar_call(call->first, arg_values);
    }

    ScalarOperand single;
    if (parse_scalar_operand(expr, single)) {
        return resolve_scalar_operand(state, single);
    }

    const auto op_pos = find_scalar_binop(expr);
    if (!op_pos) {
        return std::unexpected(DomainError{"eval", "invalid scalar expression"});
    }

    auto left = eval_scalar_expr(state, expr.substr(0, op_pos->first));
    if (!left) {
        return std::unexpected(left.error());
    }
    auto right = eval_scalar_expr(state, expr.substr(op_pos->first + 1));
    if (!right) {
        return std::unexpected(right.error());
    }
    return Interpreter::eval_scalar_op(op_pos->second, *left, *right);
}

void append_unique_var(std::vector<std::string>& vars, const std::string& name) {
    if (std::find(vars.begin(), vars.end(), name) == vars.end()) {
        vars.push_back(name);
    }
}

void collect_scalar_expr_variables(const std::string& expr_text, std::vector<std::string>& vars) {
    const std::string expr = strip_outer_parens(expr_text);
    if (expr.empty()) {
        return;
    }

    if (expr.front() == '-' || expr.front() == '+') {
        collect_scalar_expr_variables(expr.substr(1), vars);
        return;
    }

    if (const auto call = parse_scalar_call(expr)) {
        for (const auto& arg_text : call->second) {
            collect_scalar_expr_variables(arg_text, vars);
        }
        return;
    }

    ScalarOperand single;
    if (parse_scalar_operand(expr, single)) {
        if (!single.is_literal) {
            append_unique_var(vars, single.name);
        }
        return;
    }

    const auto op_pos = find_scalar_binop(expr);
    if (!op_pos) {
        return;
    }
    collect_scalar_expr_variables(expr.substr(0, op_pos->first), vars);
    collect_scalar_expr_variables(expr.substr(op_pos->first + 1), vars);
}

} // namespace

std::string Interpreter::trim(std::string s) {
    auto not_space = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}

Result<Matrix<double>> Interpreter::parse_matrix(const std::string& text) const {
    std::string s = trim(text);
    if (s.size() < 2 || s.front() != '[' || s.back() != ']') {
        return std::unexpected(DomainError{"parse_matrix", "expected [ ... ]"});
    }
    s = s.substr(1, s.size() - 2);
    if (s.empty()) {
        return Matrix<double>(0, 0);
    }

    std::vector<std::vector<double>> rows;
    std::stringstream row_stream(s);
    std::string row_text;
    while (std::getline(row_stream, row_text, ';')) {
        row_text = trim(row_text);
        if (row_text.empty()) {
            continue;
        }
        std::vector<double> row;
        std::stringstream col_stream(row_text);
        std::string cell;
        while (std::getline(col_stream, cell, ',')) {
            cell = trim(cell);
            double value = 0.0;
            if (!parse_number(cell, value)) {
                return std::unexpected(DomainError{"parse_matrix", "invalid number: " + cell});
            }
            row.push_back(value);
        }
        rows.push_back(std::move(row));
    }

    if (rows.empty()) {
        return std::unexpected(DomainError{"parse_matrix", "no rows found"});
    }
    const size_t cols = rows[0].size();
    for (const auto& row : rows) {
        if (row.size() != cols) {
            return std::unexpected(DimensionMismatch{rows.size(), row.size()});
        }
    }

    Matrix<double> m(rows.size(), cols);
    for (size_t i = 0; i < rows.size(); ++i) {
        for (size_t j = 0; j < cols; ++j) {
            m(i, j) = rows[i][j];
        }
    }
    return m;
}

bool Interpreter::try_parse_scalar_assignment(const std::string& line, std::string& name, double& value) {
    const std::string cmd = trim(line);
    const auto eq = cmd.find('=');
    if (eq == std::string::npos) {
        return false;
    }
    name = trim(cmd.substr(0, eq));
    if (name.empty()) {
        return false;
    }
    return parse_number(trim(cmd.substr(eq + 1)), value);
}

Result<std::string> Interpreter::assign_scalar(const std::string& name, double value) {
    state_.scalars[name] = value;
    return name + " = " + std::to_string(value) + "\n";
}

bool Interpreter::try_parse_scalar_binary_assignment(const std::string& line, ScalarBinaryAssign& assign) {
    const std::string cmd = trim(line);
    const auto eq = cmd.find('=');
    if (eq == std::string::npos) {
        return false;
    }
    assign.target = trim(cmd.substr(0, eq));
    if (assign.target.empty() || !is_identifier(assign.target)) {
        return false;
    }
    const std::string rhs = trim(cmd.substr(eq + 1));
    const auto op_pos = find_scalar_binop(rhs);
    if (!op_pos) {
        return false;
    }
    assign.op = op_pos->second;
    const std::string left_text = trim_copy(rhs.substr(0, op_pos->first));
    const std::string right_text = trim_copy(rhs.substr(op_pos->first + 1));
    if (!parse_scalar_operand(left_text, assign.left) || !parse_scalar_operand(right_text, assign.right)) {
        return false;
    }
    return true;
}

bool Interpreter::try_parse_scalar_expr_assignment(const std::string& line, std::string& name,
                                                     std::string& expr) {
    const std::string cmd = trim(line);
    const auto eq = cmd.find('=');
    if (eq == std::string::npos) {
        return false;
    }
    name = trim(cmd.substr(0, eq));
    if (name.empty() || !is_identifier(name)) {
        return false;
    }
    expr = trim(cmd.substr(eq + 1));
    if (expr.empty()) {
        return false;
    }
    double literal = 0.0;
    if (parse_number(expr, literal)) {
        return false;
    }
    return is_scalar_expression_rhs(expr);
}

bool is_matrix_call_callee(const std::string& callee) {
    return callee == "matmul" || callee == "tensorops_matmul" || callee == "tensorops_einsum" ||
           callee == "solve" ||
           callee == "transpose" || callee == "chol" ||
           callee == "zeros" || callee == "eye" || callee == "ones" ||
           callee == "rand" || callee == "randn" ||
           callee == "expm" || callee == "inv" ||
           callee == "pinv" || callee == "null" || callee == "orth" ||
           callee == "kron" || callee == "repmat" || callee == "linspace" ||
           callee == "rgb2gray" || callee == "sobel" || callee == "prewitt" ||
           callee == "imgaussfilt" || callee == "medfilt2" || callee == "boxfilter" ||
           callee == "bilateral" ||
           callee == "canny" || callee == "laplacian" || callee == "histeq" ||
           callee == "sharpen" || callee == "threshold_otsu" || callee == "imresize" ||
           callee == "rle_encode_vec" || callee == "rle_decode_vec" ||
           callee == "mtf_encode_vec" || callee == "mtf_decode_vec" ||
           callee == "lzw_encode_vec" || callee == "lzw_decode_vec" ||
           callee == "lz77_encode_vec" || callee == "lz77_decode_vec" ||
           callee == "bzip2_compress_vec" || callee == "bzip2_decompress_vec" ||
           callee == "huffman_encode_vec" ||
           callee == "bwt_encode_vec" ||
           callee == "delta_encode_vec" || callee == "delta_decode_vec" ||
           callee == "graph_pagerank" || callee == "quantum_hadamard" ||
           callee == "quantum_ket_normalise" || callee == "quantum_density_matrix" ||
           callee == "quantum_ket_basis" ||
           callee == "quantum_ket_superposition" ||
           callee == "quantum_fock_state" || callee == "quantum_coherent_state" ||
           callee == "quantum_partial_trace" || callee == "quantum_schrodinger" ||
           callee == "quantum_schrodinger_final" ||
           callee == "pde_heat_1d" || callee == "pde_heat_2d" || callee == "pde_wave_1d" ||
           callee == "pde_advection_1d" || callee == "pde_poisson_2d" ||
           callee == "pde_burgers_1d" ||
           callee == "topo_betti_curve" || callee == "control_bode" ||
           callee == "compress_bits_to_bytes" ||
           callee == "compress_bytes_to_bits" ||
           callee == "graph_betweenness" ||
           callee == "graph_closeness" ||
           callee == "graph_degree_centrality" ||
           callee == "fft_rfft" || callee == "fft_dft" || callee == "fft_ifft" || callee == "fft_fft2" ||
           callee == "geo_delaunay_2d" || callee == "geo_voronoi" ||
           callee == "topo_pairwise_distances" || callee == "combo_next_perm" ||
           callee == "numthy_convergents" || callee == "ml_mat_transpose" ||
           callee == "ml_mat_mul" ||
           callee == "poly_deriv" ||
           callee == "graph_floyd_warshall" || callee == "graph_mst_kruskal" ||
           callee == "graph_mst_prim" ||            callee == "graph_topological_sort" ||
           callee == "graph_greedy_colour" || callee == "graph_euler_circuit" ||
           callee == "graph_scc" ||
           callee == "fft_dct2" || callee == "fft_idct2" || callee == "fft_dst2" ||
           callee == "fftshift" ||
           callee == "diffgeo_surface_normal_sphere" ||
           callee == "imcrop" ||
           callee == "diffgeo_geodesic_euclidean" ||
           callee == "combo_unrank_permutation" || callee == "combo_unrank_combination" ||
           callee == "numthy_continued_fraction" || callee == "numthy_primes";
}

bool is_valid_matrix_call_arity(const std::string& callee, size_t arity) {
    if (callee == "matmul" || callee == "tensorops_matmul" || callee == "tensorops_einsum" ||
        callee == "solve" ||
        callee == "rand" || callee == "randn" ||
        callee == "kron") {
        return arity == 2;
    }
    if (callee == "transpose" || callee == "chol" || callee == "expm" || callee == "inv" ||
        callee == "pinv" || callee == "null" || callee == "orth" ||
        callee == "rgb2gray" || callee == "sobel" || callee == "prewitt" ||
        callee == "laplacian" || callee == "histeq" || callee == "sharpen" ||
        callee == "threshold_otsu" ||
        callee == "rle_encode_vec" || callee == "rle_decode_vec" ||
        callee == "mtf_encode_vec" || callee == "mtf_decode_vec" ||
        callee == "lzw_encode_vec" || callee == "lzw_decode_vec" ||
        callee == "lz77_decode_vec" ||
        callee == "bzip2_compress_vec" || callee == "bzip2_decompress_vec" ||
        callee == "huffman_encode_vec" ||
        callee == "bwt_encode_vec" ||
        callee == "delta_encode_vec" || callee == "delta_decode_vec" ||
        callee == "graph_pagerank" || callee == "quantum_hadamard" ||
        callee == "quantum_ket_normalise" || callee == "quantum_ket_superposition" ||
        callee == "quantum_density_matrix") {
        return arity == 1;
    }
    if (callee == "compress_bits_to_bytes" || callee == "compress_bytes_to_bits") {
        return arity == 1;
    }
    if (callee == "graph_betweenness" || callee == "graph_closeness" ||
        callee == "graph_degree_centrality" || callee == "fft_rfft" || callee == "fft_dft" ||
        callee == "fft_ifft" || callee == "fft_fft2" ||
        callee == "geo_delaunay_2d" || callee == "geo_voronoi" ||
        callee == "topo_pairwise_distances" || callee == "combo_next_perm" ||
        callee == "numthy_convergents" || callee == "ml_mat_transpose" ||
        callee == "poly_deriv" ||
        callee == "graph_floyd_warshall" || callee == "graph_mst_kruskal" ||
        callee == "graph_mst_prim" ||            callee == "graph_topological_sort" ||
           callee == "graph_greedy_colour" || callee == "graph_euler_circuit" ||
           callee == "graph_scc" ||
        callee == "fft_dct2" || callee == "fft_idct2" || callee == "fft_dst2" ||
        callee == "fftshift" || callee == "count_components") {
        return arity == 1;
    }
    if (callee == "signal_moving_average" || callee == "poly_eval" ||
        callee == "graph_bfs" || callee == "graph_dfs" || callee == "stats_percentile" ||
        callee == "stats_ttest" || callee == "stats_acf" ||
        callee == "fft_irfft" || callee == "poly_integ" || callee == "combo_next_comb") {
        return arity == 2;
    }
    if (callee == "signal_lowpass" || callee == "signal_butterworth" ||
        callee == "signal_highpass" || callee == "signal_bandpass") {
        return arity == 3 || (callee == "signal_bandpass" && arity == 4);
    }
    if (callee == "signal_convolve" || callee == "poly_add" ||
        callee == "poly_mul" || callee == "poly_sub" || callee == "poly_compose" ||
        callee == "quantum_tensor_product" || callee == "signal_correlate" ||
        callee == "ml_mat_mul") {
        return arity == 2;
    }
    if (callee == "quantum_commutator") {
        return arity == 2;
    }
    if (callee == "diffgeo_surface_normal_sphere") {
        return arity == 2;
    }
    if (callee == "control_bode") {
        return arity == 3;
    }
    if (callee == "diffgeo_geodesic_euclidean") {
        return arity == 5;
    }
    if (callee == "lz77_encode_vec") {
        return arity == 1 || arity == 3;
    }
    if (callee == "imgaussfilt" || callee == "medfilt2" || callee == "boxfilter") {
        return arity == 2;
    }
    if (callee == "imresize" || callee == "topo_betti_curve" || callee == "bilateral" ||
        callee == "canny") {
        return arity == 3;
    }
    if (callee == "quantum_partial_trace") {
        return arity == 4;
    }
    if (callee == "quantum_schrodinger" || callee == "quantum_schrodinger_final") {
        return arity == 5;
    }
    if (callee == "imcrop") {
        return arity == 5;
    }
    if (callee == "pde_heat_1d" || callee == "pde_advection_1d" ||
        callee == "pde_poisson_2d" || callee == "pde_burgers_1d") {
        return arity == 5;
    }
    if (callee == "pde_heat_2d" || callee == "pde_wave_1d") {
        return arity == 6;
    }
    if (callee == "quantum_ket_basis" || callee == "quantum_fock_state" ||
        callee == "combo_unrank_permutation" || callee == "numthy_continued_fraction" ||
        callee == "numthy_primes") {
        return arity == 2;
    }
    if (callee == "combo_unrank_combination") {
        return arity == 3;
    }
    if (callee == "quantum_coherent_state") {
        return arity == 3;
    }
    if (callee == "zeros" || callee == "eye" || callee == "ones") {
        return arity == 1 || arity == 2;
    }
    if (callee == "linspace") {
        return arity == 3;
    }
    if (callee == "repmat") {
        return arity == 3;
    }
    return false;
}

bool is_scalar_matrix_call_callee(const std::string& callee) {
    return callee == "det" || callee == "trace" || callee == "norm" || callee == "rank" ||
           callee == "cond" || callee == "geo_convex_hull_area" || callee == "geo_polygon_area" ||
           callee == "geo_polygon_perimeter" || callee == "geo_signed_area" ||
           callee == "geo_moment_of_inertia" || callee == "geo_centroid_x" ||
           callee == "geo_centroid_y" || callee == "bwt_primary_index" ||
           callee == "combo_rank_permutation" ||
           callee == "ml_vec_norm" ||
           callee == "info_entropy" || callee == "info_lz_complexity" ||
           callee == "info_mutual_info" || callee == "info_redundancy" ||
           callee == "info_efficiency" ||
           callee == "info_source_coding_rate" ||
           callee == "finance_sharpe" || callee == "finance_sortino" || callee == "finance_irr" ||
           callee == "finance_var" || callee == "finance_cvar" ||
           callee == "finance_max_drawdown" ||
           callee == "quantum_von_neumann_entropy" || callee == "quantum_concurrence" ||
           callee == "tensorops_norm" || callee == "graph_diameter" ||
           callee == "graph_radius" ||            callee == "graph_is_bipartite" ||
           callee == "graph_is_dag" || callee == "graph_is_connected" ||
           callee == "graph_is_tree" ||
           callee == "stats_mean" || callee == "stats_median" ||
           callee == "stats_stddev" || callee == "stats_skewness" ||
           callee == "stats_kurtosis" || callee == "stats_var" || callee == "stats_mode" ||
           callee == "stats_geometric_mean" || callee == "stats_harmonic_mean" ||
           callee == "stats_rms" || callee == "stats_mad" || callee == "stats_iqr" ||
           callee == "stats_min_value" || callee == "count_components";
}

std::vector<std::string> split_comma_list(const std::string& text) {
    std::vector<std::string> parts;
    std::string current;
    int paren_depth = 0;
    int bracket_depth = 0;
    for (char c : text) {
        if (c == '(') {
            ++paren_depth;
            current += c;
        } else if (c == ')') {
            --paren_depth;
            current += c;
        } else if (c == '[') {
            ++bracket_depth;
            current += c;
        } else if (c == ']') {
            --bracket_depth;
            current += c;
        } else if (c == ',' && paren_depth == 0 && bracket_depth == 0) {
            parts.push_back(trim_copy(current));
            current.clear();
        } else {
            current += c;
        }
    }
    parts.push_back(trim_copy(current));
    parts.erase(std::remove_if(parts.begin(), parts.end(),
                               [](const std::string& part) { return part.empty(); }),
                parts.end());
    return parts;
}

bool is_multi_matrix_call_callee(const std::string& callee) {
    return callee == "lu" || callee == "qr" || callee == "svd" || callee == "eig_sym";
}

bool is_valid_multi_matrix_target_count(const std::string& callee, size_t count) {
    if (callee == "qr" || callee == "eig_sym") {
        return count == 2;
    }
    if (callee == "lu" || callee == "svd") {
        return count == 2 || count == 3;
    }
    return false;
}

bool Interpreter::try_parse_multi_matrix_call_assignment(const std::string& line,
                                                           MultiMatrixCallAssign& assign) {
    const std::string cmd = trim(line);
    const auto eq = cmd.find('=');
    if (eq == std::string::npos) {
        return false;
    }
    const std::string lhs = trim(cmd.substr(0, eq));
    if (lhs.find(',') == std::string::npos) {
        return false;
    }

    assign.targets = split_comma_list(lhs);
    if (assign.targets.empty()) {
        return false;
    }
    for (const auto& target : assign.targets) {
        if (!is_identifier(target)) {
            return false;
        }
    }

    const std::string rhs = trim(cmd.substr(eq + 1));
    const auto open = rhs.find('(');
    if (open == std::string::npos) {
        return false;
    }
    assign.callee = lower(trim_copy(rhs.substr(0, open)));
    if (!is_multi_matrix_call_callee(assign.callee)) {
        return false;
    }

    const auto call_args = split_call_args(rhs);
    if (!call_args || call_args->size() != 1) {
        return false;
    }
    assign.arg = trim_copy(call_args->front());
    if (assign.arg.empty()) {
        return false;
    }

    return is_valid_multi_matrix_target_count(assign.callee, assign.targets.size());
}

bool Interpreter::try_parse_scalar_matrix_call_assignment(const std::string& line,
                                                          ScalarMatrixCallAssign& assign) {
    const std::string cmd = trim(line);
    const auto eq = cmd.find('=');
    if (eq == std::string::npos) {
        return false;
    }
    assign.target = trim(cmd.substr(0, eq));
    if (assign.target.empty() || !is_identifier(assign.target)) {
        return false;
    }
    const std::string rhs = trim(cmd.substr(eq + 1));
    const auto open = rhs.find('(');
    if (open == std::string::npos) {
        return false;
    }
    assign.callee = lower(trim_copy(rhs.substr(0, open)));
    if (!is_scalar_matrix_call_callee(assign.callee)) {
        return false;
    }
    const auto call_args = split_call_args(rhs);
    if (!call_args || call_args->size() != 1) {
        return false;
    }
    assign.arg = trim_copy(call_args->front());
    return !assign.arg.empty();
}

bool Interpreter::try_parse_matrix_call_assignment(const std::string& line, MatrixCallAssign& assign) {
    const std::string cmd = trim(line);
    const auto eq = cmd.find('=');
    if (eq == std::string::npos) {
        return false;
    }
    assign.target = trim(cmd.substr(0, eq));
    if (assign.target.empty() || !is_identifier(assign.target)) {
        return false;
    }
    const std::string rhs = trim(cmd.substr(eq + 1));
    const auto open = rhs.find('(');
    if (open == std::string::npos) {
        return false;
    }
    assign.callee = lower(trim_copy(rhs.substr(0, open)));
    if (!is_matrix_call_callee(assign.callee)) {
        return false;
    }
    const auto call_args = split_call_args(rhs);
    if (!call_args || !is_valid_matrix_call_arity(assign.callee, call_args->size())) {
        return false;
    }
    assign.args.clear();
    for (const auto& arg : *call_args) {
        const std::string trimmed = trim_copy(arg);
        if (trimmed.empty()) {
            return false;
        }
        assign.args.push_back(trimmed);
    }
    return true;
}

std::vector<std::string> Interpreter::list_scalar_expr_variables(const std::string& expr) {
    std::vector<std::string> vars;
    collect_scalar_expr_variables(expr, vars);
    return vars;
}

Result<double> Interpreter::eval_scalar_call(const std::string& name,
                                             const std::vector<double>& args) {
    const std::string fn = lower(name);
    if (args.size() == 1) {
        const double arg = args[0];
        if (fn == "sin") {
            return std::sin(arg);
        }
        if (fn == "cos") {
            return std::cos(arg);
        }
        if (fn == "tan") {
            return std::tan(arg);
        }
        if (fn == "asin") {
            return std::asin(arg);
        }
        if (fn == "acos") {
            return std::acos(arg);
        }
        if (fn == "atan") {
            return std::atan(arg);
        }
        if (fn == "sinh") {
            return std::sinh(arg);
        }
        if (fn == "cosh") {
            return std::cosh(arg);
        }
        if (fn == "tanh") {
            return std::tanh(arg);
        }
        if (fn == "sqrt") {
            return std::sqrt(arg);
        }
        if (fn == "abs") {
            return std::fabs(arg);
        }
        if (fn == "exp") {
            return std::exp(arg);
        }
        if (fn == "log") {
            return std::log(arg);
        }
        if (fn == "log10") {
            return std::log10(arg);
        }
        if (fn == "floor") {
            return std::floor(arg);
        }
        if (fn == "ceil") {
            return std::ceil(arg);
        }
        if (fn == "erf") {
            return ms::erf(arg);
        }
        if (fn == "combo_factorial") {
            if (arg < 0.0 || std::floor(arg) != arg) {
                return std::unexpected(
                    DomainError{"combo_factorial", "expected non-negative integer n"});
            }
            return static_cast<double>(combo::factorial(static_cast<uint32_t>(arg)));
        }
        if (fn == "combo_catalan") {
            if (arg < 0.0 || std::floor(arg) != arg) {
                return std::unexpected(
                    DomainError{"combo_catalan", "expected non-negative integer n"});
            }
            return static_cast<double>(combo::catalan_num(static_cast<uint32_t>(arg)));
        }
        if (fn == "combo_bell") {
            if (arg < 0.0 || std::floor(arg) != arg) {
                return std::unexpected(
                    DomainError{"combo_bell", "expected non-negative integer n"});
            }
            return static_cast<double>(combo::bell_num(static_cast<uint32_t>(arg)));
        }
        if (fn == "combo_motzkin") {
            if (arg < 0.0 || std::floor(arg) != arg) {
                return std::unexpected(
                    DomainError{"combo_motzkin", "expected non-negative integer n"});
            }
            return static_cast<double>(combo::motzkin_num(static_cast<uint32_t>(arg)));
        }
        if (fn == "combo_subfactorial") {
            if (arg < 0.0 || std::floor(arg) != arg) {
                return std::unexpected(
                    DomainError{"combo_subfactorial", "expected non-negative integer n"});
            }
            return static_cast<double>(combo::subfactorial(static_cast<uint32_t>(arg)));
        }
        if (fn == "combo_double_factorial") {
            if (arg < 0.0 || std::floor(arg) != arg) {
                return std::unexpected(
                    DomainError{"combo_double_factorial", "expected non-negative integer n"});
            }
            return static_cast<double>(combo::double_factorial(static_cast<uint32_t>(arg)));
        }
        if (fn == "info_channel_capacity_bsc") {
            return info::channel_capacity_bsc(arg);
        }
        if (fn == "info_channel_capacity_bec") {
            return info::channel_capacity_bec(arg);
        }
        if (fn == "info_differential_entropy_gaussian") {
            return info::differential_entropy_gaussian(arg);
        }
        if (fn == "numthy_partition") {
            if (arg < 0.0 || std::floor(arg) != arg) {
                return std::unexpected(
                    DomainError{"numthy_partition", "expected non-negative integer n"});
            }
            return static_cast<double>(numthy::partition(static_cast<uint32_t>(arg)));
        }
        if (fn == "numthy_num_divisors") {
            if (arg < 0.0 || std::floor(arg) != arg) {
                return std::unexpected(
                    DomainError{"numthy_num_divisors", "expected non-negative integer n"});
            }
            return static_cast<double>(
                numthy::num_divisors(static_cast<uint64_t>(arg)));
        }
        if (fn == "numthy_factor_count") {
            if (arg < 0.0 || std::floor(arg) != arg) {
                return std::unexpected(
                    DomainError{"numthy_factor_count", "expected non-negative integer n"});
            }
            return static_cast<double>(
                numthy::factor(static_cast<uint64_t>(arg)).size());
        }
        if (fn == "numthy_sum_divisors") {
            if (arg < 0.0 || std::floor(arg) != arg) {
                return std::unexpected(
                    DomainError{"numthy_sum_divisors", "expected non-negative integer n"});
            }
            return static_cast<double>(
                numthy::sum_divisors(static_cast<uint64_t>(arg)));
        }
        if (fn == "numthy_isprime") {
            if (arg < 0.0 || std::floor(arg) != arg) {
                return std::unexpected(
                    DomainError{"numthy_isprime", "expected non-negative integer n"});
            }
            return numthy::isprime(static_cast<uint64_t>(arg)) ? 1.0 : 0.0;
        }
        if (fn == "numthy_euler_phi") {
            if (arg < 0.0 || std::floor(arg) != arg) {
                return std::unexpected(
                    DomainError{"numthy_euler_phi", "expected non-negative integer n"});
            }
            return static_cast<double>(numthy::euler_phi(static_cast<uint64_t>(arg)));
        }
        if (fn == "numthy_mobius") {
            if (arg < 0.0 || std::floor(arg) != arg) {
                return std::unexpected(
                    DomainError{"numthy_mobius", "expected non-negative integer n"});
            }
            return static_cast<double>(numthy::mobius(static_cast<uint64_t>(arg)));
        }
        if (fn == "numthy_nextprime") {
            if (arg < 0.0 || std::floor(arg) != arg) {
                return std::unexpected(
                    DomainError{"numthy_nextprime", "expected non-negative integer n"});
            }
            return static_cast<double>(numthy::nextprime(static_cast<uint64_t>(arg)));
        }
        if (fn == "numthy_prevprime") {
            if (arg < 0.0 || std::floor(arg) != arg) {
                return std::unexpected(
                    DomainError{"numthy_prevprime", "expected non-negative integer n"});
            }
            return static_cast<double>(numthy::prevprime(static_cast<uint64_t>(arg)));
        }
        if (fn == "numthy_liouville") {
            if (arg < 0.0 || std::floor(arg) != arg) {
                return std::unexpected(
                    DomainError{"numthy_liouville", "expected non-negative integer n"});
            }
            return static_cast<double>(numthy::liouville(static_cast<uint64_t>(arg)));
        }
        if (fn == "numthy_prime_pi") {
            if (arg < 0.0 || std::floor(arg) != arg) {
                return std::unexpected(
                    DomainError{"numthy_prime_pi", "expected non-negative integer n"});
            }
            return static_cast<double>(numthy::prime_pi(static_cast<uint64_t>(arg)));
        }
        if (fn == "numthy_prime_nth") {
            if (arg < 1.0 || std::floor(arg) != arg) {
                return std::unexpected(
                    DomainError{"numthy_prime_nth", "expected integer n >= 1"});
            }
            return static_cast<double>(numthy::prime_nth(static_cast<uint64_t>(arg)));
        }
        if (fn == "numthy_primitive_root") {
            if (arg < 2.0 || std::floor(arg) != arg) {
                return std::unexpected(
                    DomainError{"numthy_primitive_root", "expected prime p >= 2"});
            }
            const auto p = static_cast<uint64_t>(arg);
            if (!numthy::isprime(p)) {
                return std::unexpected(
                    DomainError{"numthy_primitive_root", "expected prime p"});
            }
            return static_cast<double>(numthy::primitive_root(p));
        }
    }
    if (args.size() == 2) {
        if (fn == "pow") {
            return std::pow(args[0], args[1]);
        }
        if (fn == "min") {
            return std::fmin(args[0], args[1]);
        }
        if (fn == "max") {
            return std::fmax(args[0], args[1]);
        }
        if (fn == "atan2") {
            return std::atan2(args[0], args[1]);
        }
        if (fn == "combo_nchoosek") {
            const int n = static_cast<int>(args[0]);
            const int k = static_cast<int>(args[1]);
            if (n < 0 || k < 0 || k > n) {
                return std::unexpected(DomainError{"combo_nchoosek", "expected 0 <= k <= n"});
            }
            return static_cast<double>(combo::binomial(static_cast<uint32_t>(n),
                                                       static_cast<uint32_t>(k)));
        }
        if (fn == "diffgeo_gaussian_curvature_sphere") {
            return eval_diffgeo_gaussian_curvature_sphere(args[0], args[1]);
        }
        if (fn == "diffgeo_mean_curvature_sphere") {
            return eval_diffgeo_mean_curvature_sphere(args[0], args[1]);
        }
        if (fn == "diffgeo_ricci_scalar_sphere") {
            return eval_diffgeo_ricci_scalar_sphere(args[0], args[1]);
        }
        if (fn == "diffgeo_einstein_scalar_sphere") {
            return eval_diffgeo_einstein_scalar_sphere(args[0], args[1]);
        }
        if (fn == "cplx_residue_inv") {
            return eval_cplx_residue_inv(args[0], args[1]);
        }
        if (fn == "combo_stirling2") {
            const int n = static_cast<int>(args[0]);
            const int k = static_cast<int>(args[1]);
            if (n < 0 || k < 0 || k > n) {
                return std::unexpected(DomainError{"combo_stirling2", "expected 0 <= k <= n"});
            }
            return static_cast<double>(combo::stirling2(static_cast<uint32_t>(n),
                                                        static_cast<uint32_t>(k)));
        }
        if (fn == "combo_stirling1") {
            const int n = static_cast<int>(args[0]);
            const int k = static_cast<int>(args[1]);
            if (n < 0 || k < 0 || k > n) {
                return std::unexpected(DomainError{"combo_stirling1", "expected 0 <= k <= n"});
            }
            return static_cast<double>(combo::stirling1(static_cast<uint32_t>(n),
                                                        static_cast<uint32_t>(k)));
        }
        if (fn == "combo_permutations") {
            const int n = static_cast<int>(args[0]);
            const int k = static_cast<int>(args[1]);
            if (n < 0 || k < 0 || k > n) {
                return std::unexpected(DomainError{"combo_permutations", "expected 0 <= k <= n"});
            }
            return static_cast<double>(combo::permutations(static_cast<uint32_t>(n),
                                                          static_cast<uint32_t>(k)));
        }
        if (fn == "combo_combinations_with_rep") {
            const int n = static_cast<int>(args[0]);
            const int k = static_cast<int>(args[1]);
            if (n < 0 || k < 0 || std::floor(args[0]) != args[0] ||
                std::floor(args[1]) != args[1]) {
                return std::unexpected(
                    DomainError{"combo_combinations_with_rep",
                                "expected non-negative integer n and k"});
            }
            return static_cast<double>(combo::combinations_with_rep(static_cast<uint32_t>(n),
                                                                    static_cast<uint32_t>(k)));
        }
        if (fn == "numthy_legendre_symbol") {
            if (std::floor(args[0]) != args[0] || std::floor(args[1]) != args[1]) {
                return std::unexpected(
                    DomainError{"numthy_legendre_symbol", "expected integer arguments"});
            }
            if (args[1] < 0.0) {
                return std::unexpected(
                    DomainError{"numthy_legendre_symbol", "expected odd prime p"});
            }
            return static_cast<double>(numthy::legendre_symbol(static_cast<int64_t>(args[0]),
                                                                static_cast<uint64_t>(args[1])));
        }
        if (fn == "numthy_jacobi_symbol") {
            if (std::floor(args[0]) != args[0] || std::floor(args[1]) != args[1]) {
                return std::unexpected(
                    DomainError{"numthy_jacobi_symbol", "expected integer arguments"});
            }
            if (args[1] < 0.0) {
                return std::unexpected(
                    DomainError{"numthy_jacobi_symbol", "expected positive odd n"});
            }
            return static_cast<double>(numthy::jacobi_symbol(static_cast<int64_t>(args[0]),
                                                               static_cast<uint64_t>(args[1])));
        }
        if (fn == "numthy_kronecker_symbol") {
            if (std::floor(args[0]) != args[0] || std::floor(args[1]) != args[1]) {
                return std::unexpected(
                    DomainError{"numthy_kronecker_symbol", "expected integer arguments"});
            }
            return static_cast<double>(numthy::kronecker_symbol(static_cast<int64_t>(args[0]),
                                                                  static_cast<int64_t>(args[1])));
        }
        if (fn == "numthy_tonelli_shanks") {
            return eval_numthy_tonelli_shanks(args[0], args[1]);
        }
        if (fn == "numthy_mod_inv") {
            return eval_numthy_mod_inv(args[0], args[1]);
        }
        if (fn == "numthy_extended_gcd") {
            return eval_numthy_extended_gcd(args[0], args[1]);
        }
        if (fn == "numthy_is_primitive_root") {
            if (std::floor(args[0]) != args[0] || std::floor(args[1]) != args[1]) {
                return std::unexpected(
                    DomainError{"numthy_is_primitive_root", "expected integer arguments"});
            }
            if (args[0] < 0.0 || args[1] <= 0.0) {
                return std::unexpected(
                    DomainError{"numthy_is_primitive_root", "expected g >= 0 and p > 0"});
            }
            return numthy::is_primitive_root(static_cast<uint64_t>(args[0]),
                                             static_cast<uint64_t>(args[1]))
                       ? 1.0
                       : 0.0;
        }
        if (fn == "finance_kelly_fraction") {
            return finance::kelly_fraction(args[0], args[1]);
        }
        if (fn == "info_shannon_hartley") {
            return info::shannon_hartley(args[0], args[1]);
        }
        if (fn == "info_differential_entropy_uniform") {
            return info::differential_entropy_uniform(args[0], args[1]);
        }
        if (fn == "info_rate_distortion_gaussian") {
            return info::rate_distortion_gaussian(args[0], args[1]);
        }
        if (fn == "geo_vec2d_length") {
            const geo::Vec2D v{args[0], args[1]};
            return geo::length(v);
        }
        if (fn == "numthy_gcd") {
            const auto a = static_cast<uint64_t>(args[0]);
            const auto b = static_cast<uint64_t>(args[1]);
            return static_cast<double>(numthy::gcd(a, b));
        }
        if (fn == "numthy_lcm") {
            const auto a = static_cast<uint64_t>(args[0]);
            const auto b = static_cast<uint64_t>(args[1]);
            return static_cast<double>(numthy::lcm(a, b));
        }
        if (fn == "cplx_joukowski") {
            const cplx::C z{args[0], args[1]};
            const cplx::C w = cplx::joukowski(z);
            return std::abs(w);
        }
        if (fn == "cplx_joukowski_inv") {
            return eval_cplx_joukowski_inv(args[0], args[1]);
        }
        if (fn == "prob_exp_cdf") {
            return exp_cdf(args[0], args[1]);
        }
        if (fn == "prob_pois_pdf") {
            return pois_pdf(args[0], args[1]);
        }
        if (fn == "prob_pois_cdf") {
            return pois_cdf(args[0], args[1]);
        }
        if (fn == "prob_exp_pdf") {
            return exp_pdf(args[0], args[1]);
        }
        if (fn == "prob_chi2_cdf") {
            return chi2_cdf(args[0], args[1]);
        }
        if (fn == "prob_t_cdf") {
            return t_cdf(args[0], args[1]);
        }
        if (fn == "prob_chi2_pdf") {
            return chi2_pdf(args[0], args[1]);
        }
    }
    if (args.size() == 4 && fn == "geo_dist2d") {
        const geo::Point2D p0{args[0], args[1]};
        const geo::Point2D p1{args[2], args[3]};
        return geo::dist(p0, p1);
    }
    if (args.size() == 4 && fn == "geo_dist_sq2d") {
        const geo::Point2D p0{args[0], args[1]};
        const geo::Point2D p1{args[2], args[3]};
        return geo::dist_sq(p0, p1);
    }
    if (args.size() == 4 && fn == "geo_cross2d") {
        const geo::Vec2D a{args[0], args[1]};
        const geo::Vec2D b{args[2], args[3]};
        return geo::cross2d(a, b);
    }
    if (args.size() == 12 && fn == "geo_volume_tetrahedron") {
        const geo::Point3D a{args[0], args[1], args[2]};
        const geo::Point3D b{args[3], args[4], args[5]};
        const geo::Point3D c{args[6], args[7], args[8]};
        const geo::Point3D d{args[9], args[10], args[11]};
        return geo::volume_tetrahedron(a, b, c, d);
    }
    if (args.size() == 9 && fn == "geo_hermite_x") {
        auto value = eval_geo_hermite_x(args[0], args[1], args[2], args[3], args[4], args[5],
                                        args[6], args[7], args[8]);
        if (!value) {
            return std::unexpected(value.error());
        }
        return *value;
    }
    if (args.size() == 6 && fn == "cplx_mobius_re") {
        auto value = eval_cplx_mobius_re(args[0], args[1], args[2], args[3], args[4], args[5]);
        if (!value) {
            return std::unexpected(value.error());
        }
        return *value;
    }
    if (args.size() == 6 && fn == "geo_dist3d") {
        const geo::Point3D p0{args[0], args[1], args[2]};
        const geo::Point3D p1{args[3], args[4], args[5]};
        return geo::dist(p0, p1);
    }
    if (args.size() == 6 && fn == "geo_dist_point_seg2d") {
        const geo::Point2D p{args[0], args[1]};
        const geo::Segment2D s{{args[2], args[3]}, {args[4], args[5]}};
        return geo::dist_point_segment(p, s);
    }
    if (args.size() == 6 && fn == "geo_triangle_area") {
        const geo::Point2D p0{args[0], args[1]};
        const geo::Point2D p1{args[2], args[3]};
        const geo::Point2D p2{args[4], args[5]};
        return geo::area(p0, p1, p2);
    }
    if (args.size() == 6 && fn == "geo_overlap_circles") {
        const geo::Circle2D a{{args[0], args[1]}, args[2]};
        const geo::Circle2D b{{args[3], args[4]}, args[5]};
        return geo::overlap_circle_circle(a, b) ? 1.0 : 0.0;
    }
    if (args.size() == 4 && fn == "cplx_hyperbolic_distance") {
        const cplx::C z1{args[0], args[1]};
        const cplx::C z2{args[2], args[3]};
        return cplx::hyperbolic_distance(z1, z2);
    }
    if ((args.size() == 3 || args.size() == 4) && fn == "finance_bond_price") {
        const int n = static_cast<int>(args[2]);
        if (n < 0 || args[2] != n) {
            return std::unexpected(
                DomainError{"finance_bond_price", "expected non-negative integer periods n"});
        }
        const double fv = args.size() == 4 ? args[3] : 100.0;
        return finance::bond_price(args[0], args[1], n, fv);
    }
    if (args.size() == 3 && fn == "cplx_poisson_kernel") {
        return cplx::poisson_kernel(args[0], args[1], args[2]);
    }
    if (args.size() == 3 && fn == "finance_bond_duration") {
        const int n = static_cast<int>(args[2]);
        if (n < 0 || args[2] != n) {
            return std::unexpected(
                DomainError{"finance_bond_duration", "expected non-negative integer periods n"});
        }
        return finance::bond_duration(args[0], args[1], n);
    }
    if (args.size() == 3 && fn == "finance_bond_modified_duration") {
        const int n = static_cast<int>(args[2]);
        if (n < 0 || args[2] != n) {
            return std::unexpected(DomainError{
                "finance_bond_modified_duration", "expected non-negative integer periods n"});
        }
        return finance::bond_modified_duration(args[0], args[1], n);
    }
    if (args.size() == 3 && fn == "finance_bond_convexity") {
        const int n = static_cast<int>(args[2]);
        if (n < 0 || args[2] != n) {
            return std::unexpected(
                DomainError{"finance_bond_convexity", "expected non-negative integer periods n"});
        }
        return finance::bond_convexity(args[0], args[1], n);
    }
    if (args.size() == 3 && fn == "finance_bond_ytm") {
        return eval_finance_bond_ytm(args[0], args[1], args[2]);
    }
    if (args.size() == 3 && fn == "finance_compound") {
        const int n = static_cast<int>(args[2]);
        if (n < 0 || args[2] != n) {
            return std::unexpected(
                DomainError{"finance_compound", "expected non-negative integer periods n_periods"});
        }
        return finance::compound(args[0], args[1], n);
    }
    if (args.size() == 4 && fn == "finance_compound") {
        const int n = static_cast<int>(args[2]);
        const int cpp = static_cast<int>(args[3]);
        if (n < 0 || args[2] != n) {
            return std::unexpected(
                DomainError{"finance_compound", "expected non-negative integer periods n_periods"});
        }
        if (cpp < 1 || args[3] != cpp) {
            return std::unexpected(DomainError{
                "finance_compound", "expected positive integer compounds_per_period"});
        }
        return finance::compound(args[0], args[1], n, cpp);
    }
    if (args.size() == 3 && fn == "finance_continuous_compound") {
        return finance::continuous_compound(args[0], args[1], args[2]);
    }
    if (args.size() == 3 && fn == "numthy_mod_pow") {
        const auto base = static_cast<uint64_t>(args[0]);
        const auto exp = static_cast<uint64_t>(args[1]);
        const auto mod = static_cast<uint64_t>(args[2]);
        return static_cast<double>(numthy::mod_pow(base, exp, mod));
    }
    if (args.size() == 3 && fn == "numthy_discrete_log") {
        return eval_numthy_discrete_log(args[0], args[1], args[2]);
    }
    if (args.size() == 3 && fn == "prob_norm_cdf") {
        return norm_cdf(args[0], args[1], args[2]);
    }
    if (args.size() == 3 && fn == "prob_norm_pdf") {
        return norm_pdf(args[0], args[1], args[2]);
    }
    if (args.size() == 3 && fn == "prob_norm_ppf") {
        return norm_ppf(args[0], args[1], args[2]);
    }
    if (args.size() == 3 && fn == "prob_binom_pdf") {
        const int k = static_cast<int>(args[0]);
        const int n = static_cast<int>(args[1]);
        if (args[0] != k || args[1] != n) {
            return std::unexpected(
                DomainError{"prob_binom_pdf", "expected integer k and n"});
        }
        return binom_pdf(k, n, args[2]);
    }
    if (args.size() == 3 && fn == "prob_binom_cdf") {
        const int k = static_cast<int>(args[0]);
        const int n = static_cast<int>(args[1]);
        if (args[0] != k || args[1] != n) {
            return std::unexpected(
                DomainError{"prob_binom_cdf", "expected integer k and n"});
        }
        return binom_cdf(k, n, args[2]);
    }
    if (args.size() == 3 && fn == "prob_uniform_cdf") {
        return uniform_cdf(args[0], args[1], args[2]);
    }
    if (args.size() == 3 && fn == "prob_gamma_pdf") {
        return gamma_pdf(args[0], args[1], args[2]);
    }
    if ((args.size() == 3 || args.size() == 4) && fn == "finance_pv") {
        const int n = static_cast<int>(args[1]);
        if (n < 0 || args[1] != n) {
            return std::unexpected(
                DomainError{"finance_pv", "expected non-negative integer n"});
        }
        const double fv = args.size() == 4 ? args[3] : 0.0;
        return finance::pv(args[0], n, args[2], fv);
    }
    if ((args.size() == 3 || args.size() == 4) && fn == "finance_fv_annuity") {
        const int n = static_cast<int>(args[1]);
        if (n < 0 || args[1] != n) {
            return std::unexpected(
                DomainError{"finance_fv_annuity", "expected non-negative integer n"});
        }
        const double pv0 = args.size() == 4 ? args[3] : 0.0;
        return finance::fv_annuity(args[0], n, args[2], pv0);
    }
    if ((args.size() == 3 || args.size() == 4) && fn == "finance_pmt_annuity") {
        const int n = static_cast<int>(args[1]);
        if (n < 0 || args[1] != n) {
            return std::unexpected(
                DomainError{"finance_pmt_annuity", "expected non-negative integer n"});
        }
        const double fv = args.size() == 4 ? args[3] : 0.0;
        return finance::pmt_annuity(args[0], n, args[2], fv);
    }
    if (args.size() == 5 && fn == "diffgeo_christoffel_sphere") {
        return eval_diffgeo_christoffel_sphere(args[0], args[1], args[2], args[3], args[4]);
    }
    if (args.size() == 5 && fn == "geo_dist_point_line2d") {
        const geo::Point2D p{args[0], args[1]};
        const geo::Line2D l{args[2], args[3], args[4]};
        return geo::dist_point_line(p, l);
    }
    if (args.size() == 5 && fn == "finance_bs_call") {
        return finance::bs_call(args[0], args[1], args[2], args[3], args[4]);
    }
    if (args.size() == 5 && fn == "finance_bs_put") {
        return finance::bs_put(args[0], args[1], args[2], args[3], args[4]);
    }
    if (args.size() == 5 && fn == "finance_bs_gamma") {
        return finance::bs_gamma(args[0], args[1], args[2], args[3], args[4]);
    }
    if (args.size() == 5 && fn == "finance_bs_vega") {
        return finance::bs_vega(args[0], args[1], args[2], args[3], args[4]);
    }
    if (args.size() == 6 && fn == "finance_bs_delta") {
        const int call = static_cast<int>(args[5]);
        if (args[5] != call) {
            return std::unexpected(
                DomainError{"finance_bs_delta", "expected integer call (0=put, 1=call)"});
        }
        return finance::bs_delta(args[0], args[1], args[2], args[3], args[4], call != 0);
    }
    if (args.size() == 6 && fn == "finance_bs_implied_vol") {
        return eval_finance_bs_implied_vol(args[0], args[1], args[2], args[3], args[4], args[5]);
    }
    if (args.size() == 6 && fn == "finance_bs_theta") {
        const int call = static_cast<int>(args[5]);
        if (args[5] != call) {
            return std::unexpected(
                DomainError{"finance_bs_theta", "expected integer call (0=put, 1=call)"});
        }
        return finance::bs_theta(args[0], args[1], args[2], args[3], args[4], call != 0);
    }
    if (args.size() == 6 && fn == "finance_bs_rho") {
        const int call = static_cast<int>(args[5]);
        if (args[5] != call) {
            return std::unexpected(
                DomainError{"finance_bs_rho", "expected integer call (0=put, 1=call)"});
        }
        return finance::bs_rho(args[0], args[1], args[2], args[3], args[4], call != 0);
    }
    if (args.size() == 6 && fn == "finance_binomial_call") {
        const int steps = static_cast<int>(args[5]);
        if (steps < 0 || args[5] != steps) {
            return std::unexpected(
                DomainError{"finance_binomial_call", "expected non-negative integer steps"});
        }
        return finance::binomial_call(args[0], args[1], args[2], args[3], args[4], steps);
    }
    if (args.size() == 6 && fn == "finance_binomial_put") {
        const int steps = static_cast<int>(args[5]);
        if (steps < 0 || args[5] != steps) {
            return std::unexpected(
                DomainError{"finance_binomial_put", "expected non-negative integer steps"});
        }
        return finance::binomial_put(args[0], args[1], args[2], args[3], args[4], steps);
    }
    if (args.size() == 8 && fn == "cplx_cross_ratio") {
        const cplx::C z1{args[0], args[1]};
        const cplx::C z2{args[2], args[3]};
        const cplx::C z3{args[4], args[5]};
        const cplx::C z4{args[6], args[7]};
        return cplx::cross_ratio(z1, z2, z3, z4);
    }
    return std::unexpected(DomainError{"eval", "unknown scalar function: " + name});
}

Result<double> Interpreter::eval_scalar_op(char op, double left, double right) {
    switch (op) {
    case '+':
        return left + right;
    case '-':
        return left - right;
    case '*':
        return left * right;
    case '/':
        if (right == 0.0) {
            return std::unexpected(DomainError{"divide", "division by zero"});
        }
        return left / right;
    default:
        return std::unexpected(DomainError{"eval", "unsupported operator"});
    }
}

Result<std::string> Interpreter::assign_scalar_binary(const ScalarBinaryAssign& assign) {
    auto resolve = [this](const ScalarOperand& operand) -> Result<double> {
        return resolve_scalar_operand(state_, operand);
    };

    auto left = resolve(assign.left);
    if (!left) {
        return std::unexpected(left.error());
    }
    auto right = resolve(assign.right);
    if (!right) {
        return std::unexpected(right.error());
    }
    auto value = eval_scalar_op(assign.op, *left, *right);
    if (!value) {
        return std::unexpected(value.error());
    }
    return assign_scalar(assign.target, *value);
}

Result<std::string> Interpreter::assign_scalar_expr(const std::string& name, const std::string& expr) {
    auto value = eval_scalar_expr(state_, expr);
    if (!value) {
        return std::unexpected(value.error());
    }
    return assign_scalar(name, *value);
}

Result<std::string> Interpreter::assign_matrix_call(const MatrixCallAssign& assign) {
    auto resolve_operand = [this](const std::string& text) { return eval_matrix_operand(text); };

    Result<Matrix<double>> result = std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "matmul" && assign.args.size() == 2) {
        auto left = resolve_operand(assign.args[0]);
        if (!left) {
            return std::unexpected(left.error());
        }
        auto right = resolve_operand(assign.args[1]);
        if (!right) {
            return std::unexpected(right.error());
        }
        result = matmul(*left, *right);
    } else if (assign.callee == "tensorops_matmul" && assign.args.size() == 2) {
        auto left = resolve_operand(assign.args[0]);
        if (!left) {
            return std::unexpected(left.error());
        }
        auto right = resolve_operand(assign.args[1]);
        if (!right) {
            return std::unexpected(right.error());
        }
        result = eval_tensorops_matmul(*left, *right);
    } else if (assign.callee == "tensorops_einsum" && assign.args.size() == 2) {
        auto left = resolve_operand(assign.args[0]);
        if (!left) {
            return std::unexpected(left.error());
        }
        auto right = resolve_operand(assign.args[1]);
        if (!right) {
            return std::unexpected(right.error());
        }
        result = eval_tensorops_einsum(*left, *right);
    } else if (assign.callee == "solve" && assign.args.size() == 2) {
        auto left = resolve_operand(assign.args[0]);
        if (!left) {
            return std::unexpected(left.error());
        }
        auto right = resolve_operand(assign.args[1]);
        if (!right) {
            return std::unexpected(right.error());
        }
        result = solve(*left, *right);
    } else if (assign.callee == "transpose" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        const auto transposed = transpose(*matrix);
        Matrix<double> stored(transposed.rows(), transposed.cols());
        for (size_t i = 0; i < transposed.rows(); ++i) {
            for (size_t j = 0; j < transposed.cols(); ++j) {
                stored(i, j) = transposed(i, j);
            }
        }
        result = stored;
    } else if (assign.callee == "chol" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        result = chol(*matrix);
    } else if (assign.callee == "expm" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        result = expm(*matrix);
    } else if (assign.callee == "inv" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        // inv via LU solve: A * inv(A) = I
        const size_t n = matrix->rows();
        auto I = eye<double>(n);
        result = solve(*matrix, I);
    } else if ((assign.callee == "zeros" || assign.callee == "eye" || assign.callee == "ones") &&
               (assign.args.size() == 1 || assign.args.size() == 2)) {
        double m_d = 0.0, n_d = 0.0;
        if (!parse_number(assign.args[0], m_d)) {
            // try resolving as scalar variable
            auto it = state_.scalars.find(assign.args[0]);
            if (it != state_.scalars.end()) {
                m_d = it->second;
            } else {
                return std::unexpected(DomainError{assign.callee, "expected numeric size argument"});
            }
        }
        if (assign.args.size() == 2) {
            if (!parse_number(assign.args[1], n_d)) {
                auto it = state_.scalars.find(assign.args[1]);
                if (it != state_.scalars.end()) {
                    n_d = it->second;
                } else {
                    return std::unexpected(DomainError{assign.callee, "expected numeric size argument"});
                }
            }
        } else {
            n_d = m_d;
        }
        const size_t rows = static_cast<size_t>(m_d);
        const size_t cols = static_cast<size_t>(n_d);
        if (assign.callee == "zeros") {
            result = zeros<double>(rows, cols);
        } else if (assign.callee == "eye") {
            auto I = eye<double>(rows);
            result = I;
        } else {
            result = ones<double>(rows, cols);
        }
    } else if ((assign.callee == "rand" || assign.callee == "randn") && assign.args.size() == 2) {
        double m_d = 0.0, n_d = 0.0;
        if (!parse_number(assign.args[0], m_d)) {
            auto it = state_.scalars.find(assign.args[0]);
            if (it != state_.scalars.end()) m_d = it->second;
            else return std::unexpected(DomainError{assign.callee, "expected numeric size"});
        }
        if (!parse_number(assign.args[1], n_d)) {
            auto it = state_.scalars.find(assign.args[1]);
            if (it != state_.scalars.end()) n_d = it->second;
            else return std::unexpected(DomainError{assign.callee, "expected numeric size"});
        }
        const size_t rows = static_cast<size_t>(m_d);
        const size_t cols = static_cast<size_t>(n_d);
        if (assign.callee == "rand") {
            auto R = rand<double>(rows, cols, 0u);
            Matrix<double> stored(rows, cols);
            for (size_t i = 0; i < rows; ++i)
                for (size_t j = 0; j < cols; ++j)
                    stored(i, j) = R(i, j);
            result = stored;
        } else {
            auto R = randn<double>(rows, cols, 0u);
            Matrix<double> stored(rows, cols);
            for (size_t i = 0; i < rows; ++i)
                for (size_t j = 0; j < cols; ++j)
                    stored(i, j) = R(i, j);
            result = stored;
        }
    } else if (assign.callee == "pinv" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        result = pinv(*matrix);
    } else if (assign.callee == "null" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        result = null(*matrix);
    } else if (assign.callee == "orth" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        result = orth(*matrix);
    } else if (assign.callee == "kron" && assign.args.size() == 2) {
        auto left = resolve_operand(assign.args[0]);
        if (!left) {
            return std::unexpected(left.error());
        }
        auto right = resolve_operand(assign.args[1]);
        if (!right) {
            return std::unexpected(right.error());
        }
        result = kron(*left, *right);
    } else if (assign.callee == "repmat" && assign.args.size() == 3) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        double p_d = 0.0, q_d = 0.0;
        if (!parse_number(assign.args[1], p_d)) {
            return std::unexpected(DomainError{"repmat", "expected numeric row repeat count"});
        }
        if (!parse_number(assign.args[2], q_d)) {
            return std::unexpected(DomainError{"repmat", "expected numeric col repeat count"});
        }
        result = repmat(*matrix, static_cast<size_t>(p_d), static_cast<size_t>(q_d));
    } else if (assign.callee == "linspace" && assign.args.size() == 3) {
        double a = 0.0, b = 0.0, n_d = 0.0;
        if (!parse_number(assign.args[0], a) || !parse_number(assign.args[1], b) ||
            !parse_number(assign.args[2], n_d)) {
            return std::unexpected(DomainError{"linspace", "expected linspace(a, b, n)"});
        }
        const auto vec = linspace(a, b, static_cast<size_t>(n_d));
        Matrix<double> col(vec.size(), 1);
        for (size_t i = 0; i < vec.size(); ++i) {
            col(i, 0) = vec[i];
        }
        result = col;
    } else if (assign.callee == "rgb2gray" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto rgb = matrix_to_rgb_image(*matrix);
        if (!rgb) {
            return std::unexpected(rgb.error());
        }
        result = gray_image_to_column(image::rgb2gray(*rgb));
    } else if (assign.callee == "sobel" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto gray = matrix_to_gray_image(*matrix);
        if (!gray) {
            return std::unexpected(gray.error());
        }
        result = gray_image_to_matrix(image::sobel(*gray));
    } else if (assign.callee == "rle_encode_vec" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        result = bytes_to_matrix_col(compress::rle_encode(matrix_to_bytes(*matrix)));
    } else if (assign.callee == "rle_decode_vec" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        if (matrix->cols() != 1) {
            return std::unexpected(
                DomainError{"rle_decode_vec", "expected Nx1 encoded byte vector"});
        }
        compress::Bytes encoded;
        encoded.reserve(matrix->rows());
        for (size_t i = 0; i < matrix->rows(); ++i) {
            const double v = (*matrix)(i, 0);
            if (v < 0.0 || v > 255.0 || std::floor(v) != v) {
                return std::unexpected(
                    DomainError{"rle_decode_vec", "encoded values must be uint8 in [0,255]"});
            }
            encoded.push_back(static_cast<uint8_t>(v));
        }
        result = bytes_to_matrix_col(compress::rle_decode(encoded));
    } else if (assign.callee == "mtf_encode_vec" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        result = bytes_to_matrix_col(compress::mtf_encode(matrix_to_bytes(*matrix)));
    } else if (assign.callee == "bwt_encode_vec" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        result = bytes_to_matrix_col(compress::bwt(matrix_to_bytes(*matrix)).data);
    } else if (assign.callee == "mtf_decode_vec" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        if (matrix->cols() != 1) {
            return std::unexpected(
                DomainError{"mtf_decode_vec", "expected Nx1 encoded byte vector"});
        }
        compress::Bytes encoded;
        encoded.reserve(matrix->rows());
        for (size_t i = 0; i < matrix->rows(); ++i) {
            const double v = (*matrix)(i, 0);
            if (v < 0.0 || v > 255.0 || std::floor(v) != v) {
                return std::unexpected(
                    DomainError{"mtf_decode_vec", "encoded values must be uint8 in [0,255]"});
            }
            encoded.push_back(static_cast<uint8_t>(v));
        }
        result = bytes_to_matrix_col(compress::mtf_decode(encoded));
    } else if (assign.callee == "delta_encode_vec" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        result = bytes_to_matrix_col(compress::delta_encode(matrix_to_bytes(*matrix)));
    } else if (assign.callee == "delta_decode_vec" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        if (matrix->cols() != 1) {
            return std::unexpected(
                DomainError{"delta_decode_vec", "expected Nx1 encoded byte vector"});
        }
        compress::Bytes encoded;
        encoded.reserve(matrix->rows());
        for (size_t i = 0; i < matrix->rows(); ++i) {
            const double v = (*matrix)(i, 0);
            if (v < 0.0 || v > 255.0 || std::floor(v) != v) {
                return std::unexpected(
                    DomainError{"delta_decode_vec", "encoded values must be uint8 in [0,255]"});
            }
            encoded.push_back(static_cast<uint8_t>(v));
        }
        result = bytes_to_matrix_col(compress::delta_decode(encoded));
    } else if (assign.callee == "lzw_encode_vec" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto encoded = eval_lzw_encode_vec(*matrix);
        if (!encoded) {
            return std::unexpected(encoded.error());
        }
        result = *encoded;
    } else if (assign.callee == "lzw_decode_vec" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto decoded = eval_lzw_decode_vec(*matrix);
        if (!decoded) {
            return std::unexpected(decoded.error());
        }
        result = *decoded;
    } else if (assign.callee == "huffman_encode_vec" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto encoded = eval_huffman_encode_vec(*matrix);
        if (!encoded) {
            return std::unexpected(encoded.error());
        }
        result = *encoded;
    } else if (assign.callee == "lz77_encode_vec" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto encoded = eval_lz77_encode_vec(*matrix);
        if (!encoded) {
            return std::unexpected(encoded.error());
        }
        result = *encoded;
    } else if (assign.callee == "lz77_encode_vec" && assign.args.size() == 3) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        double window_d = 0.0;
        double lookahead_d = 0.0;
        if (!parse_number(assign.args[1], window_d)) {
            auto w_expr = eval_scalar_expr(state_, assign.args[1]);
            if (!w_expr) {
                return std::unexpected(w_expr.error());
            }
            window_d = *w_expr;
        }
        if (!parse_number(assign.args[2], lookahead_d)) {
            auto l_expr = eval_scalar_expr(state_, assign.args[2]);
            if (!l_expr) {
                return std::unexpected(l_expr.error());
            }
            lookahead_d = *l_expr;
        }
        const int window = static_cast<int>(window_d);
        const int lookahead = static_cast<int>(lookahead_d);
        if (window < 1 || lookahead < 1 || window_d != window || lookahead_d != lookahead) {
            return std::unexpected(DomainError{
                "lz77_encode_vec", "expected positive integer window and lookahead"});
        }
        auto encoded = eval_lz77_encode_vec(*matrix, window, lookahead);
        if (!encoded) {
            return std::unexpected(encoded.error());
        }
        result = *encoded;
    } else if (assign.callee == "topo_betti_curve" && assign.args.size() == 3) {
        auto dist_m = resolve_operand(assign.args[0]);
        if (!dist_m) {
            return std::unexpected(dist_m.error());
        }
        auto thresholds_m = resolve_operand(assign.args[1]);
        if (!thresholds_m) {
            return std::unexpected(thresholds_m.error());
        }
        double max_dim_d = 0.0;
        if (!parse_number(assign.args[2], max_dim_d)) {
            auto md_expr = eval_scalar_expr(state_, assign.args[2]);
            if (!md_expr) {
                return std::unexpected(md_expr.error());
            }
            max_dim_d = *md_expr;
        }
        const int max_dim = static_cast<int>(max_dim_d);
        if (max_dim < 0 || max_dim_d != max_dim) {
            return std::unexpected(
                DomainError{"topo_betti_curve", "expected non-negative integer max_dim"});
        }
        auto curve = eval_topo_betti_curve(*dist_m, *thresholds_m, max_dim);
        if (!curve) {
            return std::unexpected(curve.error());
        }
        result = *curve;
    } else if (assign.callee == "control_bode" && assign.args.size() == 3) {
        auto num_m = resolve_operand(assign.args[0]);
        if (!num_m) {
            return std::unexpected(num_m.error());
        }
        auto den_m = resolve_operand(assign.args[1]);
        if (!den_m) {
            return std::unexpected(den_m.error());
        }
        double w = 0.0;
        if (!parse_number(assign.args[2], w)) {
            auto w_expr = eval_scalar_expr(state_, assign.args[2]);
            if (!w_expr) {
                return std::unexpected(w_expr.error());
            }
            w = *w_expr;
        }
        auto bode = eval_control_bode(*num_m, *den_m, w);
        if (!bode) {
            return std::unexpected(bode.error());
        }
        result = *bode;
    } else if (assign.callee == "compress_bits_to_bytes" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto encoded = eval_compress_bits_to_bytes(*matrix);
        if (!encoded) {
            return std::unexpected(encoded.error());
        }
        result = *encoded;
    } else if (assign.callee == "compress_bytes_to_bits" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto decoded = eval_compress_bytes_to_bits(*matrix);
        if (!decoded) {
            return std::unexpected(decoded.error());
        }
        result = *decoded;
    } else if (assign.callee == "diffgeo_geodesic_euclidean" && assign.args.size() == 5) {
        double x0 = 0.0;
        double y0 = 0.0;
        double vx = 0.0;
        double vy = 0.0;
        double s_end = 0.0;
        if (!parse_number(assign.args[0], x0) || !parse_number(assign.args[1], y0) ||
            !parse_number(assign.args[2], vx) || !parse_number(assign.args[3], vy) ||
            !parse_number(assign.args[4], s_end)) {
            return std::unexpected(DomainError{
                "diffgeo_geodesic_euclidean",
                "expected diffgeo_geodesic_euclidean(x0,y0,vx,vy,s_end)"});
        }
        auto traj = eval_diffgeo_geodesic_euclidean(x0, y0, vx, vy, s_end);
        if (!traj) {
            return std::unexpected(traj.error());
        }
        result = *traj;
    } else if (assign.callee == "lz77_decode_vec" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto decoded = eval_lz77_decode_vec(*matrix);
        if (!decoded) {
            return std::unexpected(decoded.error());
        }
        result = *decoded;
    } else if (assign.callee == "bzip2_compress_vec" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto compressed = eval_bzip2_compress_vec(*matrix);
        if (!compressed) {
            return std::unexpected(compressed.error());
        }
        result = *compressed;
    } else if (assign.callee == "bzip2_decompress_vec" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto decompressed = eval_bzip2_decompress_vec(*matrix);
        if (!decompressed) {
            return std::unexpected(decompressed.error());
        }
        result = *decompressed;
    } else if (assign.callee == "combo_unrank_permutation" && assign.args.size() == 2) {
        double n_d = 0.0;
        double rank_d = 0.0;
        if (!parse_number(assign.args[0], n_d)) {
            auto it = state_.scalars.find(assign.args[0]);
            if (it != state_.scalars.end()) {
                n_d = it->second;
            } else {
                return std::unexpected(
                    DomainError{assign.callee, "expected numeric n argument"});
            }
        }
        if (!parse_number(assign.args[1], rank_d)) {
            auto it = state_.scalars.find(assign.args[1]);
            if (it != state_.scalars.end()) {
                rank_d = it->second;
            } else {
                return std::unexpected(
                    DomainError{assign.callee, "expected numeric rank argument"});
            }
        }
        const int n = static_cast<int>(n_d);
        if (n < 0 || n_d != n || rank_d < 0.0 || std::floor(rank_d) != rank_d) {
            return std::unexpected(DomainError{
                assign.callee, "expected non-negative integer n and rank"});
        }
        auto perm = eval_combo_unrank_permutation(n, static_cast<uint64_t>(rank_d));
        if (!perm) {
            return std::unexpected(perm.error());
        }
        result = *perm;
    } else if (assign.callee == "numthy_continued_fraction" && assign.args.size() == 2) {
        double x = 0.0;
        double n_d = 0.0;
        if (!parse_number(assign.args[0], x)) {
            auto it = state_.scalars.find(assign.args[0]);
            if (it != state_.scalars.end()) {
                x = it->second;
            } else {
                return std::unexpected(
                    DomainError{assign.callee, "expected numeric x argument"});
            }
        }
        if (!parse_number(assign.args[1], n_d)) {
            auto it = state_.scalars.find(assign.args[1]);
            if (it != state_.scalars.end()) {
                n_d = it->second;
            } else {
                return std::unexpected(
                    DomainError{assign.callee, "expected numeric n argument"});
            }
        }
        const int n = static_cast<int>(n_d);
        if (n < 1 || n_d != n) {
            return std::unexpected(DomainError{
                assign.callee, "expected positive integer n"});
        }
        auto cf = eval_numthy_continued_fraction(x, n);
        if (!cf) {
            return std::unexpected(cf.error());
        }
        result = *cf;
    } else if (assign.callee == "numthy_primes" && assign.args.size() == 2) {
        double lo_d = 0.0;
        double hi_d = 0.0;
        if (!parse_number(assign.args[0], lo_d)) {
            auto it = state_.scalars.find(assign.args[0]);
            if (it != state_.scalars.end()) {
                lo_d = it->second;
            } else {
                return std::unexpected(
                    DomainError{assign.callee, "expected numeric lo argument"});
            }
        }
        if (!parse_number(assign.args[1], hi_d)) {
            auto it = state_.scalars.find(assign.args[1]);
            if (it != state_.scalars.end()) {
                hi_d = it->second;
            } else {
                return std::unexpected(
                    DomainError{assign.callee, "expected numeric hi argument"});
            }
        }
        if (lo_d < 0.0 || hi_d < 0.0 || std::floor(lo_d) != lo_d || std::floor(hi_d) != hi_d) {
            return std::unexpected(
                DomainError{assign.callee, "expected non-negative integer bounds"});
        }
        auto primes = eval_numthy_primes(static_cast<uint64_t>(lo_d), static_cast<uint64_t>(hi_d));
        if (!primes) {
            return std::unexpected(primes.error());
        }
        result = *primes;
    } else if (assign.callee == "combo_unrank_combination" && assign.args.size() == 3) {
        double n_d = 0.0;
        double k_d = 0.0;
        double rank_d = 0.0;
        if (!parse_number(assign.args[0], n_d)) {
            auto it = state_.scalars.find(assign.args[0]);
            if (it != state_.scalars.end()) {
                n_d = it->second;
            } else {
                return std::unexpected(
                    DomainError{assign.callee, "expected numeric n argument"});
            }
        }
        if (!parse_number(assign.args[1], k_d)) {
            auto it = state_.scalars.find(assign.args[1]);
            if (it != state_.scalars.end()) {
                k_d = it->second;
            } else {
                return std::unexpected(
                    DomainError{assign.callee, "expected numeric k argument"});
            }
        }
        if (!parse_number(assign.args[2], rank_d)) {
            auto it = state_.scalars.find(assign.args[2]);
            if (it != state_.scalars.end()) {
                rank_d = it->second;
            } else {
                return std::unexpected(
                    DomainError{assign.callee, "expected numeric rank argument"});
            }
        }
        const int n = static_cast<int>(n_d);
        const int k = static_cast<int>(k_d);
        if (n < 0 || k < 0 || n_d != n || k_d != k || rank_d < 0.0 ||
            std::floor(rank_d) != rank_d) {
            return std::unexpected(DomainError{
                assign.callee, "expected non-negative integer n, k and rank"});
        }
        auto comb = eval_combo_unrank_combination(n, k, static_cast<uint64_t>(rank_d));
        if (!comb) {
            return std::unexpected(comb.error());
        }
        result = *comb;
    } else if (assign.callee == "quantum_coherent_state" && assign.args.size() == 3) {
        double alpha_re = 0.0;
        double alpha_im = 0.0;
        double n_max_d = 0.0;
        if (!parse_number(assign.args[0], alpha_re)) {
            auto it = state_.scalars.find(assign.args[0]);
            if (it != state_.scalars.end()) {
                alpha_re = it->second;
            } else {
                return std::unexpected(
                    DomainError{assign.callee, "expected numeric alpha_re argument"});
            }
        }
        if (!parse_number(assign.args[1], alpha_im)) {
            auto it = state_.scalars.find(assign.args[1]);
            if (it != state_.scalars.end()) {
                alpha_im = it->second;
            } else {
                return std::unexpected(
                    DomainError{assign.callee, "expected numeric alpha_im argument"});
            }
        }
        if (!parse_number(assign.args[2], n_max_d)) {
            auto it = state_.scalars.find(assign.args[2]);
            if (it != state_.scalars.end()) {
                n_max_d = it->second;
            } else {
                return std::unexpected(
                    DomainError{assign.callee, "expected numeric n_max argument"});
            }
        }
        const int n_max = static_cast<int>(n_max_d);
        if (n_max < 0 || n_max_d != n_max) {
            return std::unexpected(DomainError{
                assign.callee, "expected non-negative integer n_max"});
        }
        auto state = eval_quantum_coherent_state(alpha_re, alpha_im, n_max);
        if (!state) {
            return std::unexpected(state.error());
        }
        result = *state;
    } else if (assign.callee == "imgaussfilt" && assign.args.size() == 2) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        double sigma = 0.0;
        if (!parse_number(assign.args[1], sigma)) {
            return std::unexpected(DomainError{"imgaussfilt", "expected imgaussfilt(M, sigma)"});
        }
        auto gray = matrix_to_gray_image(*matrix);
        if (!gray) {
            return std::unexpected(gray.error());
        }
        result = gray_image_to_matrix(image::imgaussfilt(*gray, static_cast<float>(sigma)));
    } else if (assign.callee == "medfilt2" && assign.args.size() == 2) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        double ksize_d = 0.0;
        if (!parse_number(assign.args[1], ksize_d)) {
            return std::unexpected(DomainError{"medfilt2", "expected medfilt2(M, ksize)"});
        }
        const int ksize = static_cast<int>(ksize_d);
        if (ksize < 1 || ksize_d != ksize || (ksize % 2) == 0) {
            return std::unexpected(
                DomainError{"medfilt2", "expected positive odd integer ksize"});
        }
        auto gray = matrix_to_gray_image(*matrix);
        if (!gray) {
            return std::unexpected(gray.error());
        }
        result = gray_image_to_matrix(image::medfilt2(*gray, ksize));
    } else if (assign.callee == "boxfilter" && assign.args.size() == 2) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        double ksize_d = 0.0;
        if (!parse_number(assign.args[1], ksize_d)) {
            return std::unexpected(DomainError{"boxfilter", "expected boxfilter(M, ksize)"});
        }
        const int ksize = static_cast<int>(ksize_d);
        if (ksize < 1 || ksize_d != ksize || (ksize % 2) == 0) {
            return std::unexpected(
                DomainError{"boxfilter", "expected positive odd integer ksize"});
        }
        auto gray = matrix_to_gray_image(*matrix);
        if (!gray) {
            return std::unexpected(gray.error());
        }
        result = gray_image_to_matrix(image::boxfilter(*gray, ksize));
    } else if (assign.callee == "bilateral" && assign.args.size() == 3) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        double sigma_s = 0.0;
        double sigma_r = 0.0;
        if (!parse_number(assign.args[1], sigma_s) || !parse_number(assign.args[2], sigma_r)) {
            return std::unexpected(
                DomainError{"bilateral", "expected bilateral(M, sigma_s, sigma_r)"});
        }
        auto gray = matrix_to_gray_image(*matrix);
        if (!gray) {
            return std::unexpected(gray.error());
        }
        result = gray_image_to_matrix(
            image::bilateral(*gray, static_cast<float>(sigma_s), static_cast<float>(sigma_r)));
    } else if (assign.callee == "canny" && assign.args.size() == 3) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        double low = 0.0;
        double high = 0.0;
        if (!parse_number(assign.args[1], low) || !parse_number(assign.args[2], high)) {
            return std::unexpected(DomainError{"canny", "expected canny(M, low, high)"});
        }
        auto gray = matrix_to_gray_image(*matrix);
        if (!gray) {
            return std::unexpected(gray.error());
        }
        result = gray_image_to_matrix(
            image::canny(*gray, static_cast<float>(low), static_cast<float>(high)));
    } else if (assign.callee == "laplacian" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto gray = matrix_to_gray_image(*matrix);
        if (!gray) {
            return std::unexpected(gray.error());
        }
        result = gray_image_to_matrix(image::laplacian(*gray));
    } else if (assign.callee == "histeq" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto gray = matrix_to_gray_image(*matrix);
        if (!gray) {
            return std::unexpected(gray.error());
        }
        result = gray_image_to_matrix(image::histeq(*gray));
    } else if (assign.callee == "sharpen" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto gray = matrix_to_gray_image(*matrix);
        if (!gray) {
            return std::unexpected(gray.error());
        }
        result = gray_image_to_matrix(image::sharpen(*gray));
    } else if (assign.callee == "threshold_otsu" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto gray = matrix_to_gray_image(*matrix);
        if (!gray) {
            return std::unexpected(gray.error());
        }
        result = gray_image_to_matrix(image::threshold_otsu(*gray));
    } else if (assign.callee == "imresize" && assign.args.size() == 3) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        double rows_d = 0.0;
        double cols_d = 0.0;
        if (!parse_number(assign.args[1], rows_d) || !parse_number(assign.args[2], cols_d)) {
            return std::unexpected(DomainError{"imresize", "expected imresize(M, rows, cols)"});
        }
        auto gray = matrix_to_gray_image(*matrix);
        if (!gray) {
            return std::unexpected(gray.error());
        }
        result = gray_image_to_matrix(
            image::imresize(*gray, static_cast<int>(rows_d), static_cast<int>(cols_d)));
    } else if (assign.callee == "graph_pagerank" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto G = graph_from_adjacency(*matrix, "graph_pagerank");
        if (!G) {
            return std::unexpected(G.error());
        }
        result = vector_to_column(graph::pagerank(*G));
    } else if (assign.callee == "graph_betweenness" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto bc = eval_graph_betweenness(*matrix);
        if (!bc) {
            return std::unexpected(bc.error());
        }
        result = *bc;
    } else if (assign.callee == "graph_closeness" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto cc = eval_graph_closeness(*matrix);
        if (!cc) {
            return std::unexpected(cc.error());
        }
        result = *cc;
    } else if (assign.callee == "graph_degree_centrality" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto dc = eval_graph_degree_centrality(*matrix);
        if (!dc) {
            return std::unexpected(dc.error());
        }
        result = *dc;
    } else if (assign.callee == "fft_rfft" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto spectrum = eval_fft_rfft(*matrix);
        if (!spectrum) {
            return std::unexpected(spectrum.error());
        }
        result = *spectrum;
    } else if (assign.callee == "fft_dft" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto spectrum = eval_fft_dft(*matrix);
        if (!spectrum) {
            return std::unexpected(spectrum.error());
        }
        result = *spectrum;
    } else if (assign.callee == "fft_ifft" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto signal = eval_fft_ifft(*matrix);
        if (!signal) {
            return std::unexpected(signal.error());
        }
        result = *signal;
    } else if (assign.callee == "fft_fft2" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto spectrum = eval_fft_fft2(*matrix);
        if (!spectrum) {
            return std::unexpected(spectrum.error());
        }
        result = *spectrum;
    } else if (assign.callee == "graph_topological_sort" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto order = eval_graph_topological_sort(*matrix);
        if (!order) {
            return std::unexpected(order.error());
        }
        result = *order;
    } else if (assign.callee == "graph_greedy_colour" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto colors = eval_graph_greedy_colour(*matrix);
        if (!colors) {
            return std::unexpected(colors.error());
        }
        result = *colors;
    } else if (assign.callee == "graph_euler_circuit" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto circuit = eval_graph_euler_circuit(*matrix);
        if (!circuit) {
            return std::unexpected(circuit.error());
        }
        result = *circuit;
    } else if (assign.callee == "geo_delaunay_2d" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto tris = eval_geo_delaunay_2d(*matrix);
        if (!tris) {
            return std::unexpected(tris.error());
        }
        result = *tris;
    } else if (assign.callee == "geo_voronoi" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto verts = eval_geo_voronoi(*matrix);
        if (!verts) {
            return std::unexpected(verts.error());
        }
        result = *verts;
    } else if (assign.callee == "topo_pairwise_distances" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto dist = eval_topo_pairwise_distances(*matrix);
        if (!dist) {
            return std::unexpected(dist.error());
        }
        result = *dist;
    } else if (assign.callee == "combo_next_perm" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto perm = eval_combo_next_perm(*matrix);
        if (!perm) {
            return std::unexpected(perm.error());
        }
        result = *perm;
    } else if (assign.callee == "numthy_convergents" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto conv = eval_numthy_convergents(*matrix);
        if (!conv) {
            return std::unexpected(conv.error());
        }
        result = *conv;
    } else if (assign.callee == "ml_mat_transpose" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto At = eval_ml_mat_transpose(*matrix);
        if (!At) {
            return std::unexpected(At.error());
        }
        result = *At;
    } else if (assign.callee == "graph_scc" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto scc = eval_graph_scc(*matrix);
        if (!scc) {
            return std::unexpected(scc.error());
        }
        result = *scc;
    } else if (assign.callee == "prewitt" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto edge = eval_prewitt(*matrix);
        if (!edge) {
            return std::unexpected(edge.error());
        }
        result = *edge;
    } else if (assign.callee == "fftshift" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto shifted = eval_fftshift(*matrix);
        if (!shifted) {
            return std::unexpected(shifted.error());
        }
        result = *shifted;
    } else if (assign.callee == "poly_deriv" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto deriv = eval_poly_deriv(*matrix);
        if (!deriv) {
            return std::unexpected(deriv.error());
        }
        result = *deriv;
    } else if (assign.callee == "graph_floyd_warshall" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto dist = eval_graph_floyd_warshall(*matrix);
        if (!dist) {
            return std::unexpected(dist.error());
        }
        result = *dist;
    } else if (assign.callee == "graph_mst_kruskal" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto edges = eval_graph_mst_kruskal(*matrix);
        if (!edges) {
            return std::unexpected(edges.error());
        }
        result = *edges;
    } else if (assign.callee == "graph_mst_prim" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto edges = eval_graph_mst_prim(*matrix);
        if (!edges) {
            return std::unexpected(edges.error());
        }
        result = *edges;
    } else if (assign.callee == "fft_dct2" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto coeffs = eval_fft_dct2(*matrix);
        if (!coeffs) {
            return std::unexpected(coeffs.error());
        }
        result = *coeffs;
    } else if (assign.callee == "fft_idct2" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto signal = eval_fft_idct2(*matrix);
        if (!signal) {
            return std::unexpected(signal.error());
        }
        result = *signal;
    } else if (assign.callee == "fft_dst2" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto coeffs = eval_fft_dst2(*matrix);
        if (!coeffs) {
            return std::unexpected(coeffs.error());
        }
        result = *coeffs;
    } else if (assign.callee == "diffgeo_surface_normal_sphere" && assign.args.size() == 2) {
        double u = 0.0;
        double v = 0.0;
        if (!parse_number(assign.args[0], u)) {
            auto u_expr = eval_scalar_expr(state_, assign.args[0]);
            if (!u_expr) {
                return std::unexpected(u_expr.error());
            }
            u = *u_expr;
        }
        if (!parse_number(assign.args[1], v)) {
            auto v_expr = eval_scalar_expr(state_, assign.args[1]);
            if (!v_expr) {
                return std::unexpected(v_expr.error());
            }
            v = *v_expr;
        }
        auto normal = eval_diffgeo_surface_normal_sphere(u, v);
        if (!normal) {
            return std::unexpected(normal.error());
        }
        result = *normal;
    } else if (assign.callee == "quantum_hadamard" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto psi = matrix_to_ket2(*matrix, "quantum_hadamard");
        if (!psi) {
            return std::unexpected(psi.error());
        }
        const auto out = quantum::op_apply(quantum::hadamard(), *psi);
        Matrix<double> col(2, 1);
        col(0, 0) = out[0].real();
        col(1, 0) = out[1].real();
        result = col;
    } else if (assign.callee == "quantum_ket_normalise" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        result = eval_quantum_ket_normalise_matrix(*matrix);
    } else if (assign.callee == "quantum_density_matrix" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        result = eval_quantum_density_matrix(*matrix);
    } else if (assign.callee == "quantum_ket_superposition" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        result = eval_quantum_ket_superposition_matrix(*matrix);
    } else if (assign.callee == "quantum_ket_basis" && assign.args.size() == 2) {
        double dim_d = 0.0;
        double index_d = 0.0;
        if (!parse_number(assign.args[0], dim_d)) {
            auto it = state_.scalars.find(assign.args[0]);
            if (it != state_.scalars.end()) {
                dim_d = it->second;
            } else {
                return std::unexpected(
                    DomainError{assign.callee, "expected numeric dim argument"});
            }
        }
        if (!parse_number(assign.args[1], index_d)) {
            auto it = state_.scalars.find(assign.args[1]);
            if (it != state_.scalars.end()) {
                index_d = it->second;
            } else {
                return std::unexpected(
                    DomainError{assign.callee, "expected numeric index argument"});
            }
        }
        const int dim = static_cast<int>(dim_d);
        const int index = static_cast<int>(index_d);
        if (dim < 1 || dim_d != dim || index_d != index) {
            return std::unexpected(DomainError{
                assign.callee, "expected positive integer dim and integer index"});
        }
        result = eval_quantum_ket_basis(dim, index);
    } else if (assign.callee == "quantum_fock_state" && assign.args.size() == 2) {
        double n_d = 0.0;
        double n_max_d = 0.0;
        if (!parse_number(assign.args[0], n_d)) {
            auto it = state_.scalars.find(assign.args[0]);
            if (it != state_.scalars.end()) {
                n_d = it->second;
            } else {
                return std::unexpected(
                    DomainError{assign.callee, "expected numeric n argument"});
            }
        }
        if (!parse_number(assign.args[1], n_max_d)) {
            auto it = state_.scalars.find(assign.args[1]);
            if (it != state_.scalars.end()) {
                n_max_d = it->second;
            } else {
                return std::unexpected(
                    DomainError{assign.callee, "expected numeric n_max argument"});
            }
        }
        const int n = static_cast<int>(n_d);
        const int n_max = static_cast<int>(n_max_d);
        if (n_max < 0 || n_d != n || n_max_d != n_max) {
            return std::unexpected(DomainError{
                assign.callee, "expected non-negative integer n and n_max"});
        }
        result = eval_quantum_fock_state(n, n_max);
    } else if (assign.callee == "quantum_partial_trace" && assign.args.size() == 4) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        double d1_d = 0.0;
        double d2_d = 0.0;
        double sub_d = 0.0;
        if (!parse_number(assign.args[1], d1_d) || !parse_number(assign.args[2], d2_d) ||
            !parse_number(assign.args[3], sub_d)) {
            return std::unexpected(DomainError{
                "quantum_partial_trace",
                "expected quantum_partial_trace(rho, d1, d2, subsystem)"});
        }
        const int d1 = static_cast<int>(d1_d);
        const int d2 = static_cast<int>(d2_d);
        const int subsystem = static_cast<int>(sub_d);
        if (d1 < 1 || d2 < 1 || d1_d != d1 || d2_d != d2 || (subsystem != 0 && subsystem != 1) ||
            sub_d != subsystem) {
            return std::unexpected(DomainError{
                "quantum_partial_trace",
                "expected positive integer d1, d2 and subsystem 0 or 1"});
        }
        result = eval_quantum_partial_trace_matrix(*matrix, d1, d2, subsystem);
    } else if (assign.callee == "quantum_schrodinger" && assign.args.size() == 5) {
        auto H_m = resolve_operand(assign.args[0]);
        if (!H_m) {
            return std::unexpected(H_m.error());
        }
        auto psi0_m = resolve_operand(assign.args[1]);
        if (!psi0_m) {
            return std::unexpected(psi0_m.error());
        }
        double t0 = 0.0;
        double t1 = 0.0;
        double n_steps_d = 0.0;
        if (!parse_number(assign.args[2], t0) || !parse_number(assign.args[3], t1) ||
            !parse_number(assign.args[4], n_steps_d)) {
            return std::unexpected(DomainError{
                "quantum_schrodinger",
                "expected quantum_schrodinger(H, psi0, t0, t1, n_steps)"});
        }
        const int n_steps = static_cast<int>(n_steps_d);
        if (n_steps < 0 || n_steps_d != n_steps) {
            return std::unexpected(DomainError{
                "quantum_schrodinger", "expected non-negative integer n_steps"});
        }
        result = eval_quantum_schrodinger_matrix(*H_m, *psi0_m, t0, t1, n_steps);
    } else if (assign.callee == "quantum_schrodinger_final" && assign.args.size() == 5) {
        auto H_m = resolve_operand(assign.args[0]);
        if (!H_m) {
            return std::unexpected(H_m.error());
        }
        auto psi0_m = resolve_operand(assign.args[1]);
        if (!psi0_m) {
            return std::unexpected(psi0_m.error());
        }
        double t0 = 0.0;
        double t1 = 0.0;
        double n_steps_d = 0.0;
        if (!parse_number(assign.args[2], t0) || !parse_number(assign.args[3], t1) ||
            !parse_number(assign.args[4], n_steps_d)) {
            return std::unexpected(DomainError{
                "quantum_schrodinger_final",
                "expected quantum_schrodinger_final(H, psi0, t0, t1, n_steps)"});
        }
        const int n_steps = static_cast<int>(n_steps_d);
        if (n_steps < 0 || n_steps_d != n_steps) {
            return std::unexpected(DomainError{
                "quantum_schrodinger_final", "expected non-negative integer n_steps"});
        }
        result = eval_quantum_schrodinger_final(*H_m, *psi0_m, t0, t1, n_steps);
    } else if (assign.callee == "pde_heat_1d" && assign.args.size() == 5) {
        auto x0_m = resolve_operand(assign.args[0]);
        if (!x0_m) {
            return std::unexpected(x0_m.error());
        }
        double alpha = 0.0;
        double dx = 0.0;
        double dt = 0.0;
        double steps_d = 0.0;
        if (!parse_number(assign.args[1], alpha) || !parse_number(assign.args[2], dx) ||
            !parse_number(assign.args[3], dt) || !parse_number(assign.args[4], steps_d)) {
            return std::unexpected(DomainError{
                "pde_heat_1d", "expected pde_heat_1d(x0, alpha, dx, dt, steps)"});
        }
        const int steps_i = static_cast<int>(steps_d);
        if (steps_i < 0 || steps_d != steps_i) {
            return std::unexpected(
                DomainError{"pde_heat_1d", "expected non-negative integer steps"});
        }
        result = eval_pde_heat_1d(*x0_m, alpha, dx, dt, static_cast<std::size_t>(steps_i));
    } else if (assign.callee == "pde_advection_1d" && assign.args.size() == 5) {
        auto u0_m = resolve_operand(assign.args[0]);
        if (!u0_m) {
            return std::unexpected(u0_m.error());
        }
        double v = 0.0;
        double dx = 0.0;
        double dt = 0.0;
        double steps_d = 0.0;
        if (!parse_number(assign.args[1], v) || !parse_number(assign.args[2], dx) ||
            !parse_number(assign.args[3], dt) || !parse_number(assign.args[4], steps_d)) {
            return std::unexpected(DomainError{
                "pde_advection_1d", "expected pde_advection_1d(u0, v, dx, dt, steps)"});
        }
        const int steps_i = static_cast<int>(steps_d);
        if (steps_i < 0 || steps_d != steps_i) {
            return std::unexpected(
                DomainError{"pde_advection_1d", "expected non-negative integer steps"});
        }
        result = eval_pde_advection_1d(*u0_m, v, dx, dt, static_cast<std::size_t>(steps_i));
    } else if (assign.callee == "pde_poisson_2d" && assign.args.size() == 5) {
        auto f_m = resolve_operand(assign.args[0]);
        if (!f_m) {
            return std::unexpected(f_m.error());
        }
        double dx = 0.0;
        double dy = 0.0;
        double max_iter_d = 0.0;
        double tolerance = 0.0;
        if (!parse_number(assign.args[1], dx) || !parse_number(assign.args[2], dy) ||
            !parse_number(assign.args[3], max_iter_d) ||
            !parse_number(assign.args[4], tolerance)) {
            return std::unexpected(DomainError{
                "pde_poisson_2d",
                "expected pde_poisson_2d(f, dx, dy, max_iterations, tolerance)"});
        }
        const int max_iter_i = static_cast<int>(max_iter_d);
        if (max_iter_i < 0 || max_iter_d != max_iter_i) {
            return std::unexpected(DomainError{
                "pde_poisson_2d", "expected non-negative integer max_iterations"});
        }
        result = eval_pde_poisson_2d(*f_m, dx, dy, static_cast<std::size_t>(max_iter_i),
                                     tolerance);
    } else if (assign.callee == "pde_burgers_1d" && assign.args.size() == 5) {
        auto u0_m = resolve_operand(assign.args[0]);
        if (!u0_m) {
            return std::unexpected(u0_m.error());
        }
        double nu = 0.0;
        double dx = 0.0;
        double dt = 0.0;
        double steps_d = 0.0;
        if (!parse_number(assign.args[1], nu) || !parse_number(assign.args[2], dx) ||
            !parse_number(assign.args[3], dt) || !parse_number(assign.args[4], steps_d)) {
            return std::unexpected(DomainError{
                "pde_burgers_1d", "expected pde_burgers_1d(u0, nu, dx, dt, steps)"});
        }
        const int steps_i = static_cast<int>(steps_d);
        if (steps_i < 0 || steps_d != steps_i) {
            return std::unexpected(
                DomainError{"pde_burgers_1d", "expected non-negative integer steps"});
        }
        result = eval_pde_burgers_1d(*u0_m, nu, dx, dt, static_cast<std::size_t>(steps_i));
    } else if (assign.callee == "pde_heat_2d" && assign.args.size() == 6) {
        auto u0_m = resolve_operand(assign.args[0]);
        if (!u0_m) {
            return std::unexpected(u0_m.error());
        }
        double alpha = 0.0;
        double dx = 0.0;
        double dy = 0.0;
        double dt = 0.0;
        double steps_d = 0.0;
        if (!parse_number(assign.args[1], alpha) || !parse_number(assign.args[2], dx) ||
            !parse_number(assign.args[3], dy) || !parse_number(assign.args[4], dt) ||
            !parse_number(assign.args[5], steps_d)) {
            return std::unexpected(DomainError{
                "pde_heat_2d", "expected pde_heat_2d(u0, alpha, dx, dy, dt, steps)"});
        }
        const int steps_i = static_cast<int>(steps_d);
        if (steps_i < 0 || steps_d != steps_i) {
            return std::unexpected(
                DomainError{"pde_heat_2d", "expected non-negative integer steps"});
        }
        result = eval_pde_heat_2d(*u0_m, alpha, dx, dy, dt, static_cast<std::size_t>(steps_i));
    } else if (assign.callee == "pde_wave_1d" && assign.args.size() == 6) {
        auto u0_m = resolve_operand(assign.args[0]);
        if (!u0_m) {
            return std::unexpected(u0_m.error());
        }
        auto v0_m = resolve_operand(assign.args[1]);
        if (!v0_m) {
            return std::unexpected(v0_m.error());
        }
        double c = 0.0;
        double dx = 0.0;
        double dt = 0.0;
        double steps_d = 0.0;
        if (!parse_number(assign.args[2], c) || !parse_number(assign.args[3], dx) ||
            !parse_number(assign.args[4], dt) || !parse_number(assign.args[5], steps_d)) {
            return std::unexpected(DomainError{
                "pde_wave_1d", "expected pde_wave_1d(u0, v0, c, dx, dt, steps)"});
        }
        const int steps_i = static_cast<int>(steps_d);
        if (steps_i < 0 || steps_d != steps_i) {
            return std::unexpected(
                DomainError{"pde_wave_1d", "expected non-negative integer steps"});
        }
        result = eval_pde_wave_1d(*u0_m, *v0_m, c, dx, dt, static_cast<std::size_t>(steps_i));
    } else if (assign.callee == "imcrop" && assign.args.size() == 5) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        double r0_d = 0.0;
        double c0_d = 0.0;
        double r1_d = 0.0;
        double c1_d = 0.0;
        if (!parse_number(assign.args[1], r0_d) || !parse_number(assign.args[2], c0_d) ||
            !parse_number(assign.args[3], r1_d) || !parse_number(assign.args[4], c1_d)) {
            return std::unexpected(
                DomainError{"imcrop", "expected imcrop(M, r0, c0, r1, c1)"});
        }
        const int r0 = static_cast<int>(r0_d);
        const int c0 = static_cast<int>(c0_d);
        const int r1 = static_cast<int>(r1_d);
        const int c1 = static_cast<int>(c1_d);
        if (r0_d != r0 || c0_d != c0 || r1_d != r1 || c1_d != c1) {
            return std::unexpected(DomainError{"imcrop", "expected integer crop bounds"});
        }
        auto cropped = eval_imcrop(*matrix, r0, c0, r1, c1);
        if (!cropped) {
            return std::unexpected(cropped.error());
        }
        result = *cropped;
    }
    if (!result) {
        return std::unexpected(result.error());
    }

    state_.matrices[assign.target] = *result;
    std::ostringstream out;
    out << assign.target << " =\n";
    print_matrix(out, *result);
    return out.str();
}

Result<std::string> Interpreter::assign_scalar_matrix_call(const ScalarMatrixCallAssign& assign) {
    auto matrix = resolve_matrix(assign.arg);
    if (!matrix) {
        matrix = parse_matrix(assign.arg);
    }
    if (!matrix) {
        return std::unexpected(matrix.error());
    }

    Result<double> value = std::unexpected(DomainError{"assign", "unsupported scalar matrix call"});
    if (assign.callee == "det") {
        value = det(*matrix);
    } else if (assign.callee == "trace") {
        value = trace(*matrix);
    } else if (assign.callee == "norm") {
        value = norm(*matrix);
    } else if (assign.callee == "rank") {
        auto ranked = rank(*matrix);
        if (!ranked) {
            return std::unexpected(ranked.error());
        }
        value = static_cast<double>(*ranked);
    } else if (assign.callee == "cond") {
        value = cond(*matrix);
    } else if (assign.callee == "geo_convex_hull_area") {
        auto points = matrix_to_points2d(*matrix, "geo_convex_hull_area");
        if (!points) {
            return std::unexpected(points.error());
        }
        if (points->size() < 3) {
            return std::unexpected(
                DomainError{"geo_convex_hull_area", "expected at least 3 points"});
        }
        const auto hull = geo::convex_hull_2d(*points);
        if (hull.size() < 3) {
            return std::unexpected(
                DomainError{"geo_convex_hull_area", "degenerate convex hull"});
        }
        value = geo::area(hull);
    } else if (assign.callee == "geo_polygon_area") {
        value = eval_geo_polygon_area(*matrix);
    } else if (assign.callee == "geo_polygon_perimeter") {
        value = eval_geo_polygon_perimeter(*matrix);
    } else if (assign.callee == "geo_signed_area") {
        value = eval_geo_signed_area(*matrix);
    } else if (assign.callee == "geo_moment_of_inertia") {
        value = eval_geo_moment_of_inertia(*matrix);
    } else if (assign.callee == "geo_centroid_x") {
        value = eval_geo_centroid_x(*matrix);
    } else if (assign.callee == "geo_centroid_y") {
        value = eval_geo_centroid_y(*matrix);
    } else if (assign.callee == "bwt_primary_index") {
        value = eval_bwt_primary_index(*matrix);
    } else if (assign.callee == "combo_rank_permutation") {
        value = eval_combo_rank_permutation(*matrix);
    } else if (assign.callee == "ml_vec_norm") {
        value = eval_ml_vec_norm(*matrix);
    } else if (assign.callee == "info_entropy") {
        value = eval_info_entropy(*matrix);
    } else if (assign.callee == "info_lz_complexity") {
        value = eval_info_lz_complexity(*matrix);
    } else if (assign.callee == "info_redundancy") {
        value = eval_info_redundancy(*matrix);
    } else if (assign.callee == "info_efficiency") {
        value = eval_info_efficiency(*matrix);
    } else if (assign.callee == "info_source_coding_rate") {
        value = eval_info_source_coding_rate(*matrix);
    } else if (assign.callee == "info_mutual_info") {
        value = eval_info_mutual_info(*matrix);
    } else if (assign.callee == "finance_irr") {
        value = eval_finance_irr(*matrix);
    } else if (assign.callee == "finance_sharpe") {
        value = eval_finance_sharpe(*matrix);
    } else if (assign.callee == "finance_sortino") {
        value = eval_finance_sortino(*matrix);
    } else if (assign.callee == "finance_var") {
        value = eval_finance_var(*matrix);
    } else if (assign.callee == "finance_cvar") {
        value = eval_finance_cvar(*matrix);
    } else if (assign.callee == "finance_max_drawdown") {
        value = eval_finance_max_drawdown(*matrix);
    } else if (assign.callee == "quantum_von_neumann_entropy") {
        value = eval_quantum_von_neumann_entropy(*matrix);
    } else if (assign.callee == "quantum_concurrence") {
        value = eval_quantum_concurrence(*matrix);
    } else if (assign.callee == "tensorops_norm") {
        value = eval_tensorops_norm(*matrix);
    } else if (assign.callee == "graph_diameter") {
        value = eval_graph_diameter(*matrix);
    } else if (assign.callee == "graph_radius") {
        value = eval_graph_radius(*matrix);
    } else if (assign.callee == "graph_is_bipartite") {
        value = eval_graph_is_bipartite(*matrix);
    } else if (assign.callee == "graph_is_dag") {
        value = eval_graph_is_dag(*matrix);
    } else if (assign.callee == "stats_mean") {
        value = eval_stats_mean(*matrix);
    } else if (assign.callee == "stats_median") {
        value = eval_stats_median(*matrix);
    } else if (assign.callee == "stats_stddev") {
        value = eval_stats_stddev(*matrix);
    } else if (assign.callee == "stats_skewness") {
        value = eval_stats_skewness(*matrix);
    } else if (assign.callee == "stats_kurtosis") {
        value = eval_stats_kurtosis(*matrix);
    } else if (assign.callee == "stats_var") {
        value = eval_stats_var(*matrix);
    } else if (assign.callee == "stats_mode") {
        value = eval_stats_mode(*matrix);
    } else if (assign.callee == "stats_geometric_mean") {
        value = eval_stats_geometric_mean(*matrix);
    } else if (assign.callee == "stats_harmonic_mean") {
        value = eval_stats_harmonic_mean(*matrix);
    } else if (assign.callee == "stats_rms") {
        value = eval_stats_rms(*matrix);
    } else if (assign.callee == "stats_mad") {
        value = eval_stats_mad(*matrix);
    } else if (assign.callee == "stats_iqr") {
        value = eval_stats_iqr(*matrix);
    } else if (assign.callee == "stats_min_value") {
        value = eval_stats_min_value(*matrix);
    } else if (assign.callee == "count_components") {
        value = eval_count_components(*matrix);
    } else if (assign.callee == "graph_is_connected") {
        value = eval_graph_is_connected(*matrix);
    } else if (assign.callee == "graph_is_tree") {
        value = eval_graph_is_tree(*matrix);
    }
    if (!value) {
        return std::unexpected(value.error());
    }
    return assign_scalar(assign.target, *value);
}

Result<std::string> Interpreter::assign_multi_matrix_call(const MultiMatrixCallAssign& assign) {
    auto matrix = resolve_matrix(assign.arg);
    if (!matrix) {
        matrix = parse_matrix(assign.arg);
    }
    if (!matrix) {
        return std::unexpected(matrix.error());
    }

    std::ostringstream out;
    if (assign.callee == "lu") {
        auto factored = lu(*matrix);
        if (!factored) {
            return std::unexpected(factored.error());
        }
        const auto& [L, U, P] = *factored;
        state_.matrices[assign.targets[0]] = L;
        state_.matrices[assign.targets[1]] = U;
        out << assign.targets[0] << " =\n";
        print_matrix(out, L);
        out << assign.targets[1] << " =\n";
        print_matrix(out, U);
        if (assign.targets.size() == 3) {
            state_.matrices[assign.targets[2]] = P;
            out << assign.targets[2] << " =\n";
            print_matrix(out, P);
        }
        return out.str();
    }

    if (assign.callee == "qr") {
        auto factored = qr(*matrix);
        if (!factored) {
            return std::unexpected(factored.error());
        }
        const auto& [Q, R] = *factored;
        state_.matrices[assign.targets[0]] = Q;
        state_.matrices[assign.targets[1]] = R;
        out << assign.targets[0] << " =\n";
        print_matrix(out, Q);
        out << assign.targets[1] << " =\n";
        print_matrix(out, R);
        return out.str();
    }

    if (assign.callee == "svd") {
        auto decomp = svd(*matrix);
        if (!decomp) {
            return std::unexpected(decomp.error());
        }
        state_.matrices[assign.targets[0]] = decomp->U;
        state_.matrices[assign.targets[1]] = decomp->S;
        out << assign.targets[0] << " =\n";
        print_matrix(out, decomp->U);
        out << assign.targets[1] << " =\n";
        print_matrix(out, decomp->S);
        if (assign.targets.size() == 3) {
            state_.matrices[assign.targets[2]] = decomp->V;
            out << assign.targets[2] << " =\n";
            print_matrix(out, decomp->V);
        }
        return out.str();
    }

    if (assign.callee == "eig_sym") {
        auto decomp = eig_sym(*matrix);
        if (!decomp) {
            return std::unexpected(decomp.error());
        }
        state_.matrices[assign.targets[0]] = decomp->values;
        state_.matrices[assign.targets[1]] = decomp->vectors;
        out << assign.targets[0] << " =\n";
        print_matrix(out, decomp->values);
        out << assign.targets[1] << " =\n";
        print_matrix(out, decomp->vectors);
        return out.str();
    }

    return std::unexpected(DomainError{"assign", "unsupported multi matrix call"});
}

Result<Matrix<double>> Interpreter::eval_matrix_operand(const std::string& text) {
    const std::string s = trim(text);
    auto matrix = resolve_matrix(s);
    if (matrix) {
        return matrix;
    }
    auto parsed = parse_matrix(s);
    if (parsed) {
        return parsed;
    }
    MatrixCallAssign nested;
    if (try_parse_matrix_call_assignment("_ = " + s, nested)) {
        static constexpr const char* kTmp = "__ms_nested_eval__";
        const auto prev = state_.matrices.find(kTmp);
        const bool existed = prev != state_.matrices.end();
        Matrix<double> saved;
        if (existed) {
            saved = prev->second;
        }
        nested.target = kTmp;
        const auto out = assign_matrix_call(nested);
        if (!out) {
            return std::unexpected(out.error());
        }
        const auto it = state_.matrices.find(kTmp);
        if (it == state_.matrices.end()) {
            return std::unexpected(DomainError{"eval", "matrix call produced no result"});
        }
        Matrix<double> result = it->second;
        if (existed) {
            state_.matrices[kTmp] = saved;
        } else {
            state_.matrices.erase(kTmp);
        }
        return result;
    }
    if (const auto nullary = parse_nullary_matrix_call(s)) {
        return eval_nullary_matrix_call(*nullary);
    }
    return std::unexpected(DomainError{"resolve", "unknown matrix: " + s});
}

Result<Matrix<double>> Interpreter::resolve_matrix(const std::string& name) const {
    const auto it = state_.matrices.find(name);
    if (it == state_.matrices.end()) {
        return std::unexpected(DomainError{"resolve", "unknown matrix: " + name});
    }
    return it->second;
}

void print_matrix(std::ostream& out, const Matrix<double>& m) {
    out << std::fixed << std::setprecision(6);
    for (size_t i = 0; i < m.rows(); ++i) {
        out << "  [";
        for (size_t j = 0; j < m.cols(); ++j) {
            if (j > 0) {
                out << ", ";
            }
            out << m(i, j);
        }
        out << "]\n";
    }
}

std::string matrix_to_line(const Matrix<double>& m) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < m.rows(); ++i) {
        if (i > 0) {
            out << "; ";
        }
        for (size_t j = 0; j < m.cols(); ++j) {
            if (j > 0) {
                out << ", ";
            }
            out << m(i, j);
        }
    }
    out << "]";
    return out.str();
}

const char* plot_kind_name(PlotSeries::Kind kind) {
    switch (kind) {
    case PlotSeries::Kind::Line:
        return "line";
    case PlotSeries::Kind::Bar:
        return "bar";
    case PlotSeries::Kind::Scatter:
        return "scatter";
    case PlotSeries::Kind::Heatmap:
        return "heatmap";
    case PlotSeries::Kind::Spy:
        return "spy";
    case PlotSeries::Kind::Surface3D:
        return "surface3d";
    }
    return "line";
}

std::optional<PlotSeries::Kind> parse_plot_kind(const std::string& text) {
    const std::string kind = lower(trim_copy(text));
    if (kind == "line") {
        return PlotSeries::Kind::Line;
    }
    if (kind == "bar") {
        return PlotSeries::Kind::Bar;
    }
    if (kind == "scatter") {
        return PlotSeries::Kind::Scatter;
    }
    if (kind == "heatmap") {
        return PlotSeries::Kind::Heatmap;
    }
    if (kind == "spy") {
        return PlotSeries::Kind::Spy;
    }
    if (kind == "surface3d" || kind == "surface") {
        return PlotSeries::Kind::Surface3D;
    }
    return std::nullopt;
}

std::vector<double> parse_double_list(const std::string& text) {
    std::vector<double> values;
    std::stringstream stream(text);
    std::string cell;
    while (std::getline(stream, cell, ',')) {
        cell = trim_copy(cell);
        if (cell.empty()) {
            continue;
        }
        double value = 0.0;
        if (parse_number(cell, value)) {
            values.push_back(value);
        }
    }
    return values;
}

void write_double_list(std::ostream& out, const char* label, const std::vector<double>& values) {
    if (values.empty()) {
        return;
    }
    out << label;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            out << ',';
        }
        out << values[i];
    }
    out << '\n';
}

Result<std::string> Interpreter::set_plot(const Matrix<double>& xs, const Matrix<double>& ys,
                                          PlotSeries::Kind kind) {
    const size_t n = ys.rows() * ys.cols();
    if (n == 0) {
        return std::unexpected(DomainError{"plot", "empty data"});
    }
    std::vector<double> x;
    std::vector<double> y;
    y.reserve(n);
    if (xs.rows() == 0 && xs.cols() == 0) {
        x.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            x.push_back(static_cast<double>(i));
        }
    } else {
        const size_t xn = xs.rows() * xs.cols();
        if (xn != n) {
            return std::unexpected(DimensionMismatch{xn, n});
        }
        x.reserve(n);
        if (xs.cols() == 1) {
            for (size_t i = 0; i < xs.rows(); ++i) {
                x.push_back(xs(i, 0));
            }
        } else {
            for (size_t i = 0; i < xs.cols(); ++i) {
                x.push_back(xs(0, i));
            }
        }
    }
    if (ys.cols() == 1) {
        for (size_t i = 0; i < ys.rows(); ++i) {
            y.push_back(ys(i, 0));
        }
    } else {
        for (size_t i = 0; i < ys.cols(); ++i) {
            y.push_back(ys(0, i));
        }
    }
    state_.plot.x = std::move(x);
    state_.plot.y = std::move(y);
    state_.plot.kind = kind;
    state_.plot.matrix_rows = 0;
    state_.plot.matrix_cols = 0;
    state_.plot.nnz = 0;
    state_.plot.grid = Matrix<double>{};
    state_.plot.valid = true;
    const char* label = kind == PlotSeries::Kind::Scatter ? "scatter" : "plot";
    return std::string{label} + " updated (" + std::to_string(state_.plot.y.size()) + " points)\n" +
           format_plot_preview(state_.plot);
}

Result<std::string> Interpreter::set_plot_heatmap(const Matrix<double>& m) {
    if (m.rows() == 0 || m.cols() == 0) {
        return std::unexpected(DomainError{"imshow", "empty matrix"});
    }
    state_.plot.x.clear();
    state_.plot.y.clear();
    state_.plot.kind = PlotSeries::Kind::Heatmap;
    state_.plot.matrix_rows = m.rows();
    state_.plot.matrix_cols = m.cols();
    state_.plot.nnz = 0;
    state_.plot.grid = m;
    state_.plot.valid = true;
    return "imshow updated (" + std::to_string(m.rows()) + "x" + std::to_string(m.cols()) + ")\n" +
           format_plot_preview(state_.plot);
}

Result<std::string> Interpreter::set_plot_spy(const Matrix<double>& m) {
    size_t count = 0;
    for (size_t i = 0; i < m.rows(); ++i) {
        for (size_t j = 0; j < m.cols(); ++j) {
            if (m(i, j) != 0.0) {
                ++count;
            }
        }
    }
    state_.plot.x.clear();
    state_.plot.y.clear();
    state_.plot.kind = PlotSeries::Kind::Spy;
    state_.plot.matrix_rows = m.rows();
    state_.plot.matrix_cols = m.cols();
    state_.plot.nnz = count;
    state_.plot.grid = m;
    state_.plot.valid = true;
    return "spy updated (" + std::to_string(count) + " nonzeros)\n" + format_plot_preview(state_.plot);
}

Result<std::string> Interpreter::set_plot_surf(const Matrix<double>& z) {
    if (z.rows() == 0 || z.cols() == 0) {
        return std::unexpected(DomainError{"surf", "empty matrix"});
    }
    state_.plot.x.clear();
    state_.plot.y.clear();
    state_.plot.kind = PlotSeries::Kind::Surface3D;
    state_.plot.matrix_rows = z.rows();
    state_.plot.matrix_cols = z.cols();
    state_.plot.nnz = 0;
    state_.plot.grid = z;
    state_.plot.valid = true;
    return "surf updated (" + std::to_string(z.rows()) + "x" + std::to_string(z.cols()) +
           " grid)\n" + format_plot_preview(state_.plot);
}

Result<std::string> Interpreter::set_plot_surf(const Matrix<double>& x, const Matrix<double>& y,
                                               const Matrix<double>& z) {
    if (x.rows() != y.rows() || x.cols() != y.cols() || x.rows() != z.rows() ||
        x.cols() != z.cols()) {
        return std::unexpected(DimensionMismatch{x.rows() * x.cols(), z.rows() * z.cols()});
    }
    if (z.rows() == 0 || z.cols() == 0) {
        return std::unexpected(DomainError{"surf", "empty grid"});
    }
    state_.plot.x.clear();
    state_.plot.y.clear();
    state_.plot.kind = PlotSeries::Kind::Surface3D;
    state_.plot.matrix_rows = z.rows();
    state_.plot.matrix_cols = z.cols();
    state_.plot.nnz = 0;
    state_.plot.grid = z;
    state_.plot.valid = true;
    return "surf updated (" + std::to_string(z.rows()) + "x" + std::to_string(z.cols()) +
           " mesh)\n" + format_plot_preview(state_.plot);
}

Result<std::string> Interpreter::set_plot_bars(std::vector<double> x, std::vector<double> y) {
    if (x.size() != y.size() || x.empty()) {
        return std::unexpected(DimensionMismatch{x.size(), y.size()});
    }
    state_.plot.x = std::move(x);
    state_.plot.y = std::move(y);
    state_.plot.kind = PlotSeries::Kind::Bar;
    state_.plot.grid = Matrix<double>{};
    state_.plot.valid = true;
    return "histogram updated (" + std::to_string(state_.plot.y.size()) + " bins)\n" +
           format_plot_preview(state_.plot);
}

Result<void> Interpreter::save_session(const std::string& path) const {
    std::ofstream out(path);
    if (!out) {
        return std::unexpected(DomainError{"save", "cannot open: " + path});
    }
    out << "# MathScript session\n";
    for (const auto& [name, value] : state_.scalars) {
        out << "scalar " << name << " = " << value << "\n";
    }
    for (const auto& [name, matrix] : state_.matrices) {
        out << "matrix " << name << " = " << matrix_to_line(matrix) << "\n";
    }
    if (state_.plot.valid) {
        out << "plot meta valid=1 kind=" << plot_kind_name(state_.plot.kind)
            << " rows=" << state_.plot.matrix_rows << " cols=" << state_.plot.matrix_cols
            << " nnz=" << state_.plot.nnz << "\n";
        write_double_list(out, "plot x ", state_.plot.x);
        write_double_list(out, "plot y ", state_.plot.y);
        if (state_.plot.grid.rows() > 0 && state_.plot.grid.cols() > 0) {
            out << "plot grid = " << matrix_to_line(state_.plot.grid) << "\n";
        }
    }
    return {};
}

Result<void> Interpreter::load_session(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        return std::unexpected(DomainError{"load", "cannot open: " + path});
    }
    reset();
    PlotSeries pending_plot{};
    bool have_plot = false;
    std::string line;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }
        if (line.rfind("plot meta ", 0) == 0) {
            const std::string meta = trim(line.substr(10));
            have_plot = meta.find("valid=1") != std::string::npos;
            if (const auto kind_pos = meta.find("kind="); kind_pos != std::string::npos) {
                const auto kind_end = meta.find(' ', kind_pos + 5);
                const std::string kind_text = meta.substr(kind_pos + 5, kind_end - (kind_pos + 5));
                if (const auto kind = parse_plot_kind(kind_text)) {
                    pending_plot.kind = *kind;
                }
            }
            if (const auto rows_pos = meta.find("rows="); rows_pos != std::string::npos) {
                pending_plot.matrix_rows =
                    static_cast<size_t>(std::strtoull(meta.c_str() + rows_pos + 5, nullptr, 10));
            }
            if (const auto cols_pos = meta.find("cols="); cols_pos != std::string::npos) {
                pending_plot.matrix_cols =
                    static_cast<size_t>(std::strtoull(meta.c_str() + cols_pos + 5, nullptr, 10));
            }
            if (const auto nnz_pos = meta.find("nnz="); nnz_pos != std::string::npos) {
                pending_plot.nnz =
                    static_cast<size_t>(std::strtoull(meta.c_str() + nnz_pos + 4, nullptr, 10));
            }
            continue;
        }
        if (line.rfind("plot x ", 0) == 0) {
            pending_plot.x = parse_double_list(trim(line.substr(7)));
            continue;
        }
        if (line.rfind("plot y ", 0) == 0) {
            pending_plot.y = parse_double_list(trim(line.substr(7)));
            continue;
        }
        if (line.rfind("plot grid = ", 0) == 0) {
            auto grid = parse_matrix(trim(line.substr(12)));
            if (!grid) {
                return std::unexpected(grid.error());
            }
            pending_plot.grid = *grid;
            continue;
        }
        const auto eq = line.find('=');
        if (eq == std::string::npos) {
            continue;
        }
        const std::string lhs = trim(line.substr(0, eq));
        const std::string rhs = trim(line.substr(eq + 1));
        const auto space = lhs.find(' ');
        if (space == std::string::npos) {
            continue;
        }
        const std::string kind = lower(lhs.substr(0, space));
        const std::string name = trim(lhs.substr(space + 1));
        if (kind == "scalar") {
            double value = 0.0;
            if (!parse_number(rhs, value)) {
                return std::unexpected(DomainError{"load", "invalid scalar: " + name});
            }
            state_.scalars[name] = value;
        } else if (kind == "matrix") {
            auto matrix = parse_matrix(rhs);
            if (!matrix) {
                return std::unexpected(matrix.error());
            }
            state_.matrices[name] = *matrix;
        }
    }
    if (have_plot) {
        pending_plot.valid = true;
        state_.plot = std::move(pending_plot);
    }
    return {};
}

Result<std::string> Interpreter::execute(const std::string& line) {
    const std::string cmd = trim(line);
    if (cmd.empty()) {
        return std::string{};
    }
    state_.history.push_back(cmd);

    const std::string lcmd = lower(cmd);
    if (lcmd == "help") {
        return std::string{
            "Commands:\n"
            "  help, vars, version, show, saveplot, clear, topology, simd, dispatch, balance, gpu, mpi, frameworks, exit\n"
            "  izaac seed <n>   gria(M)   axiom evolve\n"
            "  save <file.ms>  load <file.ms>\n"
            "  name = [1, 2; 3, 4]     matrix assignment\n"
            "  name = 3.14              scalar assignment\n"
            "  name = x + 2             scalar expression (+, -, *, /; () precedence)\n"
            "  name = sin(x) pow(x,2)   scalar calls (sin, cos, sqrt, pow, min, max, ...)\n"
            "  name = matmul(A, B)      matrix multiply assignment\n"
            "  name = solve(A, B)       linear solve assignment\n"
            "  name = transpose(A)      transpose assignment\n"
            "  name = chol(A)           Cholesky factor assignment\n"
            "  name = pinv(A) null(A) orth(A)  pseudo-inverse / null space / orthonormal basis\n"
            "  name = kron(A, B)        Kronecker product\n"
            "  name = repmat(A, p, q)   tile matrix p x q\n"
            "  name = linspace(a, b, n) equally spaced column vector\n"
            "  name = rgb2gray(M)       RGB rows (H*W)x3 to grayscale column vector\n"
            "  name = sobel(M)          Sobel gradient magnitude on HxW grayscale matrix\n"
            "  name = prewitt(M)        Prewitt edge magnitude on HxW grayscale matrix\n"
            "  name = count_components(B) connected component count in binary image B\n"
            "  name = imgaussfilt(M,s)  Gaussian blur on grayscale matrix\n"
            "  name = medfilt2(M,k)     median filter on grayscale matrix (odd k)\n"
            "  name = boxfilter(M,k)    box filter on grayscale matrix (odd k)\n"
            "  name = bilateral(M,sigma_s,sigma_r) bilateral filter on grayscale matrix\n"
            "  name = canny(M,low,high) Canny edge detection on grayscale matrix (sigma=1)\n"
            "  name = laplacian(M)      Laplacian edge filter on grayscale matrix\n"
            "  name = histeq(M)         histogram equalization on grayscale matrix\n"
            "  name = sharpen(M)        sharpen grayscale matrix\n"
            "  name = threshold_otsu(M) Otsu binary threshold on grayscale matrix\n"
            "  name = imresize(M,r,c)   resize grayscale matrix to r x c\n"
            "  name = rle_encode_vec(M) run-length encode flattened matrix bytes\n"
            "  name = rle_decode_vec(M) decode RLE byte vector to column vector\n"
            "  name = mtf_encode_vec(M) move-to-front encode flattened matrix bytes\n"
            "  name = mtf_decode_vec(M) decode MTF byte vector to column vector\n"
            "  name = lzw_encode_vec(M) LZW encode flattened matrix bytes to Nx1 code column\n"
            "  name = lzw_decode_vec(C) decode Nx1 LZW code column to byte column vector\n"
            "  name = lz77_encode_vec(M) LZ77 encode bytes to Nx3 token matrix [offset,length,next_char]\n"
            "  name = lz77_encode_vec(M,window,lookahead) LZ77 encode with custom window/lookahead\n"
            "  name = lz77_decode_vec(T) decode Nx3 LZ77 tokens to byte column vector\n"
            "  name = huffman_encode_vec(M) Huffman encode flattened matrix bytes to byte column\n"
            "  name = bzip2_compress_vec(M) bzip2-like compress bytes to byte column\n"
            "  name = compress_bits_to_bytes(bits_vec) pack Nx1 bit column into byte column\n"
            "  name = compress_bytes_to_bits(bytes_vec) unpack Nx1 byte column into bit column\n"
            "  name = bzip2_decompress_vec(C) bzip2-like decompress byte column (primary index in first 4 bytes)\n"
            "  name = huffman_decode_vec(orig_M,E) Huffman decode using orig size (re-encode internally)\n"
            "  name = bwt_encode_vec(M) Burrows-Wheeler encode flattened matrix bytes\n"
            "  name = bwt_primary_index(M) BWT primary index of flattened matrix bytes\n"
            "  name = bwt_decode_vec(L,primary_index) inverse BWT on L column and primary index\n"
            "  name = delta_encode_vec(M) delta-encode flattened matrix bytes\n"
            "  name = delta_decode_vec(M) decode delta byte vector to column vector\n"
            "  name = bigint(\"495\")     parse decimal string to scalar double\n"
            "  name = bigint_factorial(n) bigint factorial as scalar (if fits double)\n"
            "  name = bigint_fib(n)     bigint Fibonacci as scalar (if fits double)\n"
            "  name = bigint_gcd(\"a\",\"b\") bigint GCD as scalar (if fits double)\n"
            "  name = ml_accuracy(p,t)  ML accuracy on Nx1 prediction/label vectors\n"
            "  name = ml_rmse(p,t)      ML RMSE on Nx1 vectors\n"
            "  name = ml_mse(p,t)       ML MSE on Nx1 vectors\n"
            "  name = ml_r2(p,t)        ML R² on Nx1 vectors\n"
            "  name = ml_f1(p,t)        ML F1 on binary Nx1 vectors\n"
            "  name = ml_precision(p,t) ML precision on binary Nx1 vectors\n"
            "  name = ml_recall(p,t)    ML recall on binary Nx1 vectors\n"
            "  name = ml_mae(p,t)       ML MAE on Nx1 vectors\n"
            "  name = ml_huber(p,t)     ML Huber loss on Nx1 vectors (delta=1)\n"
            "  name = ml_hinge(p,t)     ML hinge loss on Nx1 vectors\n"
            "  name = ml_binary_crossentropy(p,t) ML binary cross-entropy on Nx1 vectors\n"
            "  name = ml_categorical_crossentropy(p,t) ML categorical cross-entropy on matrices\n"
            "  name = ml_mat_transpose(A) matrix transpose via ml::mat_T\n"
            "  name = ml_mat_mul(A,B) matrix product via ml::mat_mul\n"
            "  name = ml_vec_norm(v)    Euclidean norm of Nx1 or 1xN vector\n"
            "  name = ml_vec_dot(a,b)   dot product of Nx1 or 1xN vectors\n"
            "  name = graph_pagerank(A) PageRank scores from NxN adjacency matrix\n"
            "  name = graph_diameter(A) graph diameter from NxN adjacency matrix\n"
            "  name = graph_radius(A) graph radius from NxN adjacency matrix (undirected)\n"
            "  name = graph_betweenness(A) betweenness centrality column from adjacency\n"
            "  name = graph_closeness(A) closeness centrality column from adjacency\n"
            "  name = graph_degree_centrality(A) degree centrality column from adjacency\n"
            "  name = graph_max_flow(A,source,sink) max flow from source to sink\n"
            "  name = graph_is_bipartite(A) 1 if undirected graph is bipartite else 0\n"
            "  name = graph_is_connected(A) 1 if undirected graph is connected else 0\n"
            "  name = graph_is_tree(A) 1 if undirected graph is a tree else 0\n"
            "  name = graph_is_dag(A) 1 if directed graph is a DAG else 0\n"
            "  name = graph_topological_sort(A) topological order of DAG as Nx1 column\n"
            "  name = graph_greedy_colour(A) greedy vertex colours of undirected graph as Nx1 column\n"
            "  name = graph_euler_circuit(A) Euler circuit vertex order of undirected graph as Nx1 column\n"
            "  name = graph_floyd_warshall(A) all-pairs shortest-path distance matrix\n"
            "  name = graph_bfs(A,source) BFS visit order from source as Nx1 column\n"
            "  name = graph_dfs(A,source) DFS visit order from source as Nx1 column\n"
            "  name = graph_astar(A,source,target,h) A* path from source to target as Nx1 column\n"
            "  name = graph_mst_kruskal(A) minimum spanning tree as Mx3 [from,to,weight]\n"
            "  name = graph_mst_prim(A) minimum spanning tree (Prim) as Mx3 [from,to,weight]\n"
            "  name = graph_scc(A) strongly connected components as KxM vertex index matrix\n"
            "  name = imcrop(M,r0,c0,r1,c1) crop grayscale matrix to [r0:r1,c0:c1)\n"
            "  name = geo_dist2d(x1,y1,x2,y2) 2D Euclidean distance\n"
            "  name = geo_dist_sq2d(x1,y1,x2,y2) squared 2D Euclidean distance\n"
            "  name = geo_vec2d_length(x,y) 2D vector length\n"
            "  name = geo_cross2d(x1,y1,x2,y2) 2D scalar cross product\n"
            "  name = geo_dist3d(x1,y1,z1,x2,y2,z2) 3D Euclidean distance\n"
            "  name = geo_triangle_area(x1,y1,x2,y2,x3,y3) triangle area from three vertices\n"
            "  name = geo_convex_hull_area(P) convex hull area of Nx2 points\n"
            "  name = geo_polygon_area(P)     polygon area of Nx2 vertices\n"
            "  name = geo_polygon_perimeter(P) polygon perimeter of Nx2 vertices\n"
            "  name = geo_signed_area(P)      signed polygon area of Nx2 vertices\n"
            "  name = geo_moment_of_inertia(P) polygon moment of inertia about centroid\n"
            "  name = geo_centroid_x(P) polygon centroid x-coordinate of Nx2 vertices\n"
            "  name = geo_centroid_y(P) polygon centroid y-coordinate of Nx2 vertices\n"
            "  name = geo_dist_point_seg2d(px,py,x1,y1,x2,y2) point-to-segment distance in 2D\n"
            "  name = geo_dist_point_line2d(px,py,a,b,c) point-to-line distance in 2D\n"
            "  name = geo_volume_tetrahedron(x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4) tetrahedron volume\n"
            "  name = geo_bezier_eval_x(P,t) Bezier curve x-coordinate at parameter t\n"
            "  name = geo_bezier_eval_y(P,t) Bezier curve y-coordinate at parameter t\n"
            "  name = geo_hermite_x(p0x,p0y,m0x,m0y,p1x,p1y,m1x,m1y,t) Hermite curve x at t\n"
            "  name = geo_point_in_polygon(px,py,P) 1 if point inside Nx2 polygon else 0\n"
            "  name = geo_delaunay_2d(P) Delaunay triangulation Mx3 index matrix from Nx2 points\n"
            "  name = geo_voronoi(P) Voronoi vertex coordinates Mx2 from Nx2 points\n"
            "  name = geo_kdtree_nearest(P,x,y) nearest point index in Nx2 set to query (x,y)\n"
            "  name = topo_pairwise_distances(P) NxN pairwise distance matrix from Nx2 points\n"
            "  name = geo_overlap_circles(x1,y1,r1,x2,y2,r2) 1 if circles overlap else 0\n"
            "  name = combo_nchoosek(n,k) binomial coefficient C(n,k)\n"
            "  name = combo_stirling1(n,k) unsigned Stirling number of first kind |s(n,k)|\n"
            "  name = combo_stirling2(n,k) Stirling number of second kind S(n,k)\n"
            "  name = combo_factorial(n)  integer factorial n!\n"
            "  name = combo_catalan(n)    Catalan number C_n\n"
            "  name = combo_bell(n)       Bell number B_n\n"
            "  name = combo_motzkin(n)    Motzkin number M_n\n"
            "  name = combo_permutations(n,k) permutations P(n,k)\n"
            "  name = combo_subfactorial(n) derangement count D(n)\n"
            "  name = combo_double_factorial(n) double factorial n!!\n"
            "  name = combo_combinations_with_rep(n,k) combinations with repetition C(n+k-1,k)\n"
            "  name = combo_multinomial(n,ks) multinomial coefficient n!/(k1! k2! ...)\n"
            "  name = combo_rank_permutation(v) rank of permutation vector v\n"
            "  name = combo_next_perm(v) lexicographic next permutation as Nx1 column\n"
            "  name = combo_next_comb(v,n) lexicographic next k-combination in n\n"
            "  name = combo_rank_combination(v,n) rank of k-combination vector v in n\n"
            "  name = combo_derangements(n) all derangements of 0..n-1 as k×n int matrix\n"
            "  name = combo_all_permutations(n) all permutations of 0..n-1 as n!×n int matrix\n"
            "  name = combo_all_subsets(n) all subsets of {0..n-1} as 2^n×n int matrix\n"
            "  name = combo_all_compositions(n) all compositions of n as r×k int matrix (zero-padded)\n"
            "  name = combo_all_partitions(n) all partitions of n as r×k int matrix (zero-padded)\n"
            "  name = combo_unrank_permutation(n,rank) unrank permutation of n elements\n"
            "  name = combo_unrank_combination(n,k,rank) unrank k-combination of n elements\n"
            "  name = numthy_gcd(a,b)   integer GCD\n"
            "  name = numthy_lcm(a,b)   integer LCM\n"
            "  name = numthy_mod_pow(base,exp,mod) modular exponentiation base^exp mod mod\n"
            "  name = numthy_partition(n) integer partition count p(n)\n"
            "  name = numthy_num_divisors(n) divisor count tau(n)\n"
            "  name = numthy_factor_count(n) prime factor count with multiplicity\n"
            "  name = numthy_sum_divisors(n) sum of divisors sigma(n)\n"
            "  name = numthy_isprime(n)   1 if n is prime else 0\n"
            "  name = numthy_euler_phi(n) Euler totient phi(n)\n"
            "  name = numthy_mobius(n)    Möbius function mu(n): -1, 0, or 1\n"
            "  name = numthy_nextprime(n) smallest prime strictly greater than n\n"
            "  name = numthy_prevprime(n) largest prime strictly less than n (0 if none)\n"
            "  name = numthy_liouville(n) Liouville function lambda(n): -1 or 1\n"
            "  name = numthy_prime_pi(n)  prime counting function pi(n)\n"
            "  name = numthy_prime_nth(n) nth prime (1-indexed)\n"
            "  name = numthy_legendre_symbol(a,p) Legendre symbol (a/p), p odd prime\n"
            "  name = numthy_kronecker_symbol(a,n) Kronecker symbol (a/n)\n"
            "  name = numthy_tonelli_shanks(n,p) modular square root x with x^2 ≡ n (mod p)\n"
            "  name = numthy_mod_inv(a,m) modular inverse a^(-1) mod m\n"
            "  name = numthy_extended_gcd(a,b) extended GCD gcd(a,b)\n"
            "  name = numthy_crt(r,m) Chinese remainder theorem on remainder/modulus vectors\n"
            "  name = numthy_divisors_vec(n) sorted divisors of n as Nx1 column\n"
            "  name = numthy_continued_fraction(x,n) continued fraction coefficients as nx1 column\n"
            "  name = numthy_primes(lo,hi) primes in inclusive range as Kx1 column\n"
            "  name = numthy_convergents(cf) convergent pairs p/q as Kx2 matrix from cf column\n"
            "  name = numthy_factor_vec(n) prime factorization of n as kx1 column\n"
            "  name = poly_deriv(coeffs) polynomial derivative coefficient column (low-to-high)\n"
            "  name = poly_add(a,b) polynomial sum coefficient column (low-to-high)\n"
            "  name = poly_mul(a,b) polynomial product coefficient column (low-to-high)\n"
            "  name = poly_sub(a,b) polynomial difference coefficient column (low-to-high)\n"
            "  name = poly_compose(p,q) polynomial composition p(q(x)) coefficient column\n"
            "  name = poly_eval(coeffs,x) polynomial evaluation at scalar x\n"
            "  name = poly_integ(coeffs,c) polynomial integral with constant c\n"
            "  name = stats_correlation(x,y) Pearson correlation of Nx1 vectors\n"
            "  name = stats_spearman(x,y) Spearman rank correlation of Nx1 vectors\n"
            "  name = stats_kendall(x,y) Kendall tau rank correlation of Nx1 vectors\n"
            "  name = stats_mean(x) arithmetic mean of Nx1 vector\n"
            "  name = stats_median(x) median of Nx1 vector\n"
            "  name = stats_stddev(x) sample standard deviation of Nx1 vector\n"
            "  name = stats_skewness(x) skewness of Nx1 vector\n"
            "  name = stats_kurtosis(x) kurtosis of Nx1 vector\n"
            "  name = stats_var(x) sample variance of Nx1 vector\n"
            "  name = stats_percentile(x,p) p-th percentile of Nx1 vector\n"
            "  name = stats_mode(x) mode of Nx1 vector\n"
            "  name = stats_geometric_mean(x) geometric mean of positive Nx1 vector\n"
            "  name = stats_harmonic_mean(x) harmonic mean of positive Nx1 vector\n"
            "  name = stats_rms(x) root-mean-square of Nx1 vector\n"
            "  name = stats_mad(x) median absolute deviation of Nx1 vector\n"
            "  name = stats_iqr(x) interquartile range of Nx1 vector\n"
            "  name = stats_min_value(x) minimum of Nx1 vector\n"
            "  name = stats_ttest(x,mu) one-sample t-test statistic vs hypothesized mean mu\n"
            "  name = stats_ztest(x,mu,sigma) one-sample z-test statistic vs mean mu with known sigma\n"
            "  name = stats_acf(x,max_lag) autocorrelation function as (max_lag+1)x1 column\n"
            "  name = stats_two_sample_ttest(a,b) two-sample t-test statistic for Nx1 vectors\n"
            "  name = stats_chi2_gof(observed,expected) chi-squared goodness-of-fit statistic\n"
            "  name = signal_moving_average(x,window) moving average of Nx1 signal\n"
            "  name = signal_lowpass(x,cutoff,fs) lowpass filter of Nx1 signal\n"
            "  name = signal_butterworth(x,cutoff,fs) Butterworth lowpass filter of Nx1 signal\n"
            "  name = signal_highpass(x,cutoff,fs) highpass filter of Nx1 signal\n"
            "  name = signal_bandpass(x,low,high,fs) bandpass filter of Nx1 signal\n"
            "  name = signal_convolve(a,b) discrete convolution of Nx1 vectors\n"
            "  name = signal_correlate(a,b) cross-correlation of Nx1 vectors\n"
            "  name = signal_hamming(n) Hamming window as n×1 column\n"
            "  name = signal_hanning(n) Hanning window as n×1 column\n"
            "  name = signal_blackman(n) Blackman window as n×1 column\n"
            "  name = signal_parzen(n) Parzen window as n×1 column\n"
            "  name = signal_triangular(n) triangular window as n×1 column\n"
            "  name = pde_heat_1d(x0,alpha,dx,dt,steps) 1D heat equation final state column\n"
            "  name = pde_heat_2d(u0,alpha,dx,dy,dt,steps) 2D heat equation final grid (rows×cols)\n"
            "  name = pde_wave_1d(u0,v0,c,dx,dt,steps) 1D wave equation final state column\n"
            "  name = pde_advection_1d(u0,v,dx,dt,steps) 1D advection final state column\n"
            "  name = pde_poisson_2d(f,dx,dy,max_iterations,tolerance) 2D Poisson solution grid\n"
            "  name = pde_burgers_1d(u0,nu,dx,dt,steps) viscous Burgers final state column\n"
            "  name = sym_diff(\"expr\",\"var\") differentiate quoted expression w.r.t. variable\n"
            "  name = sym_simplify(\"expr\") simplify quoted symbolic expression\n"
            "  name = sym_integrate(\"expr\",\"var\") integrate quoted expression w.r.t. variable\n"
            "  name = sym_eval(\"expr\",\"var=value\") numerically evaluate quoted expression\n"
            "  name = ode_euler(\"formula\",t0,y0,t_end,steps) Euler IVP trajectory [t,y] columns\n"
            "  name = ode_rk4(\"formula\",t0,y0,t_end,steps) RK4 IVP trajectory [t,y] columns\n"
            "  name = ode_midpoint(\"formula\",t0,y0,t_end,steps) midpoint IVP trajectory [t,y] columns\n"
            "  name = ode_rk45(\"formula\",t0,y0,t_end,rtol,atol) adaptive RK45 IVP trajectory [t,y] columns\n"
            "  name = ode_backward_euler(\"formula\",t0,y0,t_end,steps) backward Euler IVP trajectory [t,y] columns\n"
            "  name = fft_rfft(x) real FFT spectrum as Nx2 [re,im] matrix\n"
            "  name = fftshift(S) cyclic shift of Nx2 complex spectrum\n"
            "  name = fft_dft(x) discrete Fourier transform as Nx2 [re,im] matrix\n"
            "  name = fft_irfft(spectrum,n) inverse real FFT from Nx2 spectrum to n×1 column\n"
            "  name = fft_ifft(spectrum) inverse FFT from Nx2 spectrum to Nx1 column\n"
            "  name = fft_fft2(S) 2D FFT from Nx2 complex spectrum to Nx2 spectrum\n"
            "  name = fft_dct2(x) type-II DCT coefficients as Nx1 column\n"
            "  name = fft_idct2(x) type-II inverse DCT signal as Nx1 column\n"
            "  name = fft_dst2(x) type-II DST coefficients as Nx1 column\n"
            "  name = numthy_is_primitive_root(g,p) 1 if g is primitive root mod p else 0\n"
            "  name = numthy_primitive_root(p) smallest primitive root mod prime p\n"
            "  name = numthy_discrete_log(g,h,p) discrete log x with g^x ≡ h (mod p)\n"
            "  name = numthy_jacobi_symbol(a,n) Jacobi symbol (a/n), n positive odd\n"
            "  name = control_step_final(num,den) final step-response sample\n"
            "  name = control_impulse_final(num,den) final impulse-response sample\n"
            "  name = control_dcgain(num,den) DC gain of transfer function\n"
            "  name = control_is_stable(num,den) 1 if all poles in LHP else 0\n"
            "  name = control_is_controllable(A,B) 1 if (A,B) controllable else 0\n"
            "  name = control_is_observable(A,C) 1 if (A,C) observable else 0\n"
            "  name = control_lyap(A,Q) continuous Lyapunov equation solution matrix\n"
            "  name = control_dlyap(A,Q) discrete Lyapunov equation solution matrix\n"
            "  name = control_lqr(A,B,Q,R) LQR state-feedback gain matrix K\n"
            "  name = control_riccati(A,B,Q,R) continuous Riccati equation solution X\n"
            "  name = control_dare(A,B,Q,R) discrete Riccati equation solution X\n"
            "  name = control_bode_mag_db(num,den,w) Bode magnitude (dB) at nearest grid point\n"
            "  name = control_bode_phase(num,den,w) Bode phase (deg) at nearest grid point\n"
            "  name = control_bode(num,den,w) Bode [mag_db, phase_deg] at nearest grid point\n"
            "  name = control_phase_margin(num,den) phase margin (deg) of transfer function\n"
            "  name = control_gain_margin(num,den) gain margin (dB) of transfer function\n"
            "  name = control_margins(num,den) gain/phase margins [gain_db, phase_deg]\n"
            "  name = control_place(A,B,poles) pole-placement gain column K\n"
            "  name = control_pidtune_kp(num,den) PID Kp from pidtune(plant,1.0)\n"
            "  name = control_pidtune_ki(num,den) PID Ki from pidtune(plant,1.0)\n"
            "  name = control_pidtune_kd(num,den) PID Kd from pidtune(plant,1.0)\n"
            "  name = quantum_hadamard(psi) apply Hadamard gate to 2x1 state\n"
            "  name = quantum_density_matrix(psi) density matrix rho=|psi><psi| (real parts)\n"
            "  name = quantum_ket_normalise(psi) normalise Nx1 state vector to unit length\n"
            "  name = quantum_ket_superposition(amps) superposition ket from amplitude vector\n"
            "  name = quantum_ket_basis(dim,index) basis ket |index> in dim-dimensional space\n"
            "  name = quantum_fock_state(n,n_max) Fock state |n> truncated at n_max\n"
            "  name = quantum_coherent_state(alpha_re,alpha_im,n_max) coherent state column vector\n"
            "  name = quantum_pauli_x()   2x2 Pauli-X gate matrix\n"
            "  name = quantum_pauli_y()   2x2 Pauli-Y gate matrix\n"
            "  name = quantum_pauli_z()   2x2 Pauli-Z gate matrix\n"
            "  name = quantum_pauli_plus()  2x2 |+><+| projector matrix\n"
            "  name = quantum_pauli_minus() 2x2 |-><-| projector matrix\n"
            "  name = quantum_cnot_gate() 4x4 CNOT gate matrix\n"
            "  name = quantum_swap_gate() 4x4 SWAP gate matrix\n"
            "  name = quantum_toffoli_gate() 8x8 Toffoli gate matrix\n"
            "  name = quantum_identity()  2x2 identity gate matrix\n"
            "  name = quantum_identity_n(dim) NxN identity gate matrix\n"
            "  name = quantum_ghz_state(n) n-qubit GHZ state column vector\n"
            "  name = quantum_w_state(n) n-qubit W state column vector\n"
            "  name = quantum_bell_state(index) Bell state ket column (index 0..3)\n"
            "  name = quantum_hadamard_gate() 2x2 Hadamard gate matrix\n"
            "  name = quantum_op_apply(op,psi) apply NxN gate matrix to Nx1 state (real parts)\n"
            "  name = quantum_rotation_z(theta) 2x2 Z-rotation gate matrix\n"
            "  name = quantum_rotation_x(theta) 2x2 X-rotation gate matrix\n"
            "  name = quantum_rotation_y(theta) 2x2 Y-rotation gate matrix\n"
            "  name = quantum_phase_gate(theta) 2x2 phase gate matrix\n"
            "  name = quantum_qft_gate(n_qubits) 2^n x 2^n QFT gate matrix\n"
            "  name = finance_bs_call(S,K,T,r,sigma) Black-Scholes call price\n"
            "  name = finance_bs_put(S,K,T,r,sigma) Black-Scholes put price\n"
            "  name = finance_bs_gamma(S,K,T,r,sigma) Black-Scholes gamma\n"
            "  name = prob_norm_cdf(x,mu,sigma) normal CDF at x with mean mu and std dev sigma\n"
            "  name = prob_norm_pdf(x,mu,sigma) normal PDF at x with mean mu and std dev sigma\n"
            "  name = prob_norm_ppf(p,mu,sigma) normal quantile at p with mean mu and std dev sigma\n"
            "  name = prob_binom_pdf(k,n,p) binomial PMF at k with n trials and success prob p\n"
            "  name = prob_binom_cdf(k,n,p) binomial CDF at k with n trials and success prob p\n"
            "  name = prob_pois_pdf(k,lambda) Poisson PMF at k with rate lambda\n"
            "  name = prob_pois_cdf(k,lambda) Poisson CDF at k with rate lambda\n"
            "  name = prob_uniform_cdf(x,a,b) uniform CDF on [a,b] at x\n"
            "  name = prob_exp_cdf(x,lambda) exponential CDF at x with rate lambda\n"
            "  name = prob_exp_pdf(x,lambda) exponential PDF at x with rate lambda\n"
            "  name = prob_chi2_cdf(x,df) chi-squared CDF at x with df degrees of freedom\n"
            "  name = prob_chi2_pdf(x,df) chi-squared PDF at x with df degrees of freedom\n"
            "  name = prob_t_cdf(x,df) Student t CDF at x with df degrees of freedom\n"
            "  name = prob_gamma_pdf(x,shape,scale) gamma PDF at x with shape and scale\n"
            "  name = finance_bs_vega(S,K,T,r,sigma) Black-Scholes vega\n"
            "  name = finance_bs_delta(S,K,T,r,sigma,call) Black-Scholes delta (call: 0=put, 1=call)\n"
            "  name = finance_bs_implied_vol(price,S,K,T,r,call) implied volatility from option price (call: 0=put, 1=call)\n"
            "  name = finance_bs_theta(S,K,T,r,sigma,call) Black-Scholes theta (call: 0=put, 1=call)\n"
            "  name = finance_bs_rho(S,K,T,r,sigma,call) Black-Scholes rho (call: 0=put, 1=call)\n"
            "  name = finance_binomial_call(S,K,T,r,sigma,steps) binomial-tree call price\n"
            "  name = finance_binomial_put(S,K,T,r,sigma,steps) binomial-tree put price\n"
            "  name = finance_bond_price(c,y,n,fv) bond price (annual coupon c, yield y, n periods, face fv)\n"
            "  name = finance_bond_duration(c,y,n) Macaulay duration (annual coupon c, yield y, n periods)\n"
            "  name = finance_bond_modified_duration(c,y,n) modified duration (annual coupon c, yield y, n periods)\n"
            "  name = finance_bond_convexity(c,y,n) bond convexity (annual coupon c, yield y, n periods)\n"
            "  name = finance_bond_ytm(price,c,n) yield to maturity from bond price (annual coupon c, n periods)\n"
            "  name = finance_compound(principal,rate,n_periods,compounds_per_period) compound interest amount\n"
            "  name = finance_continuous_compound(principal,rate,t) continuously compounded future value\n"
            "  name = finance_pv(rate,n,pmt,fv) present value of annuity (fv optional, default 0)\n"
            "  name = finance_fv_annuity(rate,n,pmt,pv0) future value of annuity (pv0 optional, default 0)\n"
            "  name = finance_pmt_annuity(rate,n,pv0,fv) periodic payment for annuity (fv optional, default 0)\n"
            "  name = finance_npv(rate,cf) net present value of cashflow vector\n"
            "  name = finance_irr(cf) internal rate of return of cashflow vector\n"
            "  name = finance_sharpe(r) Sharpe ratio from Nx1 return vector\n"
            "  name = finance_sortino(r) Sortino ratio from Nx1 return vector\n"
            "  name = finance_var(r) 95% Value-at-Risk from Nx1 return vector\n"
            "  name = finance_cvar(r) 95% Conditional VaR (Expected Shortfall) from Nx1 returns\n"
            "  name = finance_max_drawdown(equity) maximum drawdown from Nx1 equity curve\n"
            "  name = finance_kelly_fraction(p,b) Kelly criterion optimal bet fraction\n"
            "  name = finance_portfolio_return(weights,returns) portfolio return from Nx1 weights and returns\n"
            "  name = finance_portfolio_variance(weights,cov) portfolio variance from Nx1 weights and NxN covariance\n"
            "  name = info_shannon_hartley(bandwidth_hz,snr_linear) Shannon-Hartley channel capacity in bps\n"
            "  name = quantum_von_neumann_entropy(rho) von Neumann entropy of NxN density matrix\n"
            "  name = quantum_concurrence(rho) concurrence of 4x4 two-qubit density matrix\n"
            "  name = quantum_fidelity(rho,sigma) quantum fidelity between NxN density matrices\n"
            "  name = quantum_expectation_dm(rho,op) Tr(rho op) for NxN density matrices\n"
            "  name = quantum_expectation(psi,A) expectation value <psi|A|psi> for Nx1 state and NxN operator\n"
            "  name = quantum_inner(bra,ket) inner product Re(<bra|ket>) for Nx1 state vectors\n"
            "  name = quantum_trace_distance(rho,sigma) trace distance between NxN density matrices\n"
            "  name = quantum_entanglement_entropy(psi,dim_a,dim_b) entanglement entropy of Nx1 state\n"
            "  name = quantum_partial_trace(rho,d1,d2,subsystem) partial trace of NxN density matrix\n"
            "  name = quantum_schrodinger(H,psi0,t0,t1,n_steps) Schrödinger trajectory (real parts)\n"
            "  name = quantum_schrodinger_final(H,psi0,t0,t1,n_steps) final Schrödinger state column\n"
            "  name = quantum_time_evolution(H,t) time-evolution operator U(t) (real parts)\n"
            "  name = quantum_commutator(A,B) commutator [A,B] of NxN operators\n"
            "  name = info_entropy(p) Shannon entropy (bits) of Nx1 probability vector\n"
            "  name = info_redundancy(p) source coding redundancy H_max - H(X) in bits\n"
            "  name = info_efficiency(p) coding efficiency H(X)/H_max of Nx1 probability vector\n"
            "  name = info_mutual_info(joint) mutual information I(X;Y) from joint PMF matrix\n"
            "  name = info_joint_entropy(joint,rows,cols) joint entropy H(X,Y) from joint PMF matrix\n"
            "  name = info_conditional_entropy(joint,rows,cols) conditional entropy H(Y|X) from joint PMF\n"
            "  name = info_sample_entropy(x,m,r) sample entropy of Nx1 time series (template m, tolerance r)\n"
            "  name = info_lz_complexity(seq) Lempel-Ziv complexity of Nx1 integer sequence\n"
            "  name = info_kl_divergence(p,q) KL divergence D_KL(P||Q) in bits\n"
            "  name = info_cross_entropy(p,q) cross-entropy H(P,Q) in bits\n"
            "  name = info_js_divergence(p,q) Jensen-Shannon divergence in bits\n"
            "  name = info_tv_distance(p,q) total variation distance between Nx1 PMFs\n"
            "  name = info_hellinger_dist(p,q) Hellinger distance between Nx1 PMFs\n"
            "  name = info_renyi_entropy(alpha,p) Renyi entropy of order alpha (bits)\n"
            "  name = info_source_coding_rate(p) Shannon source coding rate H(X) in bits\n"
            "  name = info_tsallis_entropy(q,p) Tsallis entropy of order q for Nx1 probability vector\n"
            "  name = info_channel_capacity_bsc(p_error) BSC channel capacity in bits\n"
            "  name = info_channel_capacity_bec(epsilon) BEC erasure channel capacity in bits\n"
            "  name = info_differential_entropy_gaussian(sigma) Gaussian differential entropy (nats)\n"
            "  name = info_differential_entropy_uniform(a,b) uniform differential entropy (nats)\n"
            "  name = info_rate_distortion_gaussian(variance,distortion) Gaussian rate-distortion R(D) in bits\n"
            "  name = cplx_joukowski(re,im) Joukowski transform magnitude\n"
            "  name = cplx_joukowski_inv(re,im) |z| of Joukowski inverse matching forward point\n"
            "  name = cplx_hyperbolic_distance(z1re,z1im,z2re,z2im) Poincaré disk hyperbolic distance\n"
            "  name = cplx_mobius_re(a,b,c,d,zre,zim) real part of Möbius transform (az+b)/(cz+d)\n"
            "  name = cplx_poisson_kernel(theta,phi,r) Poisson kernel on unit disk\n"
            "  name = cplx_cross_ratio(z1re,z1im,z2re,z2im,z3re,z3im,z4re,z4im) cross ratio\n"
            "  name = cplx_power_series_eval(coeffs,zre,zim) Taylor series at z0=0\n"
            "  name = cplx_winding_number(G,z0re,z0im) winding number of closed polygon G\n"
            "  name = cplx_residue_inv(pole_re,pole_im) residue of 1/(z-pole) at pole\n"
            "  name = cplx_contour_integral_oneoverz_im() Im part of unit-circle integral of 1/z\n"
            "  name = cplx_line_integral_one() Re of line integral of f=1 from 0 to 1\n"
            "  name = cplx_blaschke_product(zre,zim,zeros) |B(z)| for zeros Nx2 [re,im] matrix\n"
            "  name = tensorops_norm(T) Frobenius norm of matrix/tensor slice\n"
            "  name = tensorops_inner(A,B) Frobenius inner product of two tensors\n"
            "  name = tensorops_matmul(A,B) 2D matrix multiply via tensor einsum\n"
            "  name = tensorops_einsum(A,B) 2D einsum ij,jk->ik (same as matmul)\n"
            "  name = diffgeo_gaussian_sphere() unit-sphere Gaussian curvature at (0.5,0.5)\n"
            "  name = diffgeo_gaussian_curvature_sphere(u,v) unit-sphere Gaussian curvature at (u,v)\n"
            "  name = diffgeo_mean_curvature_sphere(u,v) unit-sphere mean curvature at (u,v)\n"
            "  name = diffgeo_ricci_scalar_sphere(u,v) unit-sphere Ricci scalar at (u,v)\n"
            "  name = diffgeo_einstein_scalar_sphere(u,v) contracted Einstein tensor on unit sphere\n"
            "  name = diffgeo_surface_normal_sphere(u,v) unit-sphere surface normal 3x1 column\n"
            "  name = diffgeo_mean_sphere()     unit-sphere mean curvature at (0.5,0.5)\n"
            "  name = diffgeo_principal_curvature_sphere() unit-sphere principal curvature k1 at (0.3,0.7)\n"
            "  name = topo_euler_tetrahedron()  Euler characteristic of solid tetrahedron\n"
            "  name = topo_euler_sphere_surface() Euler characteristic of tetrahedron surface (chi=2)\n"
            "  name = topo_vietoris_rips_betti0(D,r,max_dim) Vietoris-Rips beta0 from distance matrix\n"
            "  name = topo_betti_curve(D,thresholds,max_dim) Betti numbers per threshold row\n"
            "  name = topo_bottleneck_distance(dgm1,dgm2,dim) bottleneck distance between diagrams\n"
            "  name = topo_wasserstein_distance(dgm1,dgm2,dim) Wasserstein distance between diagrams\n"
            "  name = topo_persistence_diagram(S,births) persistence diagram from simplex filtration\n"
            "  name = diffgeo_geodesic_euclidean(x0,y0,vx,vy,s_end) flat-space geodesic trajectory\n"
            "  name = diffgeo_christoffel_sphere(k,i,j,u,v) unit-sphere Christoffel symbol\n"
            "  name = topo_euler_tetrahedron()  Euler characteristic of solid tetrahedron (=1)\n"
            "  name = det(A)            scalar matrix call (det, trace, norm, rank, cond)\n"
            "  L, U, P = lu(A)          LU factors (L, U only also supported)\n"
            "  Q, R = qr(A)             QR factors\n"
            "  U, S, V = svd(A)         SVD factors (U, S only also supported)\n"
            "  D, V = eig_sym(A)        symmetric eigenvalues and eigenvectors\n"
            "  plot([y...])  plot([x...], [y...])  scatter([x...], [y...])  hist([...])\n"
            "  imshow(matrix)  spy(matrix)  surf(matrix)\n"
            "  surf([x...], [y...], [z...])  show  saveplot <file.txt>\n"
            "  det(A), trace(A), norm(A), rank(A), cond(A)\n"
            "  lu(A), qr(A), chol(A), solve(A,B), matmul(A,B), tensorops_matmul(A,B), tensorops_einsum(A,B), eig_sym(A), svd(A)\n"
            "  pinv(A), null(A), orth(A), kron(A,B), repmat(A,p,q), linspace(a,b,n)\n"
            "  rgb2gray(M), sobel(M), imgaussfilt(M,s), medfilt2(M,k), boxfilter(M,k), bilateral(M,sigma_s,sigma_r), canny(M,low,high), laplacian(M), histeq(M), sharpen(M)\n"
            "  threshold_otsu(M), imresize(M,r,c), imcrop(M,r0,c0,r1,c1), rle_encode_vec(M), rle_decode_vec(M), mtf_encode_vec(M), mtf_decode_vec(M), lzw_encode_vec(M), lzw_decode_vec(C), lz77_encode_vec(M), lz77_decode_vec(T), huffman_encode_vec(M), huffman_decode_vec(orig_M,E), bzip2_compress_vec(M), bzip2_decompress_vec(C), compress_bits_to_bytes(bits_vec), compress_bytes_to_bits(bytes_vec), bwt_encode_vec(M), bwt_decode_vec(L,pi)\n"
            "  delta_encode_vec(M), delta_decode_vec(M)\n"
            "  ml_accuracy(p,t), ml_rmse(p,t), ml_mse(p,t), ml_r2(p,t), ml_f1(p,t), ml_precision(p,t), ml_recall(p,t), ml_mae(p,t), ml_huber(p,t), ml_hinge(p,t), ml_binary_crossentropy(p,t), ml_categorical_crossentropy(p,t), ml_mat_transpose(A), ml_vec_norm(v), ml_vec_dot(a,b)\n"
            "  bigint_factorial(n), bigint_fib(n), bigint_gcd(\"a\",\"b\")\n"
            "  graph_pagerank(A), graph_dijkstra_dist(A,s,t), graph_bellman_ford_dist(A,s,t), graph_bfs(A,source), graph_dfs(A,source), graph_astar(A,source,target,h), graph_max_flow(A,source,sink), graph_diameter(A), graph_radius(A), graph_betweenness(A), graph_closeness(A), graph_degree_centrality(A), graph_is_bipartite(A), graph_is_connected(A), graph_is_tree(A), graph_is_dag(A), graph_topological_sort(A), graph_greedy_colour(A), graph_euler_circuit(A), graph_floyd_warshall(A), graph_mst_kruskal(A), graph_mst_prim(A)\n"
            "  geo_dist2d(x1,y1,x2,y2), geo_dist_sq2d(x1,y1,x2,y2), geo_vec2d_length(x,y), geo_cross2d(x1,y1,x2,y2), geo_dist3d(x1,y1,z1,x2,y2,z2), geo_dist_point_seg2d(px,py,x1,y1,x2,y2), geo_dist_point_line2d(px,py,a,b,c), geo_volume_tetrahedron(x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4), geo_triangle_area(x1,y1,x2,y2,x3,y3), geo_overlap_circles(x1,y1,r1,x2,y2,r2), geo_convex_hull_area(P), geo_polygon_area(P), geo_polygon_perimeter(P), geo_signed_area(P), geo_moment_of_inertia(P), geo_point_in_polygon(px,py,P), geo_delaunay_2d(P), geo_voronoi(P), geo_kdtree_nearest(P,x,y), topo_pairwise_distances(P), geo_bezier_eval_x(P,t), geo_bezier_eval_y(P,t), geo_centroid_x(P), geo_centroid_y(P), bwt_primary_index(M)\n"
            "  combo_nchoosek(n,k), combo_stirling1(n,k), combo_stirling2(n,k), combo_permutations(n,k), combo_combinations_with_rep(n,k), combo_multinomial(n,ks), combo_rank_permutation(v), combo_next_perm(v), combo_rank_combination(v,n), combo_unrank_permutation(n,rank), combo_unrank_combination(n,k,rank), combo_derangements(n), combo_all_permutations(n), combo_all_subsets(n), combo_all_compositions(n), combo_all_partitions(n), combo_factorial(n), combo_catalan(n), combo_bell(n), combo_motzkin(n), combo_subfactorial(n), combo_double_factorial(n), numthy_gcd(a,b), numthy_lcm(a,b), numthy_mod_pow(base,exp,mod), numthy_partition(n), numthy_num_divisors(n), numthy_factor_count(n), numthy_sum_divisors(n), numthy_divisors_vec(n), numthy_continued_fraction(x,n), numthy_convergents(cf), numthy_factor_vec(n), numthy_isprime(n), numthy_euler_phi(n), numthy_mobius(n), numthy_nextprime(n), numthy_prevprime(n), numthy_liouville(n), numthy_prime_pi(n), numthy_prime_nth(n), numthy_legendre_symbol(a,p), numthy_jacobi_symbol(a,n), numthy_kronecker_symbol(a,n), numthy_tonelli_shanks(n,p), numthy_mod_inv(a,m), numthy_is_primitive_root(g,p), numthy_primitive_root(p), numthy_discrete_log(g,h,p)\n"
            "  control_step_final(num,den), control_impulse_final(num,den), control_dcgain(num,den), control_is_stable(num,den), control_lyap(A,Q), control_dlyap(A,Q), control_lqr(A,B,Q,R), control_riccati(A,B,Q,R), control_dare(A,B,Q,R), control_bode_mag_db(num,den,w), control_bode_phase(num,den,w), control_bode(num,den,w), control_phase_margin(num,den), control_gain_margin(num,den), control_margins(num,den), control_place(A,B,poles), control_pidtune_kp(num,den), control_pidtune_ki(num,den), control_pidtune_kd(num,den)\n"
            "  quantum_hadamard(psi), quantum_op_apply(op,psi), quantum_ket_normalise(psi), quantum_density_matrix(psi), quantum_ket_superposition(amps), quantum_ket_basis(dim,index), quantum_fock_state(n,n_max), quantum_coherent_state(alpha_re,alpha_im,n_max), quantum_pauli_x(), quantum_pauli_y(), quantum_pauli_z(), quantum_pauli_plus(), quantum_pauli_minus(), quantum_cnot_gate(), quantum_swap_gate(), quantum_toffoli_gate(), quantum_identity(), quantum_identity_n(dim), quantum_ghz_state(n), quantum_w_state(n), quantum_bell_state(index), quantum_hadamard_gate(), quantum_rotation_z(theta), quantum_rotation_x(theta), quantum_rotation_y(theta), quantum_phase_gate(theta), quantum_qft_gate(n_qubits)\n"
            "  control_is_controllable(A,B), control_is_observable(A,C), numthy_extended_gcd(a,b), numthy_crt(r,m)\n"
            "  finance_bs_call(S,K,T,r,sigma), finance_bs_put(S,K,T,r,sigma), finance_bs_gamma(S,K,T,r,sigma), finance_bs_vega(S,K,T,r,sigma), finance_bs_delta(S,K,T,r,sigma,call), finance_bs_implied_vol(price,S,K,T,r,call), finance_bs_theta(S,K,T,r,sigma,call), finance_bs_rho(S,K,T,r,sigma,call), finance_binomial_call(S,K,T,r,sigma,steps), finance_binomial_put(S,K,T,r,sigma,steps), finance_bond_price(c,y,n,fv), finance_bond_duration(c,y,n), finance_bond_modified_duration(c,y,n), finance_bond_convexity(c,y,n), finance_bond_ytm(price,c,n), finance_compound(principal,rate,n_periods,compounds_per_period), finance_continuous_compound(principal,rate,t), finance_pv(rate,n,pmt,fv), finance_fv_annuity(rate,n,pmt,pv0), finance_pmt_annuity(rate,n,pv0,fv), finance_npv(rate,cf), finance_irr(cf), finance_sharpe(r), finance_sortino(r), finance_var(r), finance_cvar(r), finance_max_drawdown(equity), finance_kelly_fraction(p,b), finance_portfolio_return(weights,returns), finance_portfolio_variance(weights,cov)\n"
            "  quantum_von_neumann_entropy(rho), quantum_concurrence(rho), quantum_fidelity(rho,sigma), quantum_commutator(A,B), quantum_tensor_product(A,B), quantum_expectation_dm(rho,op), quantum_expectation(psi,A), quantum_inner(bra,ket), quantum_trace_distance(rho,sigma), quantum_entanglement_entropy(psi,dim_a,dim_b), quantum_partial_trace(rho,d1,d2,subsystem), quantum_schrodinger(H,psi0,t0,t1,n_steps), quantum_schrodinger_final(H,psi0,t0,t1,n_steps), quantum_time_evolution(H,t)\n"
            "  info_entropy(p), info_mutual_info(joint), info_joint_entropy(joint,rows,cols), info_conditional_entropy(joint,rows,cols), info_sample_entropy(x,m,r), info_lz_complexity(seq), info_redundancy(p), info_efficiency(p), info_source_coding_rate(p), info_kl_divergence(p,q), info_js_divergence(p,q), info_cross_entropy(p,q), info_tv_distance(p,q), info_hellinger_dist(p,q), info_renyi_entropy(alpha,p), info_tsallis_entropy(q,p), info_channel_capacity_bsc(p_error), info_channel_capacity_bec(epsilon), info_differential_entropy_gaussian(sigma), info_differential_entropy_uniform(a,b), info_rate_distortion_gaussian(variance,distortion), info_shannon_hartley(bandwidth_hz,snr_linear), stats_correlation(x,y), stats_spearman(x,y), stats_kendall(x,y), stats_mean(x), stats_median(x), stats_stddev(x), stats_skewness(x), stats_kurtosis(x), stats_var(x), stats_percentile(x,p), stats_mode(x), stats_geometric_mean(x), stats_harmonic_mean(x), stats_rms(x), stats_mad(x), stats_iqr(x), stats_ttest(x,mu), stats_ztest(x,mu,sigma), stats_acf(x,max_lag), stats_two_sample_ttest(a,b), stats_chi2_gof(observed,expected), signal_moving_average(x,window), signal_lowpass(x,cutoff,fs), signal_butterworth(x,cutoff,fs), signal_highpass(x,cutoff,fs), signal_bandpass(x,low,high,fs), signal_convolve(a,b), signal_correlate(a,b), signal_hamming(n), signal_hanning(n), signal_blackman(n), signal_parzen(n), signal_triangular(n), pde_heat_1d(x0,alpha,dx,dt,steps), pde_heat_2d(u0,alpha,dx,dy,dt,steps), pde_wave_1d(u0,v0,c,dx,dt,steps), pde_advection_1d(u0,v,dx,dt,steps), pde_poisson_2d(f,dx,dy,max_iterations,tolerance), pde_burgers_1d(u0,nu,dx,dt,steps), poly_deriv(coeffs), poly_add(a,b), poly_mul(a,b), poly_sub(a,b), poly_compose(p,q), poly_eval(coeffs,x), poly_integ(coeffs,c), fft_rfft(x), fft_dft(x), fft_irfft(spectrum,n), fft_ifft(spectrum), fft_fft2(S), fft_dct2(x), fft_idct2(x), fft_dst2(x), prob_norm_cdf(x,mu,sigma), prob_norm_pdf(x,mu,sigma), prob_norm_ppf(p,mu,sigma), prob_binom_pdf(k,n,p), prob_binom_cdf(k,n,p), prob_pois_pdf(k,lambda), prob_pois_cdf(k,lambda), prob_uniform_cdf(x,a,b), prob_exp_cdf(x,lambda), prob_exp_pdf(x,lambda), prob_chi2_cdf(x,df), prob_chi2_pdf(x,df), prob_t_cdf(x,df), prob_gamma_pdf(x,shape,scale), cplx_joukowski(re,im), cplx_joukowski_inv(re,im), cplx_hyperbolic_distance(z1re,z1im,z2re,z2im), cplx_mobius_re(a,b,c,d,zre,zim), cplx_poisson_kernel(theta,phi,r), cplx_cross_ratio(z1re,z1im,...), cplx_power_series_eval(coeffs,zre,zim), cplx_winding_number(G,z0re,z0im), cplx_residue_inv(pole_re,pole_im), cplx_contour_integral_oneoverz_im(), cplx_line_integral_one(), cplx_blaschke_product(zre,zim,zeros)\n"
            "  tensorops_norm(T), tensorops_inner(A,B), tensorops_matmul(A,B), tensorops_einsum(A,B)\n"
            "  diffgeo_gaussian_sphere(), diffgeo_mean_sphere(), diffgeo_principal_curvature_sphere(), diffgeo_gaussian_curvature_sphere(u,v), diffgeo_mean_curvature_sphere(u,v), diffgeo_ricci_scalar_sphere(u,v), diffgeo_einstein_scalar_sphere(u,v), diffgeo_surface_normal_sphere(u,v), diffgeo_christoffel_sphere(k,i,j,u,v), diffgeo_geodesic_euclidean(x0,y0,vx,vy,s_end), topo_euler_tetrahedron(), topo_euler_sphere_surface(), topo_vietoris_rips_betti0(D,r,max_dim), topo_betti_curve(D,thresholds,max_dim), topo_bottleneck_distance(dgm1,dgm2,dim), topo_wasserstein_distance(dgm1,dgm2,dim), topo_persistence_diagram(S,births)\n"
            "  fft([1,2,3,4])           vector FFT magnitude\n"
            "  erf(x), gamma(x), bessel_j0(x), spherical_jn(n,x)\n"
            "  kelvin_ber(0,x), struve_h(n,x), bessel_zero_jnu(nu,n)\n"
            "  kummer_m(a,b,x), hypergeo_2f1(a,b,c,x), whittaker_m(k,mu,x)\n"
            "  jacobi_p(n,a,b,x), ellip_k(k), jacobi_sn(u,k)\n"
            "  theta3(z,q), zeta(s), polylog(n,z), mathieu_ce(n,q,x)\n"
            "  heun_g(a,q,alpha,beta,gamma,delta,z), painleve1(x,y0,yp0)\n"
            "  legendre_p(n,x), beta(a,b)\n"
            "  clausen(theta), eta_dirichlet(s), debye(n,x)\n"
            "  sym_diff(\"expr\",\"var\"), sym_simplify(\"expr\"), sym_integrate(\"expr\",\"var\"), sym_eval(\"expr\",\"var=value\")\n"
            "  ode_euler(\"y - t*t\", 0, 1, 2, 100), ode_rk4(\"y\", 0, 1, 1, 100), ode_midpoint(\"y\", 0, 1, 1, 100), ode_rk45(\"y\", 0, 1, 1, 1e-6, 1e-9), ode_backward_euler(\"y\", 0, 1, 1, 100)\n"};
    }
    if (lcmd == "version") {
        std::ostringstream out;
        out << "mathscriptc " << ms::VERSION_STRING
            << " (commit " << ms::BUILD_COMMIT
            << ", built " << ms::BUILD_DATE << ")\n";
        return out.str();
    }
    if (lcmd == "show") {
        if (!state_.plot.valid) {
            return std::unexpected(DomainError{"show", "no plot in session"});
        }
        return format_plot_preview(state_.plot);
    }
    if (lcmd.rfind("saveplot ", 0) == 0) {
        if (!state_.plot.valid) {
            return std::unexpected(DomainError{"saveplot", "no plot in session"});
        }
        const std::string path = trim(cmd.substr(9));
        if (path.empty()) {
            return std::unexpected(DomainError{"saveplot", "missing path"});
        }
        std::ofstream out(path);
        if (!out) {
            return std::unexpected(DomainError{"saveplot", "cannot write: " + path});
        }
        out << format_plot_preview(state_.plot);
        return "saved plot preview to " + path + "\n";
    }
    if (lcmd == "vars") {
        std::ostringstream out;
        for (const auto& [name, value] : state_.scalars) {
            out << name << " = " << value << "\n";
        }
        for (const auto& [name, matrix] : state_.matrices) {
            out << name << " (" << matrix.rows() << "x" << matrix.cols() << ")\n";
        }
        if (state_.scalars.empty() && state_.matrices.empty()) {
            out << "(empty session)\n";
        }
        return out.str();
    }
    if (lcmd == "clear") {
        reset();
        return std::string{"session cleared\n"};
    }
    if (lcmd == "topology") {
        const auto topo = detect_topology();
        std::ostringstream out;
        out << "cores=" << topo.total_cores()
            << " threads=" << topo.total_threads()
            << " gpus=" << topo.total_gpus << "\n";
        return out.str();
    }
    if (lcmd == "simd") {
        const auto info = simd::dispatch_info();
        return "ISA: " + simd::isa_summary(info.isa) + "\n";
    }
    if (lcmd == "gpu") {
        std::ostringstream out;
        out << "cuda=" << (has_cuda() ? "yes" : "no")
            << " devices=" << get_gpu_count() << "\n";
        if (get_gpu_count() > 0) {
            out << "device0=" << get_gpu_model(0) << "\n";
        }
        return out.str();
    }
    if (lcmd == "dispatch") {
        const auto d = decide(512, ExecPolicy::AUTO);
        std::ostringstream out;
        out << "auto@512 backend="
            << (d.backend == Backend::CUDA ? "cuda" : "cpu")
            << " threads=" << d.n_threads << "\n";
        return out.str();
    }
    if (lcmd == "balance") {
        const auto lb = balance(1024, ExecPolicy::AUTO);
        std::ostringstream out;
        out << "backend=" << (lb.backend == Backend::CUDA ? "cuda" : "cpu")
            << " device=" << lb.cuda_device
            << " threads=" << lb.cpu_threads << "\n";
        return out.str();
    }
    if (lcmd.rfind("save ", 0) == 0) {
        const std::string path = trim(cmd.substr(5));
        if (path.empty()) {
            return std::unexpected(DomainError{"save", "missing path"});
        }
        auto saved = save_session(path);
        if (!saved) {
            return std::unexpected(saved.error());
        }
        return "saved session to " + path + "\n";
    }
    if (lcmd.rfind("load ", 0) == 0) {
        const std::string path = trim(cmd.substr(5));
        if (path.empty()) {
            return std::unexpected(DomainError{"load", "missing path"});
        }
        auto loaded = load_session(path);
        if (!loaded) {
            return std::unexpected(loaded.error());
        }
        return "loaded session from " + path + "\n";
    }

    if (lcmd == "frameworks") {
        return std::string{
            "frameworks: GRIA (gria matrix ops), Izaac (seed), AXIOM (evolve)\n"
            "  gria(M)  izaac seed N  axiom evolve\n"};
    }
    if (lcmd.rfind("izaac seed ", 0) == 0) {
        const std::string arg = trim(cmd.substr(11));
        double seed_value = 0.0;
        if (!parse_number(arg, seed_value)) {
            return std::unexpected(DomainError{"izaac", "expected numeric seed"});
        }
        izaac::seed_session(static_cast<uint64_t>(seed_value));
        return "izaac session seeded\n";
    }
    if (lcmd == "axiom evolve") {
        auto registry = axiom::PrimitiveRegistry::build_from_ms_namespace();
        axiom::Axiom engine(axiom::EvolutionConfig{.population_size = 20, .max_generations = 25}, registry);
        ColMatrix<double> data{{1, 2}, {3, 4}, {5, 6}};
        const auto best = engine
                                .evolve(
                                    [&](const axiom::Algorithm& a) {
                                        return engine.gria_fitness(a, data);
                                    },
                                    [](const axiom::Algorithm& a) { return a.fitness > 0.45; })
                                .value();
        std::ostringstream out;
        out << "axiom best fitness=" << best.fitness << " alpha-target fit\n";
        return out.str();
    }

    if (lcmd == "mpi") {
        static ms::distributed::MPIContext mpi = ms::distributed::init(0, nullptr);
        std::ostringstream out;
        out << "backend=" << ms::distributed::backend_name(mpi)
            << " rank=" << ms::distributed::rank(mpi)
            << " size=" << ms::distributed::size(mpi) << "\n";
        return out.str();
    }

    if (const auto sym = try_eval_sym_command(cmd)) {
        if (!sym->has_value()) {
            return std::unexpected(sym->error());
        }
        return **sym;
    }

    const auto eq = cmd.find('=');
    if (eq != std::string::npos) {
        const std::string lhs = trim(cmd.substr(0, eq));
        const std::string rhs = trim(cmd.substr(eq + 1));
        if (lhs.empty()) {
            return std::unexpected(DomainError{"assign", "missing variable name"});
        }

        double scalar = 0.0;
        if (parse_number(rhs, scalar)) {
            return assign_scalar(lhs, scalar);
        }

        std::string bigint_name;
        std::string bigint_decimal;
        if (try_parse_bigint_assignment(cmd, bigint_name, bigint_decimal)) {
            auto value = eval_bigint_string(bigint_decimal);
            if (!value) {
                return std::unexpected(value.error());
            }
            return assign_scalar(bigint_name, *value);
        }

        std::string bigint_unary_name;
        std::string bigint_unary_fn;
        int bigint_unary_n = 0;
        if (try_parse_bigint_unary_call(cmd, bigint_unary_name, bigint_unary_fn, bigint_unary_n)) {
            auto value = eval_bigint_unary(bigint_unary_fn, bigint_unary_n);
            if (!value) {
                return std::unexpected(value.error());
            }
            return assign_scalar(bigint_unary_name, *value);
        }

        std::string bigint_gcd_name;
        std::string bigint_gcd_a;
        std::string bigint_gcd_b;
        if (try_parse_bigint_gcd_assignment(cmd, bigint_gcd_name, bigint_gcd_a, bigint_gcd_b)) {
            auto value = eval_bigint_gcd_strings(bigint_gcd_a, bigint_gcd_b);
            if (!value) {
                return std::unexpected(value.error());
            }
            return assign_scalar(bigint_gcd_name, *value);
        }

        ScalarMatrixMixedCallAssign mixed_call{};
        if (try_parse_scalar_matrix_mixed_call_assignment(cmd, mixed_call)) {
            double scalar_arg = 0.0;
            if (!parse_number(trim_copy(mixed_call.arg_scalar), scalar_arg)) {
                auto scalar_expr = eval_scalar_expr(state_, mixed_call.arg_scalar);
                if (!scalar_expr) {
                    return std::unexpected(scalar_expr.error());
                }
                scalar_arg = *scalar_expr;
            }
            auto matrix = eval_matrix_operand(mixed_call.arg_matrix);
            if (!matrix) {
                return std::unexpected(matrix.error());
            }
            Result<double> value =
                std::unexpected(DomainError{"assign", "unsupported scalar-matrix mixed call"});
            if (mixed_call.callee == "finance_npv") {
                value = eval_finance_npv(scalar_arg, *matrix);
            } else if (mixed_call.callee == "info_renyi_entropy") {
                value = eval_info_renyi_entropy(scalar_arg, *matrix);
            } else if (mixed_call.callee == "info_tsallis_entropy") {
                value = eval_info_tsallis_entropy(scalar_arg, *matrix);
            } else if (mixed_call.callee == "combo_multinomial") {
                value = eval_combo_multinomial(scalar_arg, *matrix);
            }
            if (!value) {
                return std::unexpected(value.error());
            }
            return assign_scalar(mixed_call.target, *value);
        }

        TwoScalarMatrixMixedCallAssign two_scalar_matrix_call{};
        if (try_parse_two_scalar_matrix_mixed_call_assignment(cmd, two_scalar_matrix_call)) {
            double scalar_a = 0.0;
            double scalar_b = 0.0;
            if (!parse_number(trim_copy(two_scalar_matrix_call.arg_scalar_a), scalar_a)) {
                auto expr_a = eval_scalar_expr(state_, two_scalar_matrix_call.arg_scalar_a);
                if (!expr_a) {
                    return std::unexpected(expr_a.error());
                }
                scalar_a = *expr_a;
            }
            if (!parse_number(trim_copy(two_scalar_matrix_call.arg_scalar_b), scalar_b)) {
                auto expr_b = eval_scalar_expr(state_, two_scalar_matrix_call.arg_scalar_b);
                if (!expr_b) {
                    return std::unexpected(expr_b.error());
                }
                scalar_b = *expr_b;
            }
            auto matrix = eval_matrix_operand(two_scalar_matrix_call.arg_matrix);
            if (!matrix) {
                return std::unexpected(matrix.error());
            }
            Result<double> value = std::unexpected(
                DomainError{"assign", "unsupported two-scalar-matrix mixed call"});
            if (two_scalar_matrix_call.callee == "cplx_blaschke_product") {
                value = eval_cplx_blaschke_product(scalar_a, scalar_b, *matrix);
            }
            if (!value) {
                return std::unexpected(value.error());
            }
            return assign_scalar(two_scalar_matrix_call.target, *value);
        }

        MatrixScalarMixedCallAssign matrix_scalar_call{};
        if (try_parse_matrix_scalar_mixed_call_assignment(cmd, matrix_scalar_call)) {
            auto matrix = eval_matrix_operand(matrix_scalar_call.arg_matrix);
            if (!matrix) {
                return std::unexpected(matrix.error());
            }
            double scalar_arg = 0.0;
            if (!parse_number(trim_copy(matrix_scalar_call.arg_scalar), scalar_arg)) {
                auto scalar_expr = eval_scalar_expr(state_, matrix_scalar_call.arg_scalar);
                if (!scalar_expr) {
                    return std::unexpected(scalar_expr.error());
                }
                scalar_arg = *scalar_expr;
            }
            Result<double> value =
                std::unexpected(DomainError{"assign", "unsupported matrix-scalar mixed call"});
            if (matrix_scalar_call.callee == "bwt_decode_vec") {
                auto decoded = eval_bwt_decode_vec(*matrix, scalar_arg);
                if (!decoded) {
                    return std::unexpected(decoded.error());
                }
                state_.matrices[matrix_scalar_call.target] = *decoded;
                std::ostringstream out;
                out << matrix_scalar_call.target << " =\n";
                print_matrix(out, *decoded);
                return out.str();
            }
            if (matrix_scalar_call.callee == "quantum_time_evolution") {
                auto evolved = eval_quantum_time_evolution_matrix(*matrix, scalar_arg);
                if (!evolved) {
                    return std::unexpected(evolved.error());
                }
                state_.matrices[matrix_scalar_call.target] = *evolved;
                std::ostringstream out;
                out << matrix_scalar_call.target << " =\n";
                print_matrix(out, *evolved);
                return out.str();
            }
            if (matrix_scalar_call.callee == "combo_rank_combination") {
                const int n = static_cast<int>(scalar_arg);
                if (n < 0 || scalar_arg != n) {
                    return std::unexpected(DomainError{
                        "combo_rank_combination", "expected non-negative integer n"});
                }
                auto rank = eval_combo_rank_combination(*matrix, n);
                if (!rank) {
                    return std::unexpected(rank.error());
                }
                return assign_scalar(matrix_scalar_call.target, *rank);
            }
            if (matrix_scalar_call.callee == "combo_next_comb") {
                const int n = static_cast<int>(scalar_arg);
                if (n < 0 || scalar_arg != n) {
                    return std::unexpected(DomainError{
                        "combo_next_comb", "expected non-negative integer n"});
                }
                auto next = eval_combo_next_comb(*matrix, n);
                if (!next) {
                    return std::unexpected(next.error());
                }
                state_.matrices[matrix_scalar_call.target] = *next;
                std::ostringstream out;
                out << matrix_scalar_call.target << " =\n";
                print_matrix(out, *next);
                return out.str();
            }
            if (matrix_scalar_call.callee == "signal_moving_average") {
                const int window = static_cast<int>(scalar_arg);
                if (window < 1 || scalar_arg != window) {
                    return std::unexpected(DomainError{
                        "signal_moving_average", "expected positive integer window"});
                }
                auto smoothed = eval_signal_moving_average(*matrix, static_cast<size_t>(window));
                if (!smoothed) {
                    return std::unexpected(smoothed.error());
                }
                state_.matrices[matrix_scalar_call.target] = *smoothed;
                std::ostringstream out;
                out << matrix_scalar_call.target << " =\n";
                print_matrix(out, *smoothed);
                return out.str();
            }
            if (matrix_scalar_call.callee == "graph_bfs") {
                const int source = static_cast<int>(scalar_arg);
                if (source < 0 || scalar_arg != source) {
                    return std::unexpected(DomainError{
                        "graph_bfs", "expected non-negative integer source"});
                }
                auto order = eval_graph_bfs(*matrix, source);
                if (!order) {
                    return std::unexpected(order.error());
                }
                state_.matrices[matrix_scalar_call.target] = *order;
                std::ostringstream out;
                out << matrix_scalar_call.target << " =\n";
                print_matrix(out, *order);
                return out.str();
            }
            if (matrix_scalar_call.callee == "graph_dfs") {
                const int source = static_cast<int>(scalar_arg);
                if (source < 0 || scalar_arg != source) {
                    return std::unexpected(DomainError{
                        "graph_dfs", "expected non-negative integer source"});
                }
                auto order = eval_graph_dfs(*matrix, source);
                if (!order) {
                    return std::unexpected(order.error());
                }
                state_.matrices[matrix_scalar_call.target] = *order;
                std::ostringstream out;
                out << matrix_scalar_call.target << " =\n";
                print_matrix(out, *order);
                return out.str();
            }
            if (matrix_scalar_call.callee == "stats_percentile") {
                value = eval_stats_percentile(*matrix, scalar_arg);
            }
            if (matrix_scalar_call.callee == "stats_ttest") {
                value = eval_stats_ttest(*matrix, scalar_arg);
            }
            if (matrix_scalar_call.callee == "stats_acf") {
                const int max_lag = static_cast<int>(scalar_arg);
                if (max_lag < 0 || scalar_arg != max_lag) {
                    return std::unexpected(DomainError{
                        "stats_acf", "expected non-negative integer max_lag"});
                }
                auto acf_col = eval_stats_acf(*matrix, max_lag);
                if (!acf_col) {
                    return std::unexpected(acf_col.error());
                }
                state_.matrices[matrix_scalar_call.target] = *acf_col;
                std::ostringstream out;
                out << matrix_scalar_call.target << " =\n";
                print_matrix(out, *acf_col);
                return out.str();
            }
            if (matrix_scalar_call.callee == "fft_irfft") {
                const int n = static_cast<int>(scalar_arg);
                if (n < 1 || scalar_arg != n) {
                    return std::unexpected(DomainError{
                        "fft_irfft", "expected positive integer n"});
                }
                auto signal = eval_fft_irfft(*matrix, static_cast<size_t>(n));
                if (!signal) {
                    return std::unexpected(signal.error());
                }
                state_.matrices[matrix_scalar_call.target] = *signal;
                std::ostringstream out;
                out << matrix_scalar_call.target << " =\n";
                print_matrix(out, *signal);
                return out.str();
            }
            if (matrix_scalar_call.callee == "poly_integ") {
                auto integrated = eval_poly_integ(*matrix, scalar_arg);
                if (!integrated) {
                    return std::unexpected(integrated.error());
                }
                state_.matrices[matrix_scalar_call.target] = *integrated;
                std::ostringstream out;
                out << matrix_scalar_call.target << " =\n";
                print_matrix(out, *integrated);
                return out.str();
            }
            if (matrix_scalar_call.callee == "poly_eval") {
                value = eval_poly_eval(*matrix, scalar_arg);
            } else if (matrix_scalar_call.callee == "geo_bezier_eval_x") {
                value = eval_geo_bezier_eval_x(*matrix, scalar_arg);
            } else if (matrix_scalar_call.callee == "geo_bezier_eval_y") {
                value = eval_geo_bezier_eval_y(*matrix, scalar_arg);
            }
            if (!value) {
                return std::unexpected(value.error());
            }
            return assign_scalar(matrix_scalar_call.target, *value);
        }

        ScalarDualMatrixCallAssign dual_call{};
        if (try_parse_scalar_dual_matrix_call_assignment(cmd, dual_call)) {
            auto resolve_operand = [this](const std::string& text) { return eval_matrix_operand(text); };
            auto arg_a_m = resolve_operand(dual_call.arg_a);
            if (!arg_a_m) {
                return std::unexpected(arg_a_m.error());
            }
            auto arg_b_m = resolve_operand(dual_call.arg_b);
            if (!arg_b_m) {
                return std::unexpected(arg_b_m.error());
            }
            if (dual_call.callee == "control_step_final") {
                auto value = eval_control_step_final(*arg_a_m, *arg_b_m);
                if (!value) {
                    return std::unexpected(value.error());
                }
                return assign_scalar(dual_call.target, *value);
            }
            if (dual_call.callee == "control_pidtune_kp") {
                auto value = eval_control_pidtune_kp(*arg_a_m, *arg_b_m);
                if (!value) {
                    return std::unexpected(value.error());
                }
                return assign_scalar(dual_call.target, *value);
            }
            if (dual_call.callee == "control_pidtune_ki") {
                auto value = eval_control_pidtune_ki(*arg_a_m, *arg_b_m);
                if (!value) {
                    return std::unexpected(value.error());
                }
                return assign_scalar(dual_call.target, *value);
            }
            if (dual_call.callee == "control_pidtune_kd") {
                auto value = eval_control_pidtune_kd(*arg_a_m, *arg_b_m);
                if (!value) {
                    return std::unexpected(value.error());
                }
                return assign_scalar(dual_call.target, *value);
            }
            if (dual_call.callee == "control_impulse_final") {
                auto value = eval_control_impulse_final(*arg_a_m, *arg_b_m);
                if (!value) {
                    return std::unexpected(value.error());
                }
                return assign_scalar(dual_call.target, *value);
            }
            if (dual_call.callee == "control_dcgain") {
                auto value = eval_control_dcgain(*arg_a_m, *arg_b_m);
                if (!value) {
                    return std::unexpected(value.error());
                }
                return assign_scalar(dual_call.target, *value);
            }
            if (dual_call.callee == "control_phase_margin") {
                auto value = eval_control_phase_margin(*arg_a_m, *arg_b_m);
                if (!value) {
                    return std::unexpected(value.error());
                }
                return assign_scalar(dual_call.target, *value);
            }
            if (dual_call.callee == "control_gain_margin") {
                auto value = eval_control_gain_margin(*arg_a_m, *arg_b_m);
                if (!value) {
                    return std::unexpected(value.error());
                }
                return assign_scalar(dual_call.target, *value);
            }
            if (dual_call.callee == "control_is_stable") {
                auto value = eval_control_is_stable(*arg_a_m, *arg_b_m);
                if (!value) {
                    return std::unexpected(value.error());
                }
                return assign_scalar(dual_call.target, *value);
            }
            if (dual_call.callee == "control_is_controllable") {
                auto value = eval_control_is_controllable(*arg_a_m, *arg_b_m);
                if (!value) {
                    return std::unexpected(value.error());
                }
                return assign_scalar(dual_call.target, *value);
            }
            if (dual_call.callee == "control_is_observable") {
                auto value = eval_control_is_observable(*arg_a_m, *arg_b_m);
                if (!value) {
                    return std::unexpected(value.error());
                }
                return assign_scalar(dual_call.target, *value);
            }
            if (dual_call.callee == "numthy_crt") {
                auto value = eval_numthy_crt(*arg_a_m, *arg_b_m);
                if (!value) {
                    return std::unexpected(value.error());
                }
                return assign_scalar(dual_call.target, *value);
            }
            if (dual_call.callee == "info_kl_divergence") {
                auto value = eval_info_kl_divergence(*arg_a_m, *arg_b_m);
                if (!value) {
                    return std::unexpected(value.error());
                }
                return assign_scalar(dual_call.target, *value);
            }
            if (dual_call.callee == "tensorops_inner") {
                auto value = eval_tensorops_inner(*arg_a_m, *arg_b_m);
                if (!value) {
                    return std::unexpected(value.error());
                }
                return assign_scalar(dual_call.target, *value);
            }
            if (dual_call.callee == "finance_portfolio_return") {
                auto value = eval_finance_portfolio_return(*arg_a_m, *arg_b_m);
                if (!value) {
                    return std::unexpected(value.error());
                }
                return assign_scalar(dual_call.target, *value);
            }
            if (dual_call.callee == "finance_portfolio_variance") {
                auto value = eval_finance_portfolio_variance(*arg_a_m, *arg_b_m);
                if (!value) {
                    return std::unexpected(value.error());
                }
                return assign_scalar(dual_call.target, *value);
            }
            if (dual_call.callee == "info_cross_entropy") {
                auto value = eval_info_cross_entropy(*arg_a_m, *arg_b_m);
                if (!value) {
                    return std::unexpected(value.error());
                }
                return assign_scalar(dual_call.target, *value);
            }
            if (dual_call.callee == "info_js_divergence") {
                auto value = eval_info_js_divergence(*arg_a_m, *arg_b_m);
                if (!value) {
                    return std::unexpected(value.error());
                }
                return assign_scalar(dual_call.target, *value);
            }
            if (dual_call.callee == "info_tv_distance") {
                auto value = eval_info_tv_distance(*arg_a_m, *arg_b_m);
                if (!value) {
                    return std::unexpected(value.error());
                }
                return assign_scalar(dual_call.target, *value);
            }
            if (dual_call.callee == "info_hellinger_dist") {
                auto value = eval_info_hellinger_dist(*arg_a_m, *arg_b_m);
                if (!value) {
                    return std::unexpected(value.error());
                }
                return assign_scalar(dual_call.target, *value);
            }
            if (dual_call.callee == "quantum_fidelity") {
                auto value = eval_quantum_fidelity(*arg_a_m, *arg_b_m);
                if (!value) {
                    return std::unexpected(value.error());
                }
                return assign_scalar(dual_call.target, *value);
            }
            if (dual_call.callee == "stats_correlation") {
                auto value = eval_stats_correlation(*arg_a_m, *arg_b_m);
                if (!value) {
                    return std::unexpected(value.error());
                }
                return assign_scalar(dual_call.target, *value);
            }
            if (dual_call.callee == "stats_spearman") {
                auto value = eval_stats_spearman(*arg_a_m, *arg_b_m);
                if (!value) {
                    return std::unexpected(value.error());
                }
                return assign_scalar(dual_call.target, *value);
            }
            if (dual_call.callee == "stats_kendall") {
                auto value = eval_stats_kendall(*arg_a_m, *arg_b_m);
                if (!value) {
                    return std::unexpected(value.error());
                }
                return assign_scalar(dual_call.target, *value);
            }
            if (dual_call.callee == "stats_two_sample_ttest") {
                auto value = eval_stats_two_sample_ttest(*arg_a_m, *arg_b_m);
                if (!value) {
                    return std::unexpected(value.error());
                }
                return assign_scalar(dual_call.target, *value);
            }
            if (dual_call.callee == "stats_chi2_gof") {
                auto value = eval_stats_chi2_gof(*arg_a_m, *arg_b_m);
                if (!value) {
                    return std::unexpected(value.error());
                }
                return assign_scalar(dual_call.target, *value);
            }
            if (dual_call.callee == "quantum_expectation_dm") {
                auto value = eval_quantum_expectation_dm(*arg_a_m, *arg_b_m);
                if (!value) {
                    return std::unexpected(value.error());
                }
                return assign_scalar(dual_call.target, *value);
            }
            if (dual_call.callee == "quantum_expectation") {
                auto value = eval_quantum_expectation(*arg_a_m, *arg_b_m);
                if (!value) {
                    return std::unexpected(value.error());
                }
                return assign_scalar(dual_call.target, *value);
            }
            if (dual_call.callee == "quantum_inner") {
                auto value = eval_quantum_inner(*arg_a_m, *arg_b_m);
                if (!value) {
                    return std::unexpected(value.error());
                }
                return assign_scalar(dual_call.target, *value);
            }
            if (dual_call.callee == "quantum_trace_distance") {
                auto value = eval_quantum_trace_distance(*arg_a_m, *arg_b_m);
                if (!value) {
                    return std::unexpected(value.error());
                }
                return assign_scalar(dual_call.target, *value);
            }
            if (dual_call.callee == "ml_categorical_crossentropy") {
                auto value = eval_ml_categorical_crossentropy(*arg_a_m, *arg_b_m);
                if (!value) {
                    return std::unexpected(value.error());
                }
                return assign_scalar(dual_call.target, *value);
            }
            auto y_pred = matrix_to_ml_vec(*arg_a_m, dual_call.callee.c_str());
            if (!y_pred) {
                return std::unexpected(y_pred.error());
            }
            auto y_true = matrix_to_ml_vec(*arg_b_m, dual_call.callee.c_str());
            if (!y_true) {
                return std::unexpected(y_true.error());
            }
            auto metric = eval_ml_metric(dual_call.callee, *y_pred, *y_true);
            if (!metric) {
                return std::unexpected(metric.error());
            }
            return assign_scalar(dual_call.target, *metric);
        }

        MatrixDualMatrixCallAssign matrix_dual_call{};
        if (try_parse_matrix_dual_matrix_call_assignment(cmd, matrix_dual_call)) {
            auto resolve_operand = [this](const std::string& text) { return eval_matrix_operand(text); };
            auto arg_a_m = resolve_operand(matrix_dual_call.arg_a);
            if (!arg_a_m) {
                return std::unexpected(arg_a_m.error());
            }
            auto arg_b_m = resolve_operand(matrix_dual_call.arg_b);
            if (!arg_b_m) {
                return std::unexpected(arg_b_m.error());
            }
            if (matrix_dual_call.callee == "control_lyap") {
                auto value = eval_control_lyap(*arg_a_m, *arg_b_m);
                if (!value) {
                    return std::unexpected(value.error());
                }
                state_.matrices[matrix_dual_call.target] = *value;
                std::ostringstream out;
                out << matrix_dual_call.target << " =\n";
                print_matrix(out, *value);
                return out.str();
            }
            if (matrix_dual_call.callee == "control_dlyap") {
                auto value = eval_control_dlyap(*arg_a_m, *arg_b_m);
                if (!value) {
                    return std::unexpected(value.error());
                }
                state_.matrices[matrix_dual_call.target] = *value;
                std::ostringstream out;
                out << matrix_dual_call.target << " =\n";
                print_matrix(out, *value);
                return out.str();
            }
            if (matrix_dual_call.callee == "huffman_decode_vec") {
                auto value = eval_huffman_decode_vec(*arg_a_m, *arg_b_m);
                if (!value) {
                    return std::unexpected(value.error());
                }
                state_.matrices[matrix_dual_call.target] = *value;
                std::ostringstream out;
                out << matrix_dual_call.target << " =\n";
                print_matrix(out, *value);
                return out.str();
            }
            if (matrix_dual_call.callee == "quantum_op_apply") {
                auto value = eval_quantum_op_apply(*arg_a_m, *arg_b_m);
                if (!value) {
                    return std::unexpected(value.error());
                }
                state_.matrices[matrix_dual_call.target] = *value;
                std::ostringstream out;
                out << matrix_dual_call.target << " =\n";
                print_matrix(out, *value);
                return out.str();
            }
            if (matrix_dual_call.callee == "topo_persistence_diagram") {
                auto value = eval_topo_persistence_diagram(*arg_a_m, *arg_b_m);
                if (!value) {
                    return std::unexpected(value.error());
                }
                state_.matrices[matrix_dual_call.target] = *value;
                std::ostringstream out;
                out << matrix_dual_call.target << " =\n";
                print_matrix(out, *value);
                return out.str();
            }
            if (matrix_dual_call.callee == "control_margins") {
                auto value = eval_control_margins(*arg_a_m, *arg_b_m);
                if (!value) {
                    return std::unexpected(value.error());
                }
                state_.matrices[matrix_dual_call.target] = *value;
                std::ostringstream out;
                out << matrix_dual_call.target << " =\n";
                print_matrix(out, *value);
                return out.str();
            }
            if (matrix_dual_call.callee == "quantum_commutator") {
                auto value = eval_quantum_commutator(*arg_a_m, *arg_b_m);
                if (!value) {
                    return std::unexpected(value.error());
                }
                state_.matrices[matrix_dual_call.target] = *value;
                std::ostringstream out;
                out << matrix_dual_call.target << " =\n";
                print_matrix(out, *value);
                return out.str();
            }
            if (matrix_dual_call.callee == "signal_convolve") {
                auto value = eval_signal_convolve(*arg_a_m, *arg_b_m);
                if (!value) {
                    return std::unexpected(value.error());
                }
                state_.matrices[matrix_dual_call.target] = *value;
                std::ostringstream out;
                out << matrix_dual_call.target << " =\n";
                print_matrix(out, *value);
                return out.str();
            }
            if (matrix_dual_call.callee == "signal_correlate") {
                auto value = eval_signal_correlate(*arg_a_m, *arg_b_m);
                if (!value) {
                    return std::unexpected(value.error());
                }
                state_.matrices[matrix_dual_call.target] = *value;
                std::ostringstream out;
                out << matrix_dual_call.target << " =\n";
                print_matrix(out, *value);
                return out.str();
            }
            if (matrix_dual_call.callee == "ml_mat_mul") {
                auto value = eval_ml_mat_mul(*arg_a_m, *arg_b_m);
                if (!value) {
                    return std::unexpected(value.error());
                }
                state_.matrices[matrix_dual_call.target] = *value;
                std::ostringstream out;
                out << matrix_dual_call.target << " =\n";
                print_matrix(out, *value);
                return out.str();
            }
            if (matrix_dual_call.callee == "poly_add") {
                auto value = eval_poly_add(*arg_a_m, *arg_b_m);
                if (!value) {
                    return std::unexpected(value.error());
                }
                state_.matrices[matrix_dual_call.target] = *value;
                std::ostringstream out;
                out << matrix_dual_call.target << " =\n";
                print_matrix(out, *value);
                return out.str();
            }
            if (matrix_dual_call.callee == "poly_mul") {
                auto value = eval_poly_mul(*arg_a_m, *arg_b_m);
                if (!value) {
                    return std::unexpected(value.error());
                }
                state_.matrices[matrix_dual_call.target] = *value;
                std::ostringstream out;
                out << matrix_dual_call.target << " =\n";
                print_matrix(out, *value);
                return out.str();
            }
            if (matrix_dual_call.callee == "poly_sub") {
                auto value = eval_poly_sub(*arg_a_m, *arg_b_m);
                if (!value) {
                    return std::unexpected(value.error());
                }
                state_.matrices[matrix_dual_call.target] = *value;
                std::ostringstream out;
                out << matrix_dual_call.target << " =\n";
                print_matrix(out, *value);
                return out.str();
            }
            if (matrix_dual_call.callee == "poly_compose") {
                auto value = eval_poly_compose(*arg_a_m, *arg_b_m);
                if (!value) {
                    return std::unexpected(value.error());
                }
                state_.matrices[matrix_dual_call.target] = *value;
                std::ostringstream out;
                out << matrix_dual_call.target << " =\n";
                print_matrix(out, *value);
                return out.str();
            }
            if (matrix_dual_call.callee == "quantum_tensor_product") {
                auto value = eval_quantum_tensor_product(*arg_a_m, *arg_b_m);
                if (!value) {
                    return std::unexpected(value.error());
                }
                state_.matrices[matrix_dual_call.target] = *value;
                std::ostringstream out;
                out << matrix_dual_call.target << " =\n";
                print_matrix(out, *value);
                return out.str();
            }
            return std::unexpected(
                DomainError{"assign", "unsupported matrix dual matrix call"});
        }

        MatrixTripleMatrixCallAssign matrix_triple_call{};
        if (try_parse_matrix_triple_matrix_call_assignment(cmd, matrix_triple_call)) {
            auto resolve_operand = [this](const std::string& text) { return eval_matrix_operand(text); };
            auto arg_a_m = resolve_operand(matrix_triple_call.arg_a);
            if (!arg_a_m) {
                return std::unexpected(arg_a_m.error());
            }
            auto arg_b_m = resolve_operand(matrix_triple_call.arg_b);
            if (!arg_b_m) {
                return std::unexpected(arg_b_m.error());
            }
            auto arg_c_m = resolve_operand(matrix_triple_call.arg_c);
            if (!arg_c_m) {
                return std::unexpected(arg_c_m.error());
            }
            if (matrix_triple_call.callee == "control_place") {
                auto value = eval_control_place(*arg_a_m, *arg_b_m, *arg_c_m);
                if (!value) {
                    return std::unexpected(value.error());
                }
                state_.matrices[matrix_triple_call.target] = *value;
                std::ostringstream out;
                out << matrix_triple_call.target << " =\n";
                print_matrix(out, *value);
                return out.str();
            }
            return std::unexpected(
                DomainError{"assign", "unsupported matrix triple matrix call"});
        }

        MatrixQuadMatrixCallAssign matrix_quad_call{};
        if (try_parse_matrix_quad_matrix_call_assignment(cmd, matrix_quad_call)) {
            auto resolve_operand = [this](const std::string& text) { return eval_matrix_operand(text); };
            auto arg_a_m = resolve_operand(matrix_quad_call.arg_a);
            if (!arg_a_m) {
                return std::unexpected(arg_a_m.error());
            }
            auto arg_b_m = resolve_operand(matrix_quad_call.arg_b);
            if (!arg_b_m) {
                return std::unexpected(arg_b_m.error());
            }
            auto arg_c_m = resolve_operand(matrix_quad_call.arg_c);
            if (!arg_c_m) {
                return std::unexpected(arg_c_m.error());
            }
            auto arg_d_m = resolve_operand(matrix_quad_call.arg_d);
            if (!arg_d_m) {
                return std::unexpected(arg_d_m.error());
            }
            if (matrix_quad_call.callee == "control_lqr") {
                auto value = eval_control_lqr(*arg_a_m, *arg_b_m, *arg_c_m, *arg_d_m);
                if (!value) {
                    return std::unexpected(value.error());
                }
                state_.matrices[matrix_quad_call.target] = *value;
                std::ostringstream out;
                out << matrix_quad_call.target << " =\n";
                print_matrix(out, *value);
                return out.str();
            }
            if (matrix_quad_call.callee == "control_riccati") {
                auto value = eval_control_riccati(*arg_a_m, *arg_b_m, *arg_c_m, *arg_d_m);
                if (!value) {
                    return std::unexpected(value.error());
                }
                state_.matrices[matrix_quad_call.target] = *value;
                std::ostringstream out;
                out << matrix_quad_call.target << " =\n";
                print_matrix(out, *value);
                return out.str();
            }
            if (matrix_quad_call.callee == "control_dare") {
                auto value = eval_control_dare(*arg_a_m, *arg_b_m, *arg_c_m, *arg_d_m);
                if (!value) {
                    return std::unexpected(value.error());
                }
                state_.matrices[matrix_quad_call.target] = *value;
                std::ostringstream out;
                out << matrix_quad_call.target << " =\n";
                print_matrix(out, *value);
                return out.str();
            }
            return std::unexpected(
                DomainError{"assign", "unsupported matrix quad matrix call"});
        }

        MultiMatrixCallAssign multi_call{};
        if (try_parse_multi_matrix_call_assignment(cmd, multi_call)) {
            return assign_multi_matrix_call(multi_call);
        }

        MatrixCallAssign matrix_call{};
        if (try_parse_matrix_call_assignment(cmd, matrix_call)) {
            return assign_matrix_call(matrix_call);
        }

        ScalarMatrixCallAssign scalar_matrix_call{};
        if (try_parse_scalar_matrix_call_assignment(cmd, scalar_matrix_call)) {
            return assign_scalar_matrix_call(scalar_matrix_call);
        }

        if (const auto nullary = parse_nullary_matrix_call(rhs)) {
            auto value = eval_nullary_matrix_call(*nullary);
            if (!value) {
                return std::unexpected(value.error());
            }
            state_.matrices[lhs] = *value;
            std::ostringstream out;
            out << lhs << " =\n";
            print_matrix(out, *value);
            return out.str();
        }

        if (const auto unary = parse_unary_scalar_matrix_call(rhs)) {
            auto theta = eval_scalar_expr(state_, unary->second);
            if (!theta) {
                return std::unexpected(theta.error());
            }
            auto value = eval_unary_scalar_matrix_call(unary->first, *theta);
            if (!value) {
                return std::unexpected(value.error());
            }
            state_.matrices[lhs] = *value;
            std::ostringstream out;
            out << lhs << " =\n";
            print_matrix(out, *value);
            return out.str();
        }

        if (const auto nullary = parse_nullary_scalar_call(rhs)) {
            auto value = eval_nullary_scalar_call(*nullary);
            if (!value) {
                return std::unexpected(value.error());
            }
            return assign_scalar(lhs, *value);
        }

        if (is_scalar_expression_rhs(rhs)) {
            return assign_scalar_expr(lhs, rhs);
        }

        if (const auto open = rhs.find('('); open != std::string::npos && !rhs.empty() &&
                                         rhs.back() == ')') {
            const std::string callee = lower(trim_copy(rhs.substr(0, open)));
            if (callee == "finance_binomial_call") {
                const auto call_args = split_call_args(rhs);
                if (!call_args || call_args->size() != 6) {
                    return std::unexpected(DomainError{
                        "finance_binomial_call",
                        "expected finance_binomial_call(S,K,T,r,sigma,steps)"});
                }
                double S = 0.0;
                double K = 0.0;
                double T = 0.0;
                double r = 0.0;
                double sigma = 0.0;
                double steps_d = 0.0;
                if (!parse_number(trim_copy((*call_args)[0]), S) ||
                    !parse_number(trim_copy((*call_args)[1]), K) ||
                    !parse_number(trim_copy((*call_args)[2]), T) ||
                    !parse_number(trim_copy((*call_args)[3]), r) ||
                    !parse_number(trim_copy((*call_args)[4]), sigma) ||
                    !parse_number(trim_copy((*call_args)[5]), steps_d)) {
                    return std::unexpected(DomainError{
                        "finance_binomial_call",
                        "expected finance_binomial_call(S,K,T,r,sigma,steps)"});
                }
                const int steps = static_cast<int>(steps_d);
                if (steps < 0 || steps_d != steps) {
                    return std::unexpected(DomainError{
                        "finance_binomial_call", "expected non-negative integer steps"});
                }
                return assign_scalar(lhs, finance::binomial_call(S, K, T, r, sigma, steps));
            }
            if (callee == "finance_binomial_put") {
                const auto call_args = split_call_args(rhs);
                if (!call_args || call_args->size() != 6) {
                    return std::unexpected(DomainError{
                        "finance_binomial_put",
                        "expected finance_binomial_put(S,K,T,r,sigma,steps)"});
                }
                double S = 0.0;
                double K = 0.0;
                double T = 0.0;
                double r = 0.0;
                double sigma = 0.0;
                double steps_d = 0.0;
                if (!parse_number(trim_copy((*call_args)[0]), S) ||
                    !parse_number(trim_copy((*call_args)[1]), K) ||
                    !parse_number(trim_copy((*call_args)[2]), T) ||
                    !parse_number(trim_copy((*call_args)[3]), r) ||
                    !parse_number(trim_copy((*call_args)[4]), sigma) ||
                    !parse_number(trim_copy((*call_args)[5]), steps_d)) {
                    return std::unexpected(DomainError{
                        "finance_binomial_put",
                        "expected finance_binomial_put(S,K,T,r,sigma,steps)"});
                }
                const int steps = static_cast<int>(steps_d);
                if (steps < 0 || steps_d != steps) {
                    return std::unexpected(DomainError{
                        "finance_binomial_put", "expected non-negative integer steps"});
                }
                return assign_scalar(lhs, finance::binomial_put(S, K, T, r, sigma, steps));
            }
            if (callee == "finance_bs_delta") {
                const auto call_args = split_call_args(rhs);
                if (!call_args || call_args->size() != 6) {
                    return std::unexpected(DomainError{
                        "finance_bs_delta",
                        "expected finance_bs_delta(S,K,T,r,sigma,call)"});
                }
                double S = 0.0;
                double K = 0.0;
                double T = 0.0;
                double r = 0.0;
                double sigma = 0.0;
                double call_d = 0.0;
                if (!parse_number(trim_copy((*call_args)[0]), S) ||
                    !parse_number(trim_copy((*call_args)[1]), K) ||
                    !parse_number(trim_copy((*call_args)[2]), T) ||
                    !parse_number(trim_copy((*call_args)[3]), r) ||
                    !parse_number(trim_copy((*call_args)[4]), sigma) ||
                    !parse_number(trim_copy((*call_args)[5]), call_d)) {
                    return std::unexpected(DomainError{
                        "finance_bs_delta",
                        "expected finance_bs_delta(S,K,T,r,sigma,call)"});
                }
                const int call = static_cast<int>(call_d);
                if (call_d != call) {
                    return std::unexpected(DomainError{
                        "finance_bs_delta", "expected integer call (0=put, 1=call)"});
                }
                return assign_scalar(
                    lhs, finance::bs_delta(S, K, T, r, sigma, call != 0));
            }
            if (callee == "finance_bs_theta") {
                const auto call_args = split_call_args(rhs);
                if (!call_args || call_args->size() != 6) {
                    return std::unexpected(DomainError{
                        "finance_bs_theta",
                        "expected finance_bs_theta(S,K,T,r,sigma,call)"});
                }
                double S = 0.0;
                double K = 0.0;
                double T = 0.0;
                double r = 0.0;
                double sigma = 0.0;
                double call_d = 0.0;
                if (!parse_number(trim_copy((*call_args)[0]), S) ||
                    !parse_number(trim_copy((*call_args)[1]), K) ||
                    !parse_number(trim_copy((*call_args)[2]), T) ||
                    !parse_number(trim_copy((*call_args)[3]), r) ||
                    !parse_number(trim_copy((*call_args)[4]), sigma) ||
                    !parse_number(trim_copy((*call_args)[5]), call_d)) {
                    return std::unexpected(DomainError{
                        "finance_bs_theta",
                        "expected finance_bs_theta(S,K,T,r,sigma,call)"});
                }
                const int call = static_cast<int>(call_d);
                if (call_d != call) {
                    return std::unexpected(DomainError{
                        "finance_bs_theta", "expected integer call (0=put, 1=call)"});
                }
                return assign_scalar(
                    lhs, finance::bs_theta(S, K, T, r, sigma, call != 0));
            }
            if (callee == "finance_bs_rho") {
                const auto call_args = split_call_args(rhs);
                if (!call_args || call_args->size() != 6) {
                    return std::unexpected(DomainError{
                        "finance_bs_rho",
                        "expected finance_bs_rho(S,K,T,r,sigma,call)"});
                }
                double S = 0.0;
                double K = 0.0;
                double T = 0.0;
                double r = 0.0;
                double sigma = 0.0;
                double call_d = 0.0;
                if (!parse_number(trim_copy((*call_args)[0]), S) ||
                    !parse_number(trim_copy((*call_args)[1]), K) ||
                    !parse_number(trim_copy((*call_args)[2]), T) ||
                    !parse_number(trim_copy((*call_args)[3]), r) ||
                    !parse_number(trim_copy((*call_args)[4]), sigma) ||
                    !parse_number(trim_copy((*call_args)[5]), call_d)) {
                    return std::unexpected(DomainError{
                        "finance_bs_rho",
                        "expected finance_bs_rho(S,K,T,r,sigma,call)"});
                }
                const int call = static_cast<int>(call_d);
                if (call_d != call) {
                    return std::unexpected(DomainError{
                        "finance_bs_rho", "expected integer call (0=put, 1=call)"});
                }
                return assign_scalar(
                    lhs, finance::bs_rho(S, K, T, r, sigma, call != 0));
            }
            if (callee == "quantum_entanglement_entropy") {
                const auto call_args = split_call_args(rhs);
                if (!call_args || call_args->size() != 3) {
                    return std::unexpected(DomainError{
                        "quantum_entanglement_entropy",
                        "expected quantum_entanglement_entropy(psi, dim_a, dim_b)"});
                }
                auto psi_m = eval_matrix_operand(trim_copy(call_args->front()));
                if (!psi_m) {
                    return std::unexpected(psi_m.error());
                }
                double dim_a_d = 0.0;
                double dim_b_d = 0.0;
                if (!parse_number(trim_copy((*call_args)[1]), dim_a_d) ||
                    !parse_number(trim_copy((*call_args)[2]), dim_b_d)) {
                    return std::unexpected(DomainError{
                        "quantum_entanglement_entropy",
                        "expected quantum_entanglement_entropy(psi, dim_a, dim_b)"});
                }
                const int dim_a = static_cast<int>(dim_a_d);
                const int dim_b = static_cast<int>(dim_b_d);
                if (dim_a < 1 || dim_b < 1 || dim_a_d != dim_a || dim_b_d != dim_b) {
                    return std::unexpected(DomainError{
                        "quantum_entanglement_entropy", "expected positive integer dim_a and dim_b"});
                }
                auto value = eval_quantum_entanglement_entropy(*psi_m, dim_a, dim_b);
                if (!value) {
                    return std::unexpected(value.error());
                }
                return assign_scalar(lhs, *value);
            }
            if (callee == "graph_max_flow") {
                const auto call_args = split_call_args(rhs);
                if (!call_args || call_args->size() != 3) {
                    return std::unexpected(DomainError{
                        "graph_max_flow", "expected graph_max_flow(A, source, sink)"});
                }
                auto adj_m = eval_matrix_operand(trim_copy(call_args->front()));
                if (!adj_m) {
                    return std::unexpected(adj_m.error());
                }
                double source_d = 0.0;
                double sink_d = 0.0;
                if (!parse_number(trim_copy((*call_args)[1]), source_d) ||
                    !parse_number(trim_copy((*call_args)[2]), sink_d)) {
                    return std::unexpected(DomainError{
                        "graph_max_flow", "expected graph_max_flow(A, source, sink)"});
                }
                const int source = static_cast<int>(source_d);
                const int sink = static_cast<int>(sink_d);
                if (source_d != source || sink_d != sink) {
                    return std::unexpected(DomainError{
                        "graph_max_flow", "expected integer source and sink"});
                }
                auto value = eval_graph_max_flow(*adj_m, source, sink);
                if (!value) {
                    return std::unexpected(value.error());
                }
                return assign_scalar(lhs, *value);
            }
            if (callee == "signal_lowpass") {
                const auto call_args = split_call_args(rhs);
                if (!call_args || call_args->size() != 3) {
                    return std::unexpected(DomainError{
                        "signal_lowpass", "expected signal_lowpass(x, cutoff, fs)"});
                }
                auto x_m = eval_matrix_operand(trim_copy(call_args->front()));
                if (!x_m) {
                    return std::unexpected(x_m.error());
                }
                double cutoff = 0.0;
                double fs = 0.0;
                if (!parse_number(trim_copy((*call_args)[1]), cutoff) ||
                    !parse_number(trim_copy((*call_args)[2]), fs)) {
                    auto c_expr = eval_scalar_expr(state_, trim_copy((*call_args)[1]));
                    if (!c_expr) {
                        return std::unexpected(c_expr.error());
                    }
                    cutoff = *c_expr;
                    auto fs_expr = eval_scalar_expr(state_, trim_copy((*call_args)[2]));
                    if (!fs_expr) {
                        return std::unexpected(fs_expr.error());
                    }
                    fs = *fs_expr;
                }
                auto filtered = eval_signal_lowpass(*x_m, cutoff, fs);
                if (!filtered) {
                    return std::unexpected(filtered.error());
                }
                state_.matrices[lhs] = *filtered;
                std::ostringstream out;
                out << lhs << " =\n";
                print_matrix(out, *filtered);
                return out.str();
            }
            if (callee == "signal_butterworth" || callee == "signal_highpass" ||
                callee == "signal_bandpass") {
                const auto call_args = split_call_args(rhs);
                if (!call_args ||
                    (callee == "signal_bandpass" ? call_args->size() != 4 : call_args->size() != 3)) {
                    return std::unexpected(DomainError{
                        callee, "expected " + callee +
                                    (callee == "signal_bandpass" ? "(x, low, high, fs)"
                                                                 : "(x, cutoff, fs)")});
                }
                auto x_m = eval_matrix_operand(trim_copy(call_args->front()));
                if (!x_m) {
                    return std::unexpected(x_m.error());
                }
                double cutoff = 0.0;
                double fs = 0.0;
                double low = 0.0;
                double high = 0.0;
                if (callee == "signal_bandpass") {
                    if (!parse_number(trim_copy((*call_args)[1]), low) ||
                        !parse_number(trim_copy((*call_args)[2]), high) ||
                        !parse_number(trim_copy((*call_args)[3]), fs)) {
                        auto low_expr = eval_scalar_expr(state_, trim_copy((*call_args)[1]));
                        if (!low_expr) {
                            return std::unexpected(low_expr.error());
                        }
                        low = *low_expr;
                        auto high_expr = eval_scalar_expr(state_, trim_copy((*call_args)[2]));
                        if (!high_expr) {
                            return std::unexpected(high_expr.error());
                        }
                        high = *high_expr;
                        auto fs_expr = eval_scalar_expr(state_, trim_copy((*call_args)[3]));
                        if (!fs_expr) {
                            return std::unexpected(fs_expr.error());
                        }
                        fs = *fs_expr;
                    }
                } else if (!parse_number(trim_copy((*call_args)[1]), cutoff) ||
                           !parse_number(trim_copy((*call_args)[2]), fs)) {
                    auto c_expr = eval_scalar_expr(state_, trim_copy((*call_args)[1]));
                    if (!c_expr) {
                        return std::unexpected(c_expr.error());
                    }
                    cutoff = *c_expr;
                    auto fs_expr = eval_scalar_expr(state_, trim_copy((*call_args)[2]));
                    if (!fs_expr) {
                        return std::unexpected(fs_expr.error());
                    }
                    fs = *fs_expr;
                }
                Result<Matrix<double>> filtered;
                if (callee == "signal_butterworth") {
                    filtered = eval_signal_butterworth(*x_m, cutoff, fs);
                } else if (callee == "signal_highpass") {
                    filtered = eval_signal_highpass(*x_m, cutoff, fs);
                } else {
                    filtered = eval_signal_bandpass(*x_m, low, high, fs);
                }
                if (!filtered) {
                    return std::unexpected(filtered.error());
                }
                state_.matrices[lhs] = *filtered;
                std::ostringstream out;
                out << lhs << " =\n";
                print_matrix(out, *filtered);
                return out.str();
            }
            if (callee == "graph_astar") {
                const auto call_args = split_call_args(rhs);
                if (!call_args || call_args->size() != 4) {
                    return std::unexpected(DomainError{
                        "graph_astar", "expected graph_astar(A, source, target, h)"});
                }
                auto adj_m = eval_matrix_operand(trim_copy(call_args->front()));
                if (!adj_m) {
                    return std::unexpected(adj_m.error());
                }
                double source_d = 0.0;
                double target_d = 0.0;
                if (!parse_number(trim_copy((*call_args)[1]), source_d) ||
                    !parse_number(trim_copy((*call_args)[2]), target_d)) {
                    return std::unexpected(DomainError{
                        "graph_astar", "expected graph_astar(A, source, target, h)"});
                }
                const int source = static_cast<int>(source_d);
                const int target = static_cast<int>(target_d);
                if (source_d != source || target_d != target) {
                    return std::unexpected(DomainError{
                        "graph_astar", "expected integer source and target"});
                }
                auto h_m = eval_matrix_operand(trim_copy((*call_args)[3]));
                if (!h_m) {
                    return std::unexpected(h_m.error());
                }
                auto path = eval_graph_astar(*adj_m, source, target, *h_m);
                if (!path) {
                    return std::unexpected(path.error());
                }
                state_.matrices[lhs] = *path;
                std::ostringstream out;
                out << lhs << " =\n";
                print_matrix(out, *path);
                return out.str();
            }
            if (callee == "info_joint_entropy") {
                const auto call_args = split_call_args(rhs);
                if (!call_args || call_args->size() != 3) {
                    return std::unexpected(DomainError{
                        "info_joint_entropy",
                        "expected info_joint_entropy(joint, rows, cols)"});
                }
                auto joint_m = eval_matrix_operand(trim_copy(call_args->front()));
                if (!joint_m) {
                    return std::unexpected(joint_m.error());
                }
                double rows_d = 0.0;
                double cols_d = 0.0;
                if (!parse_number(trim_copy((*call_args)[1]), rows_d) ||
                    !parse_number(trim_copy((*call_args)[2]), cols_d)) {
                    return std::unexpected(DomainError{
                        "info_joint_entropy",
                        "expected info_joint_entropy(joint, rows, cols)"});
                }
                const int rows = static_cast<int>(rows_d);
                const int cols = static_cast<int>(cols_d);
                if (rows < 1 || cols < 1 || rows_d != rows || cols_d != cols) {
                    return std::unexpected(DomainError{
                        "info_joint_entropy", "expected positive integer rows and cols"});
                }
                auto value = eval_info_joint_entropy(*joint_m, rows, cols);
                if (!value) {
                    return std::unexpected(value.error());
                }
                return assign_scalar(lhs, *value);
            }
            if (callee == "cplx_power_series_eval" || callee == "cplx_winding_number" ||
                callee == "topo_vietoris_rips_betti0" || callee == "control_bode_mag_db" ||
                callee == "control_bode_phase" || callee == "topo_bottleneck_distance" ||
                callee == "topo_wasserstein_distance") {
                const auto call_args = split_call_args(rhs);
                if (!call_args || call_args->size() != 3) {
                    return std::unexpected(DomainError{
                        callee,
                        callee + "(...) expected three arguments"});
                }
                if (callee == "control_bode_mag_db" || callee == "control_bode_phase" ||
                    callee == "topo_bottleneck_distance" ||
                    callee == "topo_wasserstein_distance") {
                    auto arg_a_m = eval_matrix_operand(trim_copy(call_args->at(0)));
                    if (!arg_a_m) {
                        return std::unexpected(arg_a_m.error());
                    }
                    auto arg_b_m = eval_matrix_operand(trim_copy(call_args->at(1)));
                    if (!arg_b_m) {
                        return std::unexpected(arg_b_m.error());
                    }
                    double arg2 = 0.0;
                    if (!parse_number(trim_copy((*call_args)[2]), arg2)) {
                        auto w_expr = eval_scalar_expr(state_, trim_copy((*call_args)[2]));
                        if (!w_expr) {
                            return std::unexpected(w_expr.error());
                        }
                        arg2 = *w_expr;
                    }
                    Result<double> value =
                        std::unexpected(DomainError{callee, "unsupported matrix-scalar call"});
                    if (callee == "control_bode_mag_db") {
                        value = eval_control_bode_mag_db(*arg_a_m, *arg_b_m, arg2);
                    } else if (callee == "control_bode_phase") {
                        value = eval_control_bode_phase(*arg_a_m, *arg_b_m, arg2);
                    } else if (callee == "topo_wasserstein_distance") {
                        const int dim = static_cast<int>(arg2);
                        if (dim < 0 || arg2 != dim) {
                            return std::unexpected(DomainError{
                                "topo_wasserstein_distance",
                                "expected non-negative integer dim"});
                        }
                        value = eval_topo_wasserstein_distance(*arg_a_m, *arg_b_m, dim);
                    } else {
                        const int dim = static_cast<int>(arg2);
                        if (dim < 0 || arg2 != dim) {
                            return std::unexpected(DomainError{
                                "topo_bottleneck_distance",
                                "expected non-negative integer dim"});
                        }
                        value = eval_topo_bottleneck_distance(*arg_a_m, *arg_b_m, dim);
                    }
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    return assign_scalar(lhs, *value);
                }
                auto matrix_m = eval_matrix_operand(trim_copy(call_args->front()));
                if (!matrix_m) {
                    return std::unexpected(matrix_m.error());
                }
                double arg1 = 0.0;
                double arg2 = 0.0;
                if (!parse_number(trim_copy((*call_args)[1]), arg1) ||
                    !parse_number(trim_copy((*call_args)[2]), arg2)) {
                    auto s1 = eval_scalar_expr(state_, trim_copy((*call_args)[1]));
                    if (!s1) {
                        return std::unexpected(s1.error());
                    }
                    arg1 = *s1;
                    auto s2 = eval_scalar_expr(state_, trim_copy((*call_args)[2]));
                    if (!s2) {
                        return std::unexpected(s2.error());
                    }
                    arg2 = *s2;
                }
                Result<double> value =
                    std::unexpected(DomainError{callee, "unsupported matrix-scalar call"});
                if (callee == "cplx_power_series_eval") {
                    value = eval_cplx_power_series_eval(*matrix_m, arg1, arg2);
                } else if (callee == "cplx_winding_number") {
                    value = eval_cplx_winding_number(*matrix_m, arg1, arg2);
                } else {
                    const int max_dim = static_cast<int>(arg2);
                    if (max_dim < 0 || arg2 != max_dim) {
                        return std::unexpected(DomainError{
                            "topo_vietoris_rips_betti0",
                            "expected non-negative integer max_dim"});
                    }
                    value = eval_topo_vietoris_rips_betti0(*matrix_m, arg1, max_dim);
                }
                if (!value) {
                    return std::unexpected(value.error());
                }
                return assign_scalar(lhs, *value);
            }
            if (callee == "geo_point_in_polygon") {
                const auto call_args = split_call_args(rhs);
                if (!call_args || call_args->size() != 3) {
                    return std::unexpected(DomainError{
                        "geo_point_in_polygon",
                        "expected geo_point_in_polygon(px, py, P)"});
                }
                double px = 0.0;
                double py = 0.0;
                if (!parse_number(trim_copy((*call_args)[0]), px) ||
                    !parse_number(trim_copy((*call_args)[1]), py)) {
                    return std::unexpected(DomainError{
                        "geo_point_in_polygon",
                        "expected geo_point_in_polygon(px, py, P)"});
                }
                auto poly_m = eval_matrix_operand(trim_copy((*call_args)[2]));
                if (!poly_m) {
                    return std::unexpected(poly_m.error());
                }
                auto value = eval_geo_point_in_polygon(px, py, *poly_m);
                if (!value) {
                    return std::unexpected(value.error());
                }
                return assign_scalar(lhs, *value);
            }
            if (callee == "geo_kdtree_nearest") {
                const auto call_args = split_call_args(rhs);
                if (!call_args || call_args->size() != 3) {
                    return std::unexpected(DomainError{
                        "geo_kdtree_nearest",
                        "expected geo_kdtree_nearest(P, x, y)"});
                }
                auto P_m = eval_matrix_operand(trim_copy(call_args->front()));
                if (!P_m) {
                    return std::unexpected(P_m.error());
                }
                double qx = 0.0;
                double qy = 0.0;
                if (!parse_number(trim_copy((*call_args)[1]), qx) ||
                    !parse_number(trim_copy((*call_args)[2]), qy)) {
                    return std::unexpected(DomainError{
                        "geo_kdtree_nearest",
                        "expected geo_kdtree_nearest(P, x, y)"});
                }
                auto value = eval_geo_kdtree_nearest(*P_m, qx, qy);
                if (!value) {
                    return std::unexpected(value.error());
                }
                return assign_scalar(lhs, *value);
            }
            if (callee == "info_conditional_entropy") {
                const auto call_args = split_call_args(rhs);
                if (!call_args || call_args->size() != 3) {
                    return std::unexpected(DomainError{
                        "info_conditional_entropy",
                        "expected info_conditional_entropy(joint, rows, cols)"});
                }
                auto joint_m = eval_matrix_operand(trim_copy(call_args->front()));
                if (!joint_m) {
                    return std::unexpected(joint_m.error());
                }
                double rows_d = 0.0;
                double cols_d = 0.0;
                if (!parse_number(trim_copy((*call_args)[1]), rows_d) ||
                    !parse_number(trim_copy((*call_args)[2]), cols_d)) {
                    return std::unexpected(DomainError{
                        "info_conditional_entropy",
                        "expected info_conditional_entropy(joint, rows, cols)"});
                }
                const int rows = static_cast<int>(rows_d);
                const int cols = static_cast<int>(cols_d);
                if (rows < 1 || cols < 1 || rows_d != rows || cols_d != cols) {
                    return std::unexpected(DomainError{
                        "info_conditional_entropy", "expected positive integer rows and cols"});
                }
                auto value = eval_info_conditional_entropy(*joint_m, rows, cols);
                if (!value) {
                    return std::unexpected(value.error());
                }
                return assign_scalar(lhs, *value);
            }
            if (callee == "info_sample_entropy") {
                const auto call_args = split_call_args(rhs);
                if (!call_args || call_args->size() != 3) {
                    return std::unexpected(DomainError{
                        "info_sample_entropy",
                        "expected info_sample_entropy(x, m, r)"});
                }
                auto x_m = eval_matrix_operand(trim_copy(call_args->front()));
                if (!x_m) {
                    return std::unexpected(x_m.error());
                }
                double m_d = 0.0;
                double r = 0.0;
                if (!parse_number(trim_copy((*call_args)[1]), m_d) ||
                    !parse_number(trim_copy((*call_args)[2]), r)) {
                    return std::unexpected(DomainError{
                        "info_sample_entropy",
                        "expected info_sample_entropy(x, m, r)"});
                }
                const int m = static_cast<int>(m_d);
                if (m < 1 || m_d != m) {
                    return std::unexpected(DomainError{
                        "info_sample_entropy", "expected positive integer m"});
                }
                auto value = eval_info_sample_entropy(*x_m, m, r);
                if (!value) {
                    return std::unexpected(value.error());
                }
                return assign_scalar(lhs, *value);
            }
            if (callee == "stats_ztest") {
                const auto call_args = split_call_args(rhs);
                if (!call_args || call_args->size() != 3) {
                    return std::unexpected(DomainError{
                        "stats_ztest", "expected stats_ztest(x, mu, sigma)"});
                }
                auto x_m = eval_matrix_operand(trim_copy(call_args->front()));
                if (!x_m) {
                    return std::unexpected(x_m.error());
                }
                double mu = 0.0;
                double sigma = 0.0;
                if (!parse_number(trim_copy((*call_args)[1]), mu) ||
                    !parse_number(trim_copy((*call_args)[2]), sigma)) {
                    return std::unexpected(DomainError{
                        "stats_ztest", "expected stats_ztest(x, mu, sigma)"});
                }
                auto value = eval_stats_ztest(*x_m, mu, sigma);
                if (!value) {
                    return std::unexpected(value.error());
                }
                return assign_scalar(lhs, *value);
            }
            if (callee == "quantum_partial_trace") {
                const auto call_args = split_call_args(rhs);
                if (!call_args || call_args->size() != 4) {
                    return std::unexpected(DomainError{
                        "quantum_partial_trace",
                        "expected quantum_partial_trace(rho, d1, d2, subsystem)"});
                }
                auto rho_m = eval_matrix_operand(trim_copy(call_args->front()));
                if (!rho_m) {
                    return std::unexpected(rho_m.error());
                }
                double d1_d = 0.0;
                double d2_d = 0.0;
                double sub_d = 0.0;
                if (!parse_number(trim_copy((*call_args)[1]), d1_d) ||
                    !parse_number(trim_copy((*call_args)[2]), d2_d) ||
                    !parse_number(trim_copy((*call_args)[3]), sub_d)) {
                    return std::unexpected(DomainError{
                        "quantum_partial_trace",
                        "expected quantum_partial_trace(rho, d1, d2, subsystem)"});
                }
                const int d1 = static_cast<int>(d1_d);
                const int d2 = static_cast<int>(d2_d);
                const int subsystem = static_cast<int>(sub_d);
                if (d1 < 1 || d2 < 1 || d1_d != d1 || d2_d != d2 ||
                    (subsystem != 0 && subsystem != 1) || sub_d != subsystem) {
                    return std::unexpected(DomainError{
                        "quantum_partial_trace",
                        "expected positive integer d1, d2 and subsystem 0 or 1"});
                }
                auto result = eval_quantum_partial_trace_matrix(*rho_m, d1, d2, subsystem);
                if (!result) {
                    return std::unexpected(result.error());
                }
                state_.matrices[lhs] = *result;
                std::ostringstream out;
                out << lhs << " =\n";
                print_matrix(out, *result);
                return out.str();
            }
        }

        auto matrix = parse_matrix(rhs);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        state_.matrices[lhs] = *matrix;
        std::ostringstream out;
        out << lhs << " =\n";
        print_matrix(out, *matrix);
        return out.str();
    }

    if (const auto nullary = parse_nullary_matrix_call(cmd)) {
        auto value = eval_nullary_matrix_call(*nullary);
        if (!value) {
            return std::unexpected(value.error());
        }
        std::ostringstream out;
        out << "op =\n";
        print_matrix(out, *value);
        return out.str();
    }

    if (const auto unary = parse_unary_scalar_matrix_call(cmd)) {
        auto theta = eval_scalar_expr(state_, unary->second);
        if (!theta) {
            return std::unexpected(theta.error());
        }
        auto value = eval_unary_scalar_matrix_call(unary->first, *theta);
        if (!value) {
            return std::unexpected(value.error());
        }
        std::ostringstream out;
        if (unary->first == "quantum_ghz_state" || unary->first == "quantum_w_state" ||
            unary->first == "quantum_bell_state") {
            out << "state =\n";
            for (size_t i = 0; i < value->rows(); ++i) {
                out << "  [" << (*value)(i, 0) << "]\n";
            }
        } else {
            out << "op =\n";
            print_matrix(out, *value);
        }
        return out.str();
    }

    if (const auto open_paren = cmd.find('('); open_paren != std::string::npos && !cmd.empty() &&
                                     cmd.back() == ')') {
        const std::string fn = lower(trim_copy(cmd.substr(0, open_paren)));
        if (fn == "geo_volume_tetrahedron") {
            const auto call_args = split_call_args(cmd);
            if (!call_args || call_args->size() != 12) {
                return std::unexpected(DomainError{
                    fn,
                    "expected geo_volume_tetrahedron(x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4)"});
            }
            std::vector<double> args;
            args.reserve(12);
            for (const auto& arg_text : *call_args) {
                double v = 0.0;
                if (!parse_number(trim_copy(arg_text), v)) {
                    return std::unexpected(DomainError{fn, "expected numeric arguments"});
                }
                args.push_back(v);
            }
            auto value = eval_scalar_call(fn, args);
            if (!value) {
                return std::unexpected(value.error());
            }
            return std::to_string(*value) + "\n";
        }
        if (fn == "quantum_ket_basis" || fn == "quantum_fock_state" ||
            fn == "quantum_coherent_state" || fn == "diffgeo_surface_normal_sphere") {
            const auto call_args = split_call_args(cmd);
            if (fn == "quantum_coherent_state" && call_args && call_args->size() == 3) {
                double alpha_re = 0.0;
                double alpha_im = 0.0;
                double n_max_d = 0.0;
                if (!parse_number(trim_copy(call_args->at(0)), alpha_re) ||
                    !parse_number(trim_copy(call_args->at(1)), alpha_im) ||
                    !parse_number(trim_copy(call_args->at(2)), n_max_d)) {
                    return std::unexpected(
                        DomainError{fn, "expected numeric arguments"});
                }
                const int n_max = static_cast<int>(n_max_d);
                if (n_max < 0 || n_max_d != n_max) {
                    return std::unexpected(DomainError{
                        fn, "expected non-negative integer n_max"});
                }
                auto value = eval_quantum_coherent_state(alpha_re, alpha_im, n_max);
                if (!value) {
                    return std::unexpected(value.error());
                }
                std::ostringstream out;
                out << "state =\n";
                for (size_t i = 0; i < value->rows(); ++i) {
                    out << "  [" << (*value)(i, 0) << "]\n";
                }
                return out.str();
            }
            if (call_args && call_args->size() == 2) {
                double a = 0.0;
                double b = 0.0;
                if (!parse_number(trim_copy(call_args->at(0)), a) ||
                    !parse_number(trim_copy(call_args->at(1)), b)) {
                    return std::unexpected(
                        DomainError{fn, "expected numeric arguments"});
                }
                Result<Matrix<double>> value;
                if (fn == "diffgeo_surface_normal_sphere") {
                    value = eval_diffgeo_surface_normal_sphere(a, b);
                } else if (fn == "quantum_ket_basis") {
                    const int dim = static_cast<int>(a);
                    const int index = static_cast<int>(b);
                    if (dim < 1 || a != dim || b != index) {
                        return std::unexpected(DomainError{
                            fn, "expected positive integer dim and integer index"});
                    }
                    value = eval_quantum_ket_basis(dim, index);
                } else {
                    const int n = static_cast<int>(a);
                    const int n_max = static_cast<int>(b);
                    if (n_max < 0 || a != n || b != n_max) {
                        return std::unexpected(DomainError{
                            fn, "expected non-negative integer n and n_max"});
                    }
                    value = eval_quantum_fock_state(n, n_max);
                }
                if (!value) {
                    return std::unexpected(value.error());
                }
                std::ostringstream out;
                out << "state =\n";
                for (size_t i = 0; i < value->rows(); ++i) {
                    out << "  [" << (*value)(i, 0) << "]\n";
                }
                return out.str();
            }
        }
    }

    static const std::regex plot_binary(
        R"(plot\s*\(\s*(\[[^\]]+\])\s*,\s*(\[[^\]]+\])\s*\))",
        std::regex::icase);
    static const std::regex scatter_binary(
        R"(scatter\s*\(\s*(\[[^\]]+\])\s*,\s*(\[[^\]]+\])\s*\))",
        std::regex::icase);
    std::smatch match;
    if (std::regex_match(cmd, match, plot_binary)) {
        auto xs = parse_matrix(match[1].str());
        auto ys = parse_matrix(match[2].str());
        if (!xs || !ys) {
            return std::unexpected(DomainError{"plot", "invalid x/y vectors"});
        }
        return set_plot(*xs, *ys);
    }
    if (std::regex_match(cmd, match, scatter_binary)) {
        auto xs = parse_matrix(match[1].str());
        auto ys = parse_matrix(match[2].str());
        if (!xs || !ys) {
            return std::unexpected(DomainError{"scatter", "invalid x/y vectors"});
        }
        return set_plot(*xs, *ys, PlotSeries::Kind::Scatter);
    }

    static const std::regex octonary(
        R"((\w+)\(([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^)]+)\))",
        std::regex::icase);
    if (std::regex_match(cmd, match, octonary)) {
        const std::string fn = lower(match[1].str());
        if (fn == "cplx_cross_ratio") {
            double z1re = 0.0;
            double z1im = 0.0;
            double z2re = 0.0;
            double z2im = 0.0;
            double z3re = 0.0;
            double z3im = 0.0;
            double z4re = 0.0;
            double z4im = 0.0;
            if (!parse_number(trim(match[2].str()), z1re) || !parse_number(trim(match[3].str()), z1im) ||
                !parse_number(trim(match[4].str()), z2re) || !parse_number(trim(match[5].str()), z2im) ||
                !parse_number(trim(match[6].str()), z3re) || !parse_number(trim(match[7].str()), z3im) ||
                !parse_number(trim(match[8].str()), z4re) || !parse_number(trim(match[9].str()), z4im)) {
                return std::unexpected(
                    DomainError{"cplx_cross_ratio",
                                  "expected cplx_cross_ratio(z1re,z1im,z2re,z2im,z3re,z3im,z4re,z4im)"});
            }
            const cplx::C z1{z1re, z1im};
            const cplx::C z2{z2re, z2im};
            const cplx::C z3{z3re, z3im};
            const cplx::C z4{z4re, z4im};
            return std::to_string(cplx::cross_ratio(z1, z2, z3, z4)) + "\n";
        }
    }

    static const std::regex heptary(
        R"((\w+)\(([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^)]+)\))",
        std::regex::icase);
    if (std::regex_match(cmd, match, heptary)) {
        const std::string fn = lower(match[1].str());
        if (fn == "heun_g") {
            double a = 0.0;
            double q = 0.0;
            double alpha = 0.0;
            double beta = 0.0;
            double gamma = 0.0;
            double delta = 0.0;
            double z = 0.0;
            if (!parse_number(trim(match[2].str()), a) || !parse_number(trim(match[3].str()), q) ||
                !parse_number(trim(match[4].str()), alpha) || !parse_number(trim(match[5].str()), beta) ||
                !parse_number(trim(match[6].str()), gamma) || !parse_number(trim(match[7].str()), delta) ||
                !parse_number(trim(match[8].str()), z)) {
                return std::unexpected(
                    DomainError{"heun_g", "expected heun_g(a,q,alpha,beta,gamma,delta,z)"});
            }
            return std::to_string(heun_g(a, q, alpha, beta, gamma, delta, z)) + "\n";
        }
    }

    static const std::regex quinary(
        R"((\w+)\(([^,]+),([^,]+),([^,]+),([^,]+),([^)]+)\))", std::regex::icase);
    if (std::regex_match(cmd, match, quinary)) {
        const std::string fn = lower(match[1].str());
        if (fn == "finance_bs_call") {
            double S = 0.0;
            double K = 0.0;
            double T = 0.0;
            double r = 0.0;
            double sigma = 0.0;
            if (!parse_number(trim(match[2].str()), S) || !parse_number(trim(match[3].str()), K) ||
                !parse_number(trim(match[4].str()), T) || !parse_number(trim(match[5].str()), r) ||
                !parse_number(trim(match[6].str()), sigma)) {
                return std::unexpected(
                    DomainError{"finance_bs_call", "expected finance_bs_call(S,K,T,r,sigma)"});
            }
            return std::to_string(finance::bs_call(S, K, T, r, sigma)) + "\n";
        }
        if (fn == "finance_bs_put") {
            double S = 0.0;
            double K = 0.0;
            double T = 0.0;
            double r = 0.0;
            double sigma = 0.0;
            if (!parse_number(trim(match[2].str()), S) || !parse_number(trim(match[3].str()), K) ||
                !parse_number(trim(match[4].str()), T) || !parse_number(trim(match[5].str()), r) ||
                !parse_number(trim(match[6].str()), sigma)) {
                return std::unexpected(
                    DomainError{"finance_bs_put", "expected finance_bs_put(S,K,T,r,sigma)"});
            }
            return std::to_string(finance::bs_put(S, K, T, r, sigma)) + "\n";
        }
        if (fn == "finance_bs_gamma") {
            double S = 0.0;
            double K = 0.0;
            double T = 0.0;
            double r = 0.0;
            double sigma = 0.0;
            if (!parse_number(trim(match[2].str()), S) || !parse_number(trim(match[3].str()), K) ||
                !parse_number(trim(match[4].str()), T) || !parse_number(trim(match[5].str()), r) ||
                !parse_number(trim(match[6].str()), sigma)) {
                return std::unexpected(
                    DomainError{"finance_bs_gamma", "expected finance_bs_gamma(S,K,T,r,sigma)"});
            }
            return std::to_string(finance::bs_gamma(S, K, T, r, sigma)) + "\n";
        }
        if (fn == "finance_bs_vega") {
            double S = 0.0;
            double K = 0.0;
            double T = 0.0;
            double r = 0.0;
            double sigma = 0.0;
            if (!parse_number(trim(match[2].str()), S) || !parse_number(trim(match[3].str()), K) ||
                !parse_number(trim(match[4].str()), T) || !parse_number(trim(match[5].str()), r) ||
                !parse_number(trim(match[6].str()), sigma)) {
                return std::unexpected(
                    DomainError{"finance_bs_vega", "expected finance_bs_vega(S,K,T,r,sigma)"});
            }
            return std::to_string(finance::bs_vega(S, K, T, r, sigma)) + "\n";
        }
        if (fn == "geo_dist_point_line2d") {
            double px = 0.0;
            double py = 0.0;
            double a = 0.0;
            double b = 0.0;
            double c = 0.0;
            if (!parse_number(trim(match[2].str()), px) || !parse_number(trim(match[3].str()), py) ||
                !parse_number(trim(match[4].str()), a) || !parse_number(trim(match[5].str()), b) ||
                !parse_number(trim(match[6].str()), c)) {
                return std::unexpected(DomainError{
                    "geo_dist_point_line2d",
                    "expected geo_dist_point_line2d(px,py,a,b,c)"});
            }
            const geo::Point2D p{px, py};
            const geo::Line2D l{a, b, c};
            return std::to_string(geo::dist_point_line(p, l)) + "\n";
        }
        if (fn == "diffgeo_christoffel_sphere") {
            double k_d = 0.0;
            double i_d = 0.0;
            double j_d = 0.0;
            double u = 0.0;
            double v = 0.0;
            if (!parse_number(trim(match[2].str()), k_d) ||
                !parse_number(trim(match[3].str()), i_d) ||
                !parse_number(trim(match[4].str()), j_d) ||
                !parse_number(trim(match[5].str()), u) ||
                !parse_number(trim(match[6].str()), v)) {
                return std::unexpected(DomainError{
                    "diffgeo_christoffel_sphere",
                    "expected diffgeo_christoffel_sphere(k,i,j,u,v)"});
            }
            auto value = eval_diffgeo_christoffel_sphere(k_d, i_d, j_d, u, v);
            if (!value) {
                return std::unexpected(value.error());
            }
            return std::to_string(*value) + "\n";
        }
        if (fn == "pde_heat_1d" || fn == "pde_advection_1d" || fn == "pde_poisson_2d" ||
            fn == "pde_burgers_1d") {
            auto resolve_arg = [this](const std::string& text) -> Result<Matrix<double>> {
                auto matrix = parse_matrix(text);
                if (!matrix) {
                    matrix = resolve_matrix(text);
                }
                return matrix;
            };
            auto arg0_m = resolve_arg(trim(match[2].str()));
            if (!arg0_m) {
                return std::unexpected(arg0_m.error());
            }
            if (fn == "pde_heat_1d") {
                double alpha = 0.0;
                double dx = 0.0;
                double dt = 0.0;
                double steps_d = 0.0;
                if (!parse_number(trim(match[3].str()), alpha) ||
                    !parse_number(trim(match[4].str()), dx) ||
                    !parse_number(trim(match[5].str()), dt) ||
                    !parse_number(trim(match[6].str()), steps_d)) {
                    return std::unexpected(DomainError{
                        "pde_heat_1d", "expected pde_heat_1d(x0, alpha, dx, dt, steps)"});
                }
                const int steps_i = static_cast<int>(steps_d);
                if (steps_i < 0 || steps_d != steps_i) {
                    return std::unexpected(
                        DomainError{"pde_heat_1d", "expected non-negative integer steps"});
                }
                auto value = eval_pde_heat_1d(*arg0_m, alpha, dx, dt,
                                              static_cast<std::size_t>(steps_i));
                if (!value) {
                    return std::unexpected(value.error());
                }
                std::ostringstream out;
                out << "u =\n";
                print_matrix(out, *value);
                return out.str();
            }
            if (fn == "pde_advection_1d") {
                double v = 0.0;
                double dx = 0.0;
                double dt = 0.0;
                double steps_d = 0.0;
                if (!parse_number(trim(match[3].str()), v) ||
                    !parse_number(trim(match[4].str()), dx) ||
                    !parse_number(trim(match[5].str()), dt) ||
                    !parse_number(trim(match[6].str()), steps_d)) {
                    return std::unexpected(DomainError{
                        "pde_advection_1d",
                        "expected pde_advection_1d(u0, v, dx, dt, steps)"});
                }
                const int steps_i = static_cast<int>(steps_d);
                if (steps_i < 0 || steps_d != steps_i) {
                    return std::unexpected(
                        DomainError{"pde_advection_1d", "expected non-negative integer steps"});
                }
                auto value = eval_pde_advection_1d(*arg0_m, v, dx, dt,
                                                   static_cast<std::size_t>(steps_i));
                if (!value) {
                    return std::unexpected(value.error());
                }
                std::ostringstream out;
                out << "u =\n";
                print_matrix(out, *value);
                return out.str();
            }
            if (fn == "pde_poisson_2d") {
                double dx = 0.0;
                double dy = 0.0;
                double max_iter_d = 0.0;
                double tolerance = 0.0;
                if (!parse_number(trim(match[3].str()), dx) ||
                    !parse_number(trim(match[4].str()), dy) ||
                    !parse_number(trim(match[5].str()), max_iter_d) ||
                    !parse_number(trim(match[6].str()), tolerance)) {
                    return std::unexpected(DomainError{
                        "pde_poisson_2d",
                        "expected pde_poisson_2d(f, dx, dy, max_iterations, tolerance)"});
                }
                const int max_iter_i = static_cast<int>(max_iter_d);
                if (max_iter_i < 0 || max_iter_d != max_iter_i) {
                    return std::unexpected(DomainError{
                        "pde_poisson_2d", "expected non-negative integer max_iterations"});
                }
                auto value = eval_pde_poisson_2d(*arg0_m, dx, dy,
                                                 static_cast<std::size_t>(max_iter_i),
                                                 tolerance);
                if (!value) {
                    return std::unexpected(value.error());
                }
                std::ostringstream out;
                out << "u =\n";
                print_matrix(out, *value);
                return out.str();
            }
            double nu = 0.0;
            double dx = 0.0;
            double dt = 0.0;
            double steps_d = 0.0;
            if (!parse_number(trim(match[3].str()), nu) ||
                !parse_number(trim(match[4].str()), dx) ||
                !parse_number(trim(match[5].str()), dt) ||
                !parse_number(trim(match[6].str()), steps_d)) {
                return std::unexpected(DomainError{
                    "pde_burgers_1d", "expected pde_burgers_1d(u0, nu, dx, dt, steps)"});
            }
            const int steps_i = static_cast<int>(steps_d);
            if (steps_i < 0 || steps_d != steps_i) {
                return std::unexpected(
                    DomainError{"pde_burgers_1d", "expected non-negative integer steps"});
            }
            auto value = eval_pde_burgers_1d(*arg0_m, nu, dx, dt,
                                             static_cast<std::size_t>(steps_i));
            if (!value) {
                return std::unexpected(value.error());
            }
            std::ostringstream out;
            out << "u =\n";
            print_matrix(out, *value);
            return out.str();
        }
        if (fn == "ode_euler") {
            auto value = eval_ode_fixed_step_call(fn, trim(match[2].str()), trim(match[3].str()),
                                                  trim(match[4].str()), trim(match[5].str()),
                                                  trim(match[6].str()), ode_euler);
            if (!value) {
                return std::unexpected(value.error());
            }
            return *value;
        }
        if (fn == "ode_rk4") {
            auto value = eval_ode_fixed_step_call(fn, trim(match[2].str()), trim(match[3].str()),
                                                  trim(match[4].str()), trim(match[5].str()),
                                                  trim(match[6].str()), ode_rk4);
            if (!value) {
                return std::unexpected(value.error());
            }
            return *value;
        }
        if (fn == "ode_midpoint") {
            auto value = eval_ode_fixed_step_call(fn, trim(match[2].str()), trim(match[3].str()),
                                                  trim(match[4].str()), trim(match[5].str()),
                                                  trim(match[6].str()), ode_midpoint);
            if (!value) {
                return std::unexpected(value.error());
            }
            return *value;
        }
        if (fn == "ode_backward_euler") {
            auto value = eval_ode_fixed_step_call(fn, trim(match[2].str()), trim(match[3].str()),
                                                  trim(match[4].str()), trim(match[5].str()),
                                                  trim(match[6].str()), ode_backward_euler);
            if (!value) {
                return std::unexpected(value.error());
            }
            return *value;
        }
    }

    static const std::regex senary(
        R"((\w+)\(([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^)]+)\))", std::regex::icase);
    if (std::regex_match(cmd, match, senary)) {
        const std::string fn = lower(match[1].str());
        if (fn == "ode_rk45") {
            auto value = eval_ode_rk45_call(trim(match[2].str()), trim(match[3].str()),
                                            trim(match[4].str()), trim(match[5].str()),
                                            trim(match[6].str()), trim(match[7].str()));
            if (!value) {
                return std::unexpected(value.error());
            }
            return *value;
        }
        if (fn == "geo_dist3d") {
            double x1 = 0.0;
            double y1 = 0.0;
            double z1 = 0.0;
            double x2 = 0.0;
            double y2 = 0.0;
            double z2 = 0.0;
            if (!parse_number(trim(match[2].str()), x1) || !parse_number(trim(match[3].str()), y1) ||
                !parse_number(trim(match[4].str()), z1) || !parse_number(trim(match[5].str()), x2) ||
                !parse_number(trim(match[6].str()), y2) ||
                !parse_number(trim(match[7].str()), z2)) {
                return std::unexpected(
                    DomainError{"geo_dist3d", "expected geo_dist3d(x1,y1,z1,x2,y2,z2)"});
            }
            const geo::Point3D p0{x1, y1, z1};
            const geo::Point3D p1{x2, y2, z2};
            return std::to_string(geo::dist(p0, p1)) + "\n";
        }
        if (fn == "geo_triangle_area") {
            double x1 = 0.0;
            double y1 = 0.0;
            double x2 = 0.0;
            double y2 = 0.0;
            double x3 = 0.0;
            double y3 = 0.0;
            if (!parse_number(trim(match[2].str()), x1) || !parse_number(trim(match[3].str()), y1) ||
                !parse_number(trim(match[4].str()), x2) || !parse_number(trim(match[5].str()), y2) ||
                !parse_number(trim(match[6].str()), x3) ||
                !parse_number(trim(match[7].str()), y3)) {
                return std::unexpected(DomainError{
                    "geo_triangle_area",
                    "expected geo_triangle_area(x1,y1,x2,y2,x3,y3)"});
            }
            const geo::Point2D p0{x1, y1};
            const geo::Point2D p1{x2, y2};
            const geo::Point2D p2{x3, y3};
            return std::to_string(geo::area(p0, p1, p2)) + "\n";
        }
        if (fn == "cplx_mobius_re") {
            double a = 0.0;
            double b = 0.0;
            double c = 0.0;
            double d = 0.0;
            double z_re = 0.0;
            double z_im = 0.0;
            if (!parse_number(trim(match[2].str()), a) || !parse_number(trim(match[3].str()), b) ||
                !parse_number(trim(match[4].str()), c) || !parse_number(trim(match[5].str()), d) ||
                !parse_number(trim(match[6].str()), z_re) ||
                !parse_number(trim(match[7].str()), z_im)) {
                return std::unexpected(DomainError{
                    "cplx_mobius_re",
                    "expected cplx_mobius_re(a,b,c,d,zre,zim)"});
            }
            auto value = eval_cplx_mobius_re(a, b, c, d, z_re, z_im);
            if (!value) {
                return std::unexpected(value.error());
            }
            return std::to_string(*value) + "\n";
        }
        if (fn == "geo_dist_point_seg2d") {
            double px = 0.0;
            double py = 0.0;
            double x1 = 0.0;
            double y1 = 0.0;
            double x2 = 0.0;
            double y2 = 0.0;
            if (!parse_number(trim(match[2].str()), px) || !parse_number(trim(match[3].str()), py) ||
                !parse_number(trim(match[4].str()), x1) || !parse_number(trim(match[5].str()), y1) ||
                !parse_number(trim(match[6].str()), x2) ||
                !parse_number(trim(match[7].str()), y2)) {
                return std::unexpected(DomainError{
                    "geo_dist_point_seg2d",
                    "expected geo_dist_point_seg2d(px,py,x1,y1,x2,y2)"});
            }
            const geo::Point2D p{px, py};
            const geo::Segment2D s{{x1, y1}, {x2, y2}};
            return std::to_string(geo::dist_point_segment(p, s)) + "\n";
        }
        if (fn == "geo_overlap_circles") {
            double x1 = 0.0;
            double y1 = 0.0;
            double r1 = 0.0;
            double x2 = 0.0;
            double y2 = 0.0;
            double r2 = 0.0;
            if (!parse_number(trim(match[2].str()), x1) || !parse_number(trim(match[3].str()), y1) ||
                !parse_number(trim(match[4].str()), r1) || !parse_number(trim(match[5].str()), x2) ||
                !parse_number(trim(match[6].str()), y2) ||
                !parse_number(trim(match[7].str()), r2)) {
                return std::unexpected(DomainError{
                    "geo_overlap_circles",
                    "expected geo_overlap_circles(x1,y1,r1,x2,y2,r2)"});
            }
            const geo::Circle2D a{{x1, y1}, r1};
            const geo::Circle2D b{{x2, y2}, r2};
            return std::to_string(geo::overlap_circle_circle(a, b) ? 1.0 : 0.0) + "\n";
        }
        if (fn == "finance_binomial_call") {
            double S = 0.0;
            double K = 0.0;
            double T = 0.0;
            double r = 0.0;
            double sigma = 0.0;
            double steps_d = 0.0;
            if (!parse_number(trim(match[2].str()), S) || !parse_number(trim(match[3].str()), K) ||
                !parse_number(trim(match[4].str()), T) || !parse_number(trim(match[5].str()), r) ||
                !parse_number(trim(match[6].str()), sigma) ||
                !parse_number(trim(match[7].str()), steps_d)) {
                return std::unexpected(DomainError{
                    "finance_binomial_call",
                    "expected finance_binomial_call(S,K,T,r,sigma,steps)"});
            }
            const int steps = static_cast<int>(steps_d);
            if (steps < 0 || steps_d != steps) {
                return std::unexpected(
                    DomainError{"finance_binomial_call", "expected non-negative integer steps"});
            }
            return std::to_string(finance::binomial_call(S, K, T, r, sigma, steps)) + "\n";
        }
        if (fn == "finance_binomial_put") {
            double S = 0.0;
            double K = 0.0;
            double T = 0.0;
            double r = 0.0;
            double sigma = 0.0;
            double steps_d = 0.0;
            if (!parse_number(trim(match[2].str()), S) || !parse_number(trim(match[3].str()), K) ||
                !parse_number(trim(match[4].str()), T) || !parse_number(trim(match[5].str()), r) ||
                !parse_number(trim(match[6].str()), sigma) ||
                !parse_number(trim(match[7].str()), steps_d)) {
                return std::unexpected(DomainError{
                    "finance_binomial_put",
                    "expected finance_binomial_put(S,K,T,r,sigma,steps)"});
            }
            const int steps = static_cast<int>(steps_d);
            if (steps < 0 || steps_d != steps) {
                return std::unexpected(
                    DomainError{"finance_binomial_put", "expected non-negative integer steps"});
            }
            return std::to_string(finance::binomial_put(S, K, T, r, sigma, steps)) + "\n";
        }
        if (fn == "finance_bs_delta") {
            double S = 0.0;
            double K = 0.0;
            double T = 0.0;
            double r = 0.0;
            double sigma = 0.0;
            double call_d = 0.0;
            if (!parse_number(trim(match[2].str()), S) || !parse_number(trim(match[3].str()), K) ||
                !parse_number(trim(match[4].str()), T) || !parse_number(trim(match[5].str()), r) ||
                !parse_number(trim(match[6].str()), sigma) ||
                !parse_number(trim(match[7].str()), call_d)) {
                return std::unexpected(DomainError{
                    "finance_bs_delta", "expected finance_bs_delta(S,K,T,r,sigma,call)"});
            }
            const int call = static_cast<int>(call_d);
            if (call_d != call) {
                return std::unexpected(
                    DomainError{"finance_bs_delta", "expected integer call (0=put, 1=call)"});
            }
            return std::to_string(finance::bs_delta(S, K, T, r, sigma, call != 0)) + "\n";
        }
        if (fn == "finance_bs_implied_vol") {
            double price = 0.0;
            double S = 0.0;
            double K = 0.0;
            double T = 0.0;
            double r = 0.0;
            double call_d = 0.0;
            if (!parse_number(trim(match[2].str()), price) ||
                !parse_number(trim(match[3].str()), S) || !parse_number(trim(match[4].str()), K) ||
                !parse_number(trim(match[5].str()), T) || !parse_number(trim(match[6].str()), r) ||
                !parse_number(trim(match[7].str()), call_d)) {
                return std::unexpected(DomainError{
                    "finance_bs_implied_vol",
                    "expected finance_bs_implied_vol(price,S,K,T,r,call)"});
            }
            auto iv = eval_finance_bs_implied_vol(price, S, K, T, r, call_d);
            if (!iv) {
                return std::unexpected(iv.error());
            }
            return std::to_string(*iv) + "\n";
        }
        if (fn == "finance_bs_theta") {
            double S = 0.0;
            double K = 0.0;
            double T = 0.0;
            double r = 0.0;
            double sigma = 0.0;
            double call_d = 0.0;
            if (!parse_number(trim(match[2].str()), S) || !parse_number(trim(match[3].str()), K) ||
                !parse_number(trim(match[4].str()), T) || !parse_number(trim(match[5].str()), r) ||
                !parse_number(trim(match[6].str()), sigma) ||
                !parse_number(trim(match[7].str()), call_d)) {
                return std::unexpected(DomainError{
                    "finance_bs_theta", "expected finance_bs_theta(S,K,T,r,sigma,call)"});
            }
            const int call = static_cast<int>(call_d);
            if (call_d != call) {
                return std::unexpected(
                    DomainError{"finance_bs_theta", "expected integer call (0=put, 1=call)"});
            }
            return std::to_string(finance::bs_theta(S, K, T, r, sigma, call != 0)) + "\n";
        }
        if (fn == "finance_bs_rho") {
            double S = 0.0;
            double K = 0.0;
            double T = 0.0;
            double r = 0.0;
            double sigma = 0.0;
            double call_d = 0.0;
            if (!parse_number(trim(match[2].str()), S) || !parse_number(trim(match[3].str()), K) ||
                !parse_number(trim(match[4].str()), T) || !parse_number(trim(match[5].str()), r) ||
                !parse_number(trim(match[6].str()), sigma) ||
                !parse_number(trim(match[7].str()), call_d)) {
                return std::unexpected(DomainError{
                    "finance_bs_rho", "expected finance_bs_rho(S,K,T,r,sigma,call)"});
            }
            const int call = static_cast<int>(call_d);
            if (call_d != call) {
                return std::unexpected(
                    DomainError{"finance_bs_rho", "expected integer call (0=put, 1=call)"});
            }
            return std::to_string(finance::bs_rho(S, K, T, r, sigma, call != 0)) + "\n";
        }
        if (fn == "pde_heat_2d" || fn == "pde_wave_1d") {
            auto resolve_arg = [this](const std::string& text) -> Result<Matrix<double>> {
                auto matrix = parse_matrix(text);
                if (!matrix) {
                    matrix = resolve_matrix(text);
                }
                return matrix;
            };
            auto arg0_m = resolve_arg(trim(match[2].str()));
            if (!arg0_m) {
                return std::unexpected(arg0_m.error());
            }
            if (fn == "pde_heat_2d") {
                double alpha = 0.0;
                double dx = 0.0;
                double dy = 0.0;
                double dt = 0.0;
                double steps_d = 0.0;
                if (!parse_number(trim(match[3].str()), alpha) ||
                    !parse_number(trim(match[4].str()), dx) ||
                    !parse_number(trim(match[5].str()), dy) ||
                    !parse_number(trim(match[6].str()), dt) ||
                    !parse_number(trim(match[7].str()), steps_d)) {
                    return std::unexpected(DomainError{
                        "pde_heat_2d", "expected pde_heat_2d(u0, alpha, dx, dy, dt, steps)"});
                }
                const int steps_i = static_cast<int>(steps_d);
                if (steps_i < 0 || steps_d != steps_i) {
                    return std::unexpected(
                        DomainError{"pde_heat_2d", "expected non-negative integer steps"});
                }
                auto value = eval_pde_heat_2d(*arg0_m, alpha, dx, dy, dt,
                                              static_cast<std::size_t>(steps_i));
                if (!value) {
                    return std::unexpected(value.error());
                }
                std::ostringstream out;
                out << "u =\n";
                print_matrix(out, *value);
                return out.str();
            }
            auto v0_m = resolve_arg(trim(match[3].str()));
            if (!v0_m) {
                return std::unexpected(v0_m.error());
            }
            double c = 0.0;
            double dx = 0.0;
            double dt = 0.0;
            double steps_d = 0.0;
            if (!parse_number(trim(match[4].str()), c) ||
                !parse_number(trim(match[5].str()), dx) ||
                !parse_number(trim(match[6].str()), dt) ||
                !parse_number(trim(match[7].str()), steps_d)) {
                return std::unexpected(DomainError{
                    "pde_wave_1d", "expected pde_wave_1d(u0, v0, c, dx, dt, steps)"});
            }
            const int steps_i = static_cast<int>(steps_d);
            if (steps_i < 0 || steps_d != steps_i) {
                return std::unexpected(
                    DomainError{"pde_wave_1d", "expected non-negative integer steps"});
            }
            auto value = eval_pde_wave_1d(*arg0_m, *v0_m, c, dx, dt,
                                          static_cast<std::size_t>(steps_i));
            if (!value) {
                return std::unexpected(value.error());
            }
            std::ostringstream out;
            out << "u =\n";
            print_matrix(out, *value);
            return out.str();
        }
    }

    static const std::regex quaternary(
        R"((\w+)\(([^,]+),([^,]+),([^,]+),([^)]+)\))", std::regex::icase);
    if (std::regex_match(cmd, match, quaternary)) {
        const std::string fn = lower(match[1].str());
        if (fn == "geo_dist2d") {
            double x1 = 0.0;
            double y1 = 0.0;
            double x2 = 0.0;
            double y2 = 0.0;
            if (!parse_number(trim(match[2].str()), x1) || !parse_number(trim(match[3].str()), y1) ||
                !parse_number(trim(match[4].str()), x2) || !parse_number(trim(match[5].str()), y2)) {
                return std::unexpected(
                    DomainError{"geo_dist2d", "expected geo_dist2d(x1,y1,x2,y2)"});
            }
            const geo::Point2D p0{x1, y1};
            const geo::Point2D p1{x2, y2};
            return std::to_string(geo::dist(p0, p1)) + "\n";
        }
        if (fn == "geo_dist_sq2d") {
            double x1 = 0.0;
            double y1 = 0.0;
            double x2 = 0.0;
            double y2 = 0.0;
            if (!parse_number(trim(match[2].str()), x1) || !parse_number(trim(match[3].str()), y1) ||
                !parse_number(trim(match[4].str()), x2) || !parse_number(trim(match[5].str()), y2)) {
                return std::unexpected(
                    DomainError{"geo_dist_sq2d", "expected geo_dist_sq2d(x1,y1,x2,y2)"});
            }
            const geo::Point2D p0{x1, y1};
            const geo::Point2D p1{x2, y2};
            return std::to_string(geo::dist_sq(p0, p1)) + "\n";
        }
        if (fn == "geo_cross2d") {
            double x1 = 0.0;
            double y1 = 0.0;
            double x2 = 0.0;
            double y2 = 0.0;
            if (!parse_number(trim(match[2].str()), x1) || !parse_number(trim(match[3].str()), y1) ||
                !parse_number(trim(match[4].str()), x2) || !parse_number(trim(match[5].str()), y2)) {
                return std::unexpected(
                    DomainError{"geo_cross2d", "expected geo_cross2d(x1,y1,x2,y2)"});
            }
            const geo::Vec2D a{x1, y1};
            const geo::Vec2D b{x2, y2};
            return std::to_string(geo::cross2d(a, b)) + "\n";
        }
        if (fn == "cplx_hyperbolic_distance") {
            double z1re = 0.0;
            double z1im = 0.0;
            double z2re = 0.0;
            double z2im = 0.0;
            if (!parse_number(trim(match[2].str()), z1re) ||
                !parse_number(trim(match[3].str()), z1im) ||
                !parse_number(trim(match[4].str()), z2re) ||
                !parse_number(trim(match[5].str()), z2im)) {
                return std::unexpected(DomainError{
                    "cplx_hyperbolic_distance",
                    "expected cplx_hyperbolic_distance(z1re,z1im,z2re,z2im)"});
            }
            return std::to_string(
                       cplx::hyperbolic_distance(cplx::C{z1re, z1im}, cplx::C{z2re, z2im})) +
                   "\n";
        }
        if (fn == "finance_pv") {
            double rate = 0.0;
            double n_d = 0.0;
            double pmt = 0.0;
            double fv = 0.0;
            if (!parse_number(trim(match[2].str()), rate) || !parse_number(trim(match[3].str()), n_d) ||
                !parse_number(trim(match[4].str()), pmt) ||
                !parse_number(trim(match[5].str()), fv)) {
                return std::unexpected(
                    DomainError{"finance_pv", "expected finance_pv(rate,n,pmt,fv)"});
            }
            const int n = static_cast<int>(n_d);
            if (n < 0 || n_d != n) {
                return std::unexpected(
                    DomainError{"finance_pv", "expected non-negative integer n"});
            }
            return std::to_string(finance::pv(rate, n, pmt, fv)) + "\n";
        }
        if (fn == "finance_bond_price") {
            double c = 0.0;
            double y = 0.0;
            double n_d = 0.0;
            double fv = 100.0;
            if (!parse_number(trim(match[2].str()), c) || !parse_number(trim(match[3].str()), y) ||
                !parse_number(trim(match[4].str()), n_d) ||
                !parse_number(trim(match[5].str()), fv)) {
                return std::unexpected(
                    DomainError{"finance_bond_price", "expected finance_bond_price(c,y,n,fv)"});
            }
            const int n = static_cast<int>(n_d);
            if (n < 0 || n_d != n) {
                return std::unexpected(
                    DomainError{"finance_bond_price", "expected non-negative integer periods n"});
            }
            return std::to_string(finance::bond_price(c, y, n, fv)) + "\n";
        }
        if (fn == "finance_compound") {
            double principal = 0.0;
            double rate = 0.0;
            double n_d = 0.0;
            double cpp_d = 0.0;
            if (!parse_number(trim(match[2].str()), principal) ||
                !parse_number(trim(match[3].str()), rate) ||
                !parse_number(trim(match[4].str()), n_d) ||
                !parse_number(trim(match[5].str()), cpp_d)) {
                return std::unexpected(DomainError{
                    "finance_compound",
                    "expected finance_compound(principal,rate,n_periods,compounds_per_period)"});
            }
            const int n = static_cast<int>(n_d);
            const int cpp = static_cast<int>(cpp_d);
            if (n < 0 || n_d != n) {
                return std::unexpected(
                    DomainError{"finance_compound", "expected non-negative integer periods n_periods"});
            }
            if (cpp < 1 || cpp_d != cpp) {
                return std::unexpected(DomainError{
                    "finance_compound", "expected positive integer compounds_per_period"});
            }
            return std::to_string(finance::compound(principal, rate, n, cpp)) + "\n";
        }
        if (fn == "finance_fv_annuity") {
            double rate = 0.0;
            double n_d = 0.0;
            double pmt = 0.0;
            double pv0 = 0.0;
            if (!parse_number(trim(match[2].str()), rate) || !parse_number(trim(match[3].str()), n_d) ||
                !parse_number(trim(match[4].str()), pmt) ||
                !parse_number(trim(match[5].str()), pv0)) {
                return std::unexpected(
                    DomainError{"finance_fv_annuity", "expected finance_fv_annuity(rate,n,pmt,pv0)"});
            }
            const int n = static_cast<int>(n_d);
            if (n < 0 || n_d != n) {
                return std::unexpected(
                    DomainError{"finance_fv_annuity", "expected non-negative integer n"});
            }
            return std::to_string(finance::fv_annuity(rate, n, pmt, pv0)) + "\n";
        }
        if (fn == "finance_pmt_annuity") {
            double rate = 0.0;
            double n_d = 0.0;
            double pv0 = 0.0;
            double fv = 0.0;
            if (!parse_number(trim(match[2].str()), rate) || !parse_number(trim(match[3].str()), n_d) ||
                !parse_number(trim(match[4].str()), pv0) ||
                !parse_number(trim(match[5].str()), fv)) {
                return std::unexpected(
                    DomainError{"finance_pmt_annuity", "expected finance_pmt_annuity(rate,n,pv0,fv)"});
            }
            const int n = static_cast<int>(n_d);
            if (n < 0 || n_d != n) {
                return std::unexpected(
                    DomainError{"finance_pmt_annuity", "expected non-negative integer n"});
            }
            return std::to_string(finance::pmt_annuity(rate, n, pv0, fv)) + "\n";
        }
        if (fn == "hypergeo_2f1" || fn == "jacobi_p") {
            double a = 0.0;
            double b = 0.0;
            double c = 0.0;
            double d = 0.0;
            if (!parse_number(trim(match[2].str()), a) || !parse_number(trim(match[3].str()), b) ||
                !parse_number(trim(match[4].str()), c) || !parse_number(trim(match[5].str()), d)) {
                return std::unexpected(DomainError{fn, "expected numeric arguments"});
            }
            if (fn == "hypergeo_2f1") {
                return std::to_string(hypergeo_2f1(a, b, c, d)) + "\n";
            }
            return std::to_string(jacobi_p(static_cast<int>(a), b, c, d)) + "\n";
        }
        if (fn == "signal_bandpass") {
            auto resolve_arg = [this](const std::string& text) -> Result<Matrix<double>> {
                auto matrix = parse_matrix(text);
                if (!matrix) {
                    matrix = resolve_matrix(text);
                }
                return matrix;
            };
            auto x_m = resolve_arg(trim(match[2].str()));
            if (!x_m) {
                return std::unexpected(x_m.error());
            }
            double low = 0.0;
            double high = 0.0;
            double fs = 0.0;
            if (!parse_number(trim(match[3].str()), low) ||
                !parse_number(trim(match[4].str()), high) ||
                !parse_number(trim(match[5].str()), fs)) {
                return std::unexpected(
                    DomainError{"signal_bandpass", "expected signal_bandpass(x, low, high, fs)"});
            }
            auto filtered = eval_signal_bandpass(*x_m, low, high, fs);
            if (!filtered) {
                return std::unexpected(filtered.error());
            }
            std::ostringstream out;
            out << "filtered =\n";
            print_matrix(out, *filtered);
            return out.str();
        }
        if (fn == "graph_astar") {
            auto resolve_arg = [this](const std::string& text) -> Result<Matrix<double>> {
                auto matrix = parse_matrix(text);
                if (!matrix) {
                    matrix = resolve_matrix(text);
                }
                return matrix;
            };
            auto adj_m = resolve_arg(trim(match[2].str()));
            if (!adj_m) {
                return std::unexpected(adj_m.error());
            }
            double source_d = 0.0;
            double target_d = 0.0;
            if (!parse_number(trim(match[3].str()), source_d) ||
                !parse_number(trim(match[4].str()), target_d)) {
                return std::unexpected(
                    DomainError{"graph_astar", "expected graph_astar(A, source, target, h)"});
            }
            const int source = static_cast<int>(source_d);
            const int target = static_cast<int>(target_d);
            if (source_d != source || target_d != target) {
                return std::unexpected(
                    DomainError{"graph_astar", "expected integer source and target"});
            }
            auto h_m = resolve_arg(trim(match[5].str()));
            if (!h_m) {
                return std::unexpected(h_m.error());
            }
            auto path = eval_graph_astar(*adj_m, source, target, *h_m);
            if (!path) {
                return std::unexpected(path.error());
            }
            std::ostringstream out;
            out << "path =\n";
            print_matrix(out, *path);
            return out.str();
        }
    }

    static const std::regex ternary(R"((\w+)\(([^,]+),([^,]+),([^)]+)\))", std::regex::icase);
    if (lower(cmd).starts_with("surf(") && !cmd.empty() && cmd.back() == ')') {
        const auto args = split_call_args(cmd);
        if (args && (args->size() == 1u || args->size() == 3u)) {
            const auto resolve_arg = [this](const std::string& arg) -> Result<Matrix<double>> {
                auto matrix = parse_matrix(arg);
                if (!matrix) {
                    matrix = resolve_matrix(arg);
                }
                return matrix;
            };
            if (args->size() == 1u) {
                auto z = resolve_arg((*args)[0]);
                if (!z) {
                    return std::unexpected(z.error());
                }
                return set_plot_surf(*z);
            }
            auto X = resolve_arg((*args)[0]);
            if (!X) {
                return std::unexpected(X.error());
            }
            auto Y = resolve_arg((*args)[1]);
            if (!Y) {
                return std::unexpected(Y.error());
            }
            auto Z = resolve_arg((*args)[2]);
            if (!Z) {
                return std::unexpected(Z.error());
            }
            return set_plot_surf(*X, *Y, *Z);
        }
    }
    if (std::regex_match(cmd, match, ternary)) {
        const std::string fn = lower(match[1].str());
        if (fn == "graph_dijkstra_dist") {
            auto resolve_arg = [this](const std::string& text) -> Result<Matrix<double>> {
                auto matrix = parse_matrix(text);
                if (!matrix) {
                    matrix = resolve_matrix(text);
                }
                return matrix;
            };
            auto adj_m = resolve_arg(trim(match[2].str()));
            if (!adj_m) {
                return std::unexpected(adj_m.error());
            }
            double source_d = 0.0;
            double target_d = 0.0;
            if (!parse_number(trim(match[3].str()), source_d) ||
                !parse_number(trim(match[4].str()), target_d)) {
                return std::unexpected(
                    DomainError{"graph_dijkstra_dist", "expected graph_dijkstra_dist(A, source, target)"});
            }
            auto G = graph_from_adjacency(*adj_m, "graph_dijkstra_dist");
            if (!G) {
                return std::unexpected(G.error());
            }
            const int source = static_cast<int>(source_d);
            const int target = static_cast<int>(target_d);
            if (source < 0 || target < 0 || source >= G->n_vertices() || target >= G->n_vertices()) {
                return std::unexpected(
                    DomainError{"graph_dijkstra_dist", "source/target out of range"});
            }
            const auto [dist, parent] = graph::dijkstra(*G, source);
            (void)parent;
            return std::to_string(dist[static_cast<size_t>(target)]) + "\n";
        }
        if (fn == "graph_bellman_ford_dist") {
            auto resolve_arg = [this](const std::string& text) -> Result<Matrix<double>> {
                auto matrix = parse_matrix(text);
                if (!matrix) {
                    matrix = resolve_matrix(text);
                }
                return matrix;
            };
            auto adj_m = resolve_arg(trim(match[2].str()));
            if (!adj_m) {
                return std::unexpected(adj_m.error());
            }
            double source_d = 0.0;
            double target_d = 0.0;
            if (!parse_number(trim(match[3].str()), source_d) ||
                !parse_number(trim(match[4].str()), target_d)) {
                return std::unexpected(DomainError{
                    "graph_bellman_ford_dist",
                    "expected graph_bellman_ford_dist(A, source, target)"});
            }
            auto G = graph_from_adjacency(*adj_m, "graph_bellman_ford_dist");
            if (!G) {
                return std::unexpected(G.error());
            }
            const int source = static_cast<int>(source_d);
            const int target = static_cast<int>(target_d);
            if (source < 0 || target < 0 || source >= G->n_vertices() || target >= G->n_vertices()) {
                return std::unexpected(
                    DomainError{"graph_bellman_ford_dist", "source/target out of range"});
            }
            const auto result = graph::bellman_ford(*G, source);
            if (!result) {
                return std::unexpected(result.error());
            }
            const auto& [dist, parent] = *result;
            (void)parent;
            return std::to_string(dist[static_cast<size_t>(target)]) + "\n";
        }
        if (fn == "graph_max_flow") {
            auto resolve_arg = [this](const std::string& text) -> Result<Matrix<double>> {
                auto matrix = parse_matrix(text);
                if (!matrix) {
                    matrix = resolve_matrix(text);
                }
                return matrix;
            };
            auto adj_m = resolve_arg(trim(match[2].str()));
            if (!adj_m) {
                return std::unexpected(adj_m.error());
            }
            double source_d = 0.0;
            double sink_d = 0.0;
            if (!parse_number(trim(match[3].str()), source_d) ||
                !parse_number(trim(match[4].str()), sink_d)) {
                return std::unexpected(
                    DomainError{"graph_max_flow", "expected graph_max_flow(A, source, sink)"});
            }
            const int source = static_cast<int>(source_d);
            const int sink = static_cast<int>(sink_d);
            auto value = eval_graph_max_flow(*adj_m, source, sink);
            if (!value) {
                return std::unexpected(value.error());
            }
            return std::to_string(*value) + "\n";
        }
        if (fn == "boxfilter" || fn == "medfilt2" || fn == "imgaussfilt") {
            auto resolve_arg = [this](const std::string& text) -> Result<Matrix<double>> {
                auto matrix = parse_matrix(text);
                if (!matrix) {
                    matrix = resolve_matrix(text);
                }
                return matrix;
            };
            auto gray_m = resolve_arg(trim(match[2].str()));
            if (!gray_m) {
                return std::unexpected(gray_m.error());
            }
            double param_d = 0.0;
            if (!parse_number(trim(match[3].str()), param_d)) {
                return std::unexpected(DomainError{fn, "expected numeric second argument"});
            }
            auto gray = matrix_to_gray_image(*gray_m);
            if (!gray) {
                return std::unexpected(gray.error());
            }
            Matrix<double> filtered_m;
            if (fn == "boxfilter") {
                const int ksize = static_cast<int>(param_d);
                if (ksize < 1 || param_d != ksize || (ksize % 2) == 0) {
                    return std::unexpected(
                        DomainError{"boxfilter", "expected positive odd integer ksize"});
                }
                filtered_m = gray_image_to_matrix(image::boxfilter(*gray, ksize));
            } else if (fn == "medfilt2") {
                const int ksize = static_cast<int>(param_d);
                if (ksize < 1 || param_d != ksize || (ksize % 2) == 0) {
                    return std::unexpected(
                        DomainError{"medfilt2", "expected positive odd integer ksize"});
                }
                filtered_m = gray_image_to_matrix(image::medfilt2(*gray, ksize));
            } else {
                filtered_m = gray_image_to_matrix(
                    image::imgaussfilt(*gray, static_cast<float>(param_d)));
            }
            std::ostringstream out;
            out << "filtered =\n";
            print_matrix(out, filtered_m);
            return out.str();
        }
        if (fn == "signal_lowpass") {
            auto resolve_arg = [this](const std::string& text) -> Result<Matrix<double>> {
                auto matrix = parse_matrix(text);
                if (!matrix) {
                    matrix = resolve_matrix(text);
                }
                return matrix;
            };
            auto x_m = resolve_arg(trim(match[2].str()));
            if (!x_m) {
                return std::unexpected(x_m.error());
            }
            double cutoff = 0.0;
            double fs = 0.0;
            if (!parse_number(trim(match[3].str()), cutoff) ||
                !parse_number(trim(match[4].str()), fs)) {
                return std::unexpected(
                    DomainError{"signal_lowpass", "expected signal_lowpass(x, cutoff, fs)"});
            }
            auto filtered = eval_signal_lowpass(*x_m, cutoff, fs);
            if (!filtered) {
                return std::unexpected(filtered.error());
            }
            std::ostringstream out;
            out << "filtered =\n";
            print_matrix(out, *filtered);
            return out.str();
        }
        if (fn == "signal_butterworth" || fn == "signal_highpass") {
            auto resolve_arg = [this](const std::string& text) -> Result<Matrix<double>> {
                auto matrix = parse_matrix(text);
                if (!matrix) {
                    matrix = resolve_matrix(text);
                }
                return matrix;
            };
            auto x_m = resolve_arg(trim(match[2].str()));
            if (!x_m) {
                return std::unexpected(x_m.error());
            }
            double cutoff = 0.0;
            double fs = 0.0;
            if (!parse_number(trim(match[3].str()), cutoff) ||
                !parse_number(trim(match[4].str()), fs)) {
                return std::unexpected(
                    DomainError{fn, "expected " + fn + "(x, cutoff, fs)"});
            }
            Result<Matrix<double>> filtered;
            if (fn == "signal_butterworth") {
                filtered = eval_signal_butterworth(*x_m, cutoff, fs);
            } else {
                filtered = eval_signal_highpass(*x_m, cutoff, fs);
            }
            if (!filtered) {
                return std::unexpected(filtered.error());
            }
            std::ostringstream out;
            out << "filtered =\n";
            print_matrix(out, *filtered);
            return out.str();
        }
        if (fn == "prob_binom_pdf") {
            double k_d = 0.0;
            double n_d = 0.0;
            double p = 0.0;
            if (!parse_number(trim(match[2].str()), k_d) ||
                !parse_number(trim(match[3].str()), n_d) ||
                !parse_number(trim(match[4].str()), p)) {
                return std::unexpected(
                    DomainError{"prob_binom_pdf", "expected prob_binom_pdf(k, n, p)"});
            }
            const int k = static_cast<int>(k_d);
            const int n = static_cast<int>(n_d);
            if (k_d != k || n_d != n) {
                return std::unexpected(
                    DomainError{"prob_binom_pdf", "expected integer k and n"});
            }
            return std::to_string(binom_pdf(k, n, p)) + "\n";
        }
        if (fn == "prob_binom_cdf") {
            double k_d = 0.0;
            double n_d = 0.0;
            double p = 0.0;
            if (!parse_number(trim(match[2].str()), k_d) ||
                !parse_number(trim(match[3].str()), n_d) ||
                !parse_number(trim(match[4].str()), p)) {
                return std::unexpected(
                    DomainError{"prob_binom_cdf", "expected prob_binom_cdf(k, n, p)"});
            }
            const int k = static_cast<int>(k_d);
            const int n = static_cast<int>(n_d);
            if (k_d != k || n_d != n) {
                return std::unexpected(
                    DomainError{"prob_binom_cdf", "expected integer k and n"});
            }
            return std::to_string(binom_cdf(k, n, p)) + "\n";
        }
        if (fn == "prob_uniform_cdf") {
            double x = 0.0;
            double a = 0.0;
            double b = 0.0;
            if (!parse_number(trim(match[2].str()), x) ||
                !parse_number(trim(match[3].str()), a) ||
                !parse_number(trim(match[4].str()), b)) {
                return std::unexpected(
                    DomainError{"prob_uniform_cdf", "expected prob_uniform_cdf(x, a, b)"});
            }
            return std::to_string(uniform_cdf(x, a, b)) + "\n";
        }
        if (fn == "prob_gamma_pdf") {
            double x = 0.0;
            double shape = 0.0;
            double scale = 0.0;
            if (!parse_number(trim(match[2].str()), x) ||
                !parse_number(trim(match[3].str()), shape) ||
                !parse_number(trim(match[4].str()), scale)) {
                return std::unexpected(
                    DomainError{"prob_gamma_pdf", "expected prob_gamma_pdf(x, shape, scale)"});
            }
            return std::to_string(gamma_pdf(x, shape, scale)) + "\n";
        }
        if (fn == "kummer_m" || fn == "whittaker_m" || fn == "mathieu_ce" || fn == "painleve1") {
            double a = 0.0;
            double b = 0.0;
            double c = 0.0;
            if (!parse_number(trim(match[2].str()), a) || !parse_number(trim(match[3].str()), b) ||
                !parse_number(trim(match[4].str()), c)) {
                return std::unexpected(DomainError{fn, "expected numeric arguments"});
            }
            if (fn == "kummer_m") {
                return std::to_string(kummer_m(a, b, c)) + "\n";
            }
            if (fn == "whittaker_m") {
                return std::to_string(whittaker_m(a, b, c)) + "\n";
            }
            if (fn == "mathieu_ce") {
                return std::to_string(mathieu_ce(static_cast<int>(a), b, c)) + "\n";
            }
            return std::to_string(painleve1(a, b, c)) + "\n";
        }
        if (fn == "finance_bond_price") {
            double c = 0.0;
            double y = 0.0;
            double n_d = 0.0;
            if (!parse_number(trim(match[2].str()), c) || !parse_number(trim(match[3].str()), y) ||
                !parse_number(trim(match[4].str()), n_d)) {
                return std::unexpected(
                    DomainError{"finance_bond_price", "expected finance_bond_price(c,y,n)"});
            }
            const int n = static_cast<int>(n_d);
            if (n < 0 || n_d != n) {
                return std::unexpected(
                    DomainError{"finance_bond_price", "expected non-negative integer periods n"});
            }
            return std::to_string(finance::bond_price(c, y, n)) + "\n";
        }
        if (fn == "cplx_poisson_kernel") {
            double theta = 0.0;
            double phi = 0.0;
            double r = 0.0;
            if (!parse_number(trim(match[2].str()), theta) ||
                !parse_number(trim(match[3].str()), phi) ||
                !parse_number(trim(match[4].str()), r)) {
                return std::unexpected(DomainError{
                    "cplx_poisson_kernel", "expected cplx_poisson_kernel(theta,phi,r)"});
            }
            return std::to_string(cplx::poisson_kernel(theta, phi, r)) + "\n";
        }
        if (fn == "numthy_mod_pow") {
            double base_d = 0.0;
            double exp_d = 0.0;
            double mod_d = 0.0;
            if (!parse_number(trim(match[2].str()), base_d) ||
                !parse_number(trim(match[3].str()), exp_d) ||
                !parse_number(trim(match[4].str()), mod_d)) {
                return std::unexpected(
                    DomainError{"numthy_mod_pow", "expected numthy_mod_pow(base,exp,mod)"});
            }
            return std::to_string(numthy::mod_pow(static_cast<uint64_t>(base_d),
                                                    static_cast<uint64_t>(exp_d),
                                                    static_cast<uint64_t>(mod_d))) +
                   "\n";
        }
        if (fn == "numthy_discrete_log") {
            double g_d = 0.0;
            double h_d = 0.0;
            double p_d = 0.0;
            if (!parse_number(trim(match[2].str()), g_d) ||
                !parse_number(trim(match[3].str()), h_d) ||
                !parse_number(trim(match[4].str()), p_d)) {
                return std::unexpected(
                    DomainError{"numthy_discrete_log", "expected numthy_discrete_log(g,h,p)"});
            }
            auto x = eval_numthy_discrete_log(g_d, h_d, p_d);
            if (!x) {
                return std::unexpected(x.error());
            }
            return std::to_string(*x) + "\n";
        }
        if (fn == "finance_bond_duration") {
            double c = 0.0;
            double y = 0.0;
            double n_d = 0.0;
            if (!parse_number(trim(match[2].str()), c) || !parse_number(trim(match[3].str()), y) ||
                !parse_number(trim(match[4].str()), n_d)) {
                return std::unexpected(
                    DomainError{"finance_bond_duration", "expected finance_bond_duration(c,y,n)"});
            }
            const int n = static_cast<int>(n_d);
            if (n < 0 || n_d != n) {
                return std::unexpected(
                    DomainError{"finance_bond_duration", "expected non-negative integer periods n"});
            }
            return std::to_string(finance::bond_duration(c, y, n)) + "\n";
        }
        if (fn == "prob_norm_cdf") {
            double x = 0.0;
            double mu = 0.0;
            double sigma = 0.0;
            if (!parse_number(trim(match[2].str()), x) || !parse_number(trim(match[3].str()), mu) ||
                !parse_number(trim(match[4].str()), sigma)) {
                return std::unexpected(
                    DomainError{"prob_norm_cdf", "expected prob_norm_cdf(x,mu,sigma)"});
            }
            return std::to_string(norm_cdf(x, mu, sigma)) + "\n";
        }
        if (fn == "prob_norm_pdf") {
            double x = 0.0;
            double mu = 0.0;
            double sigma = 0.0;
            if (!parse_number(trim(match[2].str()), x) || !parse_number(trim(match[3].str()), mu) ||
                !parse_number(trim(match[4].str()), sigma)) {
                return std::unexpected(
                    DomainError{"prob_norm_pdf", "expected prob_norm_pdf(x,mu,sigma)"});
            }
            return std::to_string(norm_pdf(x, mu, sigma)) + "\n";
        }
        if (fn == "prob_norm_ppf") {
            double p = 0.0;
            double mu = 0.0;
            double sigma = 0.0;
            if (!parse_number(trim(match[2].str()), p) || !parse_number(trim(match[3].str()), mu) ||
                !parse_number(trim(match[4].str()), sigma)) {
                return std::unexpected(
                    DomainError{"prob_norm_ppf", "expected prob_norm_ppf(p,mu,sigma)"});
            }
            return std::to_string(norm_ppf(p, mu, sigma)) + "\n";
        }
        if (fn == "finance_bond_modified_duration") {
            double c = 0.0;
            double y = 0.0;
            double n_d = 0.0;
            if (!parse_number(trim(match[2].str()), c) || !parse_number(trim(match[3].str()), y) ||
                !parse_number(trim(match[4].str()), n_d)) {
                return std::unexpected(DomainError{
                    "finance_bond_modified_duration",
                    "expected finance_bond_modified_duration(c,y,n)"});
            }
            const int n = static_cast<int>(n_d);
            if (n < 0 || n_d != n) {
                return std::unexpected(DomainError{
                    "finance_bond_modified_duration",
                    "expected non-negative integer periods n"});
            }
            return std::to_string(finance::bond_modified_duration(c, y, n)) + "\n";
        }
        if (fn == "finance_bond_convexity") {
            double c = 0.0;
            double y = 0.0;
            double n_d = 0.0;
            if (!parse_number(trim(match[2].str()), c) || !parse_number(trim(match[3].str()), y) ||
                !parse_number(trim(match[4].str()), n_d)) {
                return std::unexpected(
                    DomainError{"finance_bond_convexity", "expected finance_bond_convexity(c,y,n)"});
            }
            const int n = static_cast<int>(n_d);
            if (n < 0 || n_d != n) {
                return std::unexpected(
                    DomainError{"finance_bond_convexity", "expected non-negative integer periods n"});
            }
            return std::to_string(finance::bond_convexity(c, y, n)) + "\n";
        }
        if (fn == "finance_bond_ytm") {
            double price = 0.0;
            double c = 0.0;
            double n_d = 0.0;
            if (!parse_number(trim(match[2].str()), price) || !parse_number(trim(match[3].str()), c) ||
                !parse_number(trim(match[4].str()), n_d)) {
                return std::unexpected(
                    DomainError{"finance_bond_ytm", "expected finance_bond_ytm(price,c,n)"});
            }
            auto ytm = eval_finance_bond_ytm(price, c, n_d);
            if (!ytm) {
                return std::unexpected(ytm.error());
            }
            return std::to_string(*ytm) + "\n";
        }
        if (fn == "cplx_poisson_kernel") {
            double theta = 0.0;
            double phi = 0.0;
            double r = 0.0;
            if (!parse_number(trim(match[2].str()), theta) ||
                !parse_number(trim(match[3].str()), phi) ||
                !parse_number(trim(match[4].str()), r)) {
                return std::unexpected(DomainError{
                    "cplx_poisson_kernel", "expected cplx_poisson_kernel(theta,phi,r)"});
            }
            return std::to_string(cplx::poisson_kernel(theta, phi, r)) + "\n";
        }
        if (fn == "finance_compound") {
            double principal = 0.0;
            double rate = 0.0;
            double n_d = 0.0;
            if (!parse_number(trim(match[2].str()), principal) ||
                !parse_number(trim(match[3].str()), rate) ||
                !parse_number(trim(match[4].str()), n_d)) {
                return std::unexpected(
                    DomainError{"finance_compound", "expected finance_compound(principal,rate,n_periods)"});
            }
            const int n = static_cast<int>(n_d);
            if (n < 0 || n_d != n) {
                return std::unexpected(
                    DomainError{"finance_compound", "expected non-negative integer periods n_periods"});
            }
            return std::to_string(finance::compound(principal, rate, n)) + "\n";
        }
        if (fn == "finance_continuous_compound") {
            double principal = 0.0;
            double rate = 0.0;
            double t = 0.0;
            if (!parse_number(trim(match[2].str()), principal) ||
                !parse_number(trim(match[3].str()), rate) ||
                !parse_number(trim(match[4].str()), t)) {
                return std::unexpected(DomainError{
                    "finance_continuous_compound",
                    "expected finance_continuous_compound(principal,rate,t)"});
            }
            return std::to_string(finance::continuous_compound(principal, rate, t)) + "\n";
        }
        if (fn == "finance_pv") {
            double rate = 0.0;
            double n_d = 0.0;
            double pmt = 0.0;
            if (!parse_number(trim(match[2].str()), rate) || !parse_number(trim(match[3].str()), n_d) ||
                !parse_number(trim(match[4].str()), pmt)) {
                return std::unexpected(
                    DomainError{"finance_pv", "expected finance_pv(rate,n,pmt)"});
            }
            const int n = static_cast<int>(n_d);
            if (n < 0 || n_d != n) {
                return std::unexpected(
                    DomainError{"finance_pv", "expected non-negative integer n"});
            }
            return std::to_string(finance::pv(rate, n, pmt)) + "\n";
        }
        if (fn == "finance_fv_annuity") {
            double rate = 0.0;
            double n_d = 0.0;
            double pmt = 0.0;
            if (!parse_number(trim(match[2].str()), rate) || !parse_number(trim(match[3].str()), n_d) ||
                !parse_number(trim(match[4].str()), pmt)) {
                return std::unexpected(
                    DomainError{"finance_fv_annuity", "expected finance_fv_annuity(rate,n,pmt)"});
            }
            const int n = static_cast<int>(n_d);
            if (n < 0 || n_d != n) {
                return std::unexpected(
                    DomainError{"finance_fv_annuity", "expected non-negative integer n"});
            }
            return std::to_string(finance::fv_annuity(rate, n, pmt)) + "\n";
        }
        if (fn == "finance_pmt_annuity") {
            double rate = 0.0;
            double n_d = 0.0;
            double pv0 = 0.0;
            if (!parse_number(trim(match[2].str()), rate) || !parse_number(trim(match[3].str()), n_d) ||
                !parse_number(trim(match[4].str()), pv0)) {
                return std::unexpected(
                    DomainError{"finance_pmt_annuity", "expected finance_pmt_annuity(rate,n,pv0)"});
            }
            const int n = static_cast<int>(n_d);
            if (n < 0 || n_d != n) {
                return std::unexpected(
                    DomainError{"finance_pmt_annuity", "expected non-negative integer n"});
            }
            return std::to_string(finance::pmt_annuity(rate, n, pv0)) + "\n";
        }
        if (fn == "quantum_entanglement_entropy") {
            auto resolve_arg = [this](const std::string& text) -> Result<Matrix<double>> {
                auto matrix = parse_matrix(text);
                if (!matrix) {
                    matrix = resolve_matrix(text);
                }
                return matrix;
            };
            auto psi_m = resolve_arg(trim(match[2].str()));
            if (!psi_m) {
                return std::unexpected(psi_m.error());
            }
            double dim_a_d = 0.0;
            double dim_b_d = 0.0;
            if (!parse_number(trim(match[3].str()), dim_a_d) ||
                !parse_number(trim(match[4].str()), dim_b_d)) {
                return std::unexpected(DomainError{
                    "quantum_entanglement_entropy",
                    "expected quantum_entanglement_entropy(psi, dim_a, dim_b)"});
            }
            const int dim_a = static_cast<int>(dim_a_d);
            const int dim_b = static_cast<int>(dim_b_d);
            if (dim_a < 1 || dim_b < 1 || dim_a_d != dim_a || dim_b_d != dim_b) {
                return std::unexpected(DomainError{
                    "quantum_entanglement_entropy", "expected positive integer dim_a and dim_b"});
            }
            auto value = eval_quantum_entanglement_entropy(*psi_m, dim_a, dim_b);
            if (!value) {
                return std::unexpected(value.error());
            }
            return std::to_string(*value) + "\n";
        }
        if (fn == "info_sample_entropy") {
            auto resolve_arg = [this](const std::string& text) -> Result<Matrix<double>> {
                auto matrix = parse_matrix(text);
                if (!matrix) {
                    matrix = resolve_matrix(text);
                }
                return matrix;
            };
            auto x_m = resolve_arg(trim(match[2].str()));
            if (!x_m) {
                return std::unexpected(x_m.error());
            }
            double m_d = 0.0;
            double r = 0.0;
            if (!parse_number(trim(match[3].str()), m_d) || !parse_number(trim(match[4].str()), r)) {
                return std::unexpected(DomainError{
                    "info_sample_entropy", "expected info_sample_entropy(x, m, r)"});
            }
            const int m = static_cast<int>(m_d);
            if (m < 1 || m_d != m) {
                return std::unexpected(DomainError{
                    "info_sample_entropy", "expected positive integer m"});
            }
            auto value = eval_info_sample_entropy(*x_m, m, r);
            if (!value) {
                return std::unexpected(value.error());
            }
            return std::to_string(*value) + "\n";
        }
        if (fn == "stats_ztest") {
            auto resolve_arg = [this](const std::string& text) -> Result<Matrix<double>> {
                auto matrix = parse_matrix(text);
                if (!matrix) {
                    matrix = resolve_matrix(text);
                }
                return matrix;
            };
            auto x_m = resolve_arg(trim(match[2].str()));
            if (!x_m) {
                return std::unexpected(x_m.error());
            }
            double mu = 0.0;
            double sigma = 0.0;
            if (!parse_number(trim(match[3].str()), mu) ||
                !parse_number(trim(match[4].str()), sigma)) {
                return std::unexpected(DomainError{
                    "stats_ztest", "expected stats_ztest(x, mu, sigma)"});
            }
            auto value = eval_stats_ztest(*x_m, mu, sigma);
            if (!value) {
                return std::unexpected(value.error());
            }
            return std::to_string(*value) + "\n";
        }
    }

    if (const auto open = cmd.find('('); open != std::string::npos && !cmd.empty() && cmd.back() == ')') {
        const std::string fn = lower(trim_copy(cmd.substr(0, open)));
        if (fn == "matmul" || fn == "tensorops_matmul" || fn == "tensorops_einsum") {
            const auto call_args = split_call_args(cmd);
            if (call_args && call_args->size() == 2) {
                auto resolve_arg = [this](const std::string& text) -> Result<Matrix<double>> {
                    auto matrix = parse_matrix(text);
                    if (!matrix) {
                        matrix = resolve_matrix(text);
                    }
                    return matrix;
                };
                auto A = resolve_arg(call_args->at(0));
                if (!A) {
                    return std::unexpected(A.error());
                }
                auto B = resolve_arg(call_args->at(1));
                if (!B) {
                    return std::unexpected(B.error());
                }
                Result<Matrix<double>> C;
                if (fn == "tensorops_matmul") {
                    C = eval_tensorops_matmul(*A, *B);
                } else if (fn == "tensorops_einsum") {
                    C = eval_tensorops_einsum(*A, *B);
                } else {
                    C = matmul(*A, *B);
                }
                if (!C) {
                    return std::unexpected(C.error());
                }
                std::ostringstream out;
                out << "C =\n";
                print_matrix(out, *C);
                return out.str();
            }
        }
        if (fn == "quantum_partial_trace") {
            const auto call_args = split_call_args(cmd);
            if (call_args && call_args->size() == 4) {
                auto resolve_arg = [this](const std::string& text) -> Result<Matrix<double>> {
                    auto matrix = parse_matrix(text);
                    if (!matrix) {
                        matrix = resolve_matrix(text);
                    }
                    return matrix;
                };
                auto rho_m = resolve_arg(call_args->at(0));
                if (!rho_m) {
                    return std::unexpected(rho_m.error());
                }
                double d1_d = 0.0;
                double d2_d = 0.0;
                double sub_d = 0.0;
                if (!parse_number(trim_copy(call_args->at(1)), d1_d) ||
                    !parse_number(trim_copy(call_args->at(2)), d2_d) ||
                    !parse_number(trim_copy(call_args->at(3)), sub_d)) {
                    return std::unexpected(DomainError{
                        "quantum_partial_trace",
                        "expected quantum_partial_trace(rho, d1, d2, subsystem)"});
                }
                const int d1 = static_cast<int>(d1_d);
                const int d2 = static_cast<int>(d2_d);
                const int subsystem = static_cast<int>(sub_d);
                if (d1 < 1 || d2 < 1 || d1_d != d1 || d2_d != d2 ||
                    (subsystem != 0 && subsystem != 1) || sub_d != subsystem) {
                    return std::unexpected(DomainError{
                        "quantum_partial_trace",
                        "expected positive integer d1, d2 and subsystem 0 or 1"});
                }
                auto result = eval_quantum_partial_trace_matrix(*rho_m, d1, d2, subsystem);
                if (!result) {
                    return std::unexpected(result.error());
                }
                std::ostringstream out;
                out << "rho =\n";
                print_matrix(out, *result);
                return out.str();
            }
        }
        if (fn == "combo_unrank_combination") {
            const auto call_args = split_call_args(cmd);
            if (call_args && call_args->size() == 3) {
                double n_d = 0.0;
                double k_d = 0.0;
                double rank_d = 0.0;
                if (!parse_number(trim_copy(call_args->at(0)), n_d) ||
                    !parse_number(trim_copy(call_args->at(1)), k_d) ||
                    !parse_number(trim_copy(call_args->at(2)), rank_d)) {
                    return std::unexpected(DomainError{
                        "combo_unrank_combination",
                        "expected combo_unrank_combination(n,k,rank)"});
                }
                const int n = static_cast<int>(n_d);
                const int k = static_cast<int>(k_d);
                if (n < 0 || k < 0 || n_d != n || k_d != k || rank_d < 0.0 ||
                    std::floor(rank_d) != rank_d) {
                    return std::unexpected(DomainError{
                        "combo_unrank_combination",
                        "expected non-negative integer n, k and rank"});
                }
                auto value = eval_combo_unrank_combination(n, k, static_cast<uint64_t>(rank_d));
                if (!value) {
                    return std::unexpected(value.error());
                }
                std::ostringstream out;
                out << "comb =\n";
                print_matrix(out, *value);
                return out.str();
            }
        }
        if (fn == "quantum_schrodinger" || fn == "quantum_schrodinger_final") {
            const auto call_args = split_call_args(cmd);
            if (call_args && call_args->size() == 5) {
                auto resolve_arg = [this](const std::string& text) -> Result<Matrix<double>> {
                    auto matrix = parse_matrix(text);
                    if (!matrix) {
                        matrix = resolve_matrix(text);
                    }
                    return matrix;
                };
                auto H_m = resolve_arg(call_args->at(0));
                if (!H_m) {
                    return std::unexpected(H_m.error());
                }
                auto psi0_m = resolve_arg(call_args->at(1));
                if (!psi0_m) {
                    return std::unexpected(psi0_m.error());
                }
                double t0 = 0.0;
                double t1 = 0.0;
                double n_steps_d = 0.0;
                if (!parse_number(trim_copy(call_args->at(2)), t0) ||
                    !parse_number(trim_copy(call_args->at(3)), t1) ||
                    !parse_number(trim_copy(call_args->at(4)), n_steps_d)) {
                    return std::unexpected(DomainError{
                        fn,
                        "expected " + fn + "(H, psi0, t0, t1, n_steps)"});
                }
                const int n_steps = static_cast<int>(n_steps_d);
                if (n_steps < 0 || n_steps_d != n_steps) {
                    return std::unexpected(DomainError{fn, "expected non-negative integer n_steps"});
                }
                if (fn == "quantum_schrodinger_final") {
                    auto result = eval_quantum_schrodinger_final(*H_m, *psi0_m, t0, t1, n_steps);
                    if (!result) {
                        return std::unexpected(result.error());
                    }
                    std::ostringstream out;
                    out << "psi =\n";
                    print_matrix(out, *result);
                    return out.str();
                }
                auto result = eval_quantum_schrodinger_matrix(*H_m, *psi0_m, t0, t1, n_steps);
                if (!result) {
                    return std::unexpected(result.error());
                }
                std::ostringstream out;
                out << "traj =\n";
                print_matrix(out, *result);
                return out.str();
            }
        }
        if (fn == "info_joint_entropy" || fn == "info_conditional_entropy" ||
            fn == "info_sample_entropy" || fn == "geo_point_in_polygon" ||
            fn == "geo_kdtree_nearest" ||
            fn == "cplx_power_series_eval" || fn == "cplx_winding_number" ||
            fn == "topo_vietoris_rips_betti0" || fn == "topo_betti_curve" ||
            fn == "control_bode_mag_db" || fn == "control_bode_phase" ||
            fn == "topo_bottleneck_distance" || fn == "topo_wasserstein_distance") {
            const auto call_args = split_call_args(cmd);
            if (call_args && call_args->size() == 3) {
                if (fn == "control_bode_mag_db" || fn == "control_bode_phase" ||
                    fn == "topo_bottleneck_distance" || fn == "topo_wasserstein_distance") {
                    auto resolve_arg = [this](const std::string& text) -> Result<Matrix<double>> {
                        auto matrix = parse_matrix(text);
                        if (!matrix) {
                            matrix = resolve_matrix(text);
                        }
                        return matrix;
                    };
                    auto arg_a_m = resolve_arg(call_args->at(0));
                    if (!arg_a_m) {
                        return std::unexpected(arg_a_m.error());
                    }
                    auto arg_b_m = resolve_arg(call_args->at(1));
                    if (!arg_b_m) {
                        return std::unexpected(arg_b_m.error());
                    }
                    double arg2 = 0.0;
                    if (!parse_number(trim_copy(call_args->at(2)), arg2)) {
                        auto w_expr = eval_scalar_expr(state_, trim_copy(call_args->at(2)));
                        if (!w_expr) {
                            return std::unexpected(w_expr.error());
                        }
                        arg2 = *w_expr;
                    }
                    Result<double> value =
                        std::unexpected(DomainError{fn, "unsupported matrix-scalar call"});
                    if (fn == "control_bode_mag_db") {
                        value = eval_control_bode_mag_db(*arg_a_m, *arg_b_m, arg2);
                    } else if (fn == "control_bode_phase") {
                        value = eval_control_bode_phase(*arg_a_m, *arg_b_m, arg2);
                    } else if (fn == "topo_wasserstein_distance") {
                        const int dim = static_cast<int>(arg2);
                        if (dim < 0 || arg2 != dim) {
                            return std::unexpected(DomainError{
                                "topo_wasserstein_distance",
                                "expected non-negative integer dim"});
                        }
                        value = eval_topo_wasserstein_distance(*arg_a_m, *arg_b_m, dim);
                    } else {
                        const int dim = static_cast<int>(arg2);
                        if (dim < 0 || arg2 != dim) {
                            return std::unexpected(DomainError{
                                "topo_bottleneck_distance",
                                "expected non-negative integer dim"});
                        }
                        value = eval_topo_bottleneck_distance(*arg_a_m, *arg_b_m, dim);
                    }
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    return std::to_string(*value) + "\n";
                }
                if (fn == "geo_point_in_polygon") {
                    double px = 0.0;
                    double py = 0.0;
                    if (!parse_number(trim_copy(call_args->at(0)), px) ||
                        !parse_number(trim_copy(call_args->at(1)), py)) {
                        return std::unexpected(DomainError{
                            "geo_point_in_polygon",
                            "expected geo_point_in_polygon(px, py, P)"});
                    }
                    auto resolve_arg = [this](const std::string& text) -> Result<Matrix<double>> {
                        auto matrix = parse_matrix(text);
                        if (!matrix) {
                            matrix = resolve_matrix(text);
                        }
                        return matrix;
                    };
                    auto poly_m = resolve_arg(call_args->at(2));
                    if (!poly_m) {
                        return std::unexpected(poly_m.error());
                    }
                    auto value = eval_geo_point_in_polygon(px, py, *poly_m);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    return std::to_string(*value) + "\n";
                }
                if (fn == "geo_kdtree_nearest") {
                    double qx = 0.0;
                    double qy = 0.0;
                    if (!parse_number(trim_copy(call_args->at(1)), qx) ||
                        !parse_number(trim_copy(call_args->at(2)), qy)) {
                        return std::unexpected(DomainError{
                            "geo_kdtree_nearest",
                            "expected geo_kdtree_nearest(P, x, y)"});
                    }
                    auto resolve_arg = [this](const std::string& text) -> Result<Matrix<double>> {
                        auto matrix = parse_matrix(text);
                        if (!matrix) {
                            matrix = resolve_matrix(text);
                        }
                        return matrix;
                    };
                    auto P_m = resolve_arg(call_args->at(0));
                    if (!P_m) {
                        return std::unexpected(P_m.error());
                    }
                    auto value = eval_geo_kdtree_nearest(*P_m, qx, qy);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    return std::to_string(*value) + "\n";
                }
                auto resolve_arg = [this](const std::string& text) -> Result<Matrix<double>> {
                    auto matrix = parse_matrix(text);
                    if (!matrix) {
                        matrix = resolve_matrix(text);
                    }
                    return matrix;
                };
                auto matrix_m = resolve_arg(call_args->at(0));
                if (!matrix_m) {
                    return std::unexpected(matrix_m.error());
                }
                if (fn == "info_sample_entropy") {
                    double m_d = 0.0;
                    double r = 0.0;
                    if (!parse_number(trim_copy(call_args->at(1)), m_d) ||
                        !parse_number(trim_copy(call_args->at(2)), r)) {
                        return std::unexpected(DomainError{
                            "info_sample_entropy",
                            "expected info_sample_entropy(x, m, r)"});
                    }
                    const int m = static_cast<int>(m_d);
                    if (m < 1 || m_d != m) {
                        return std::unexpected(DomainError{
                            "info_sample_entropy", "expected positive integer m"});
                    }
                    auto value = eval_info_sample_entropy(*matrix_m, m, r);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    return std::to_string(*value) + "\n";
                }
                if (fn == "cplx_power_series_eval" || fn == "cplx_winding_number" ||
                    fn == "topo_vietoris_rips_betti0") {
                    double arg1 = 0.0;
                    double arg2 = 0.0;
                    if (!parse_number(trim_copy(call_args->at(1)), arg1) ||
                        !parse_number(trim_copy(call_args->at(2)), arg2)) {
                        auto s1 = eval_scalar_expr(state_, trim_copy(call_args->at(1)));
                        if (!s1) {
                            return std::unexpected(s1.error());
                        }
                        arg1 = *s1;
                        auto s2 = eval_scalar_expr(state_, trim_copy(call_args->at(2)));
                        if (!s2) {
                            return std::unexpected(s2.error());
                        }
                        arg2 = *s2;
                    }
                    Result<double> value =
                        std::unexpected(DomainError{fn, "unsupported matrix-scalar call"});
                    if (fn == "cplx_power_series_eval") {
                        value = eval_cplx_power_series_eval(*matrix_m, arg1, arg2);
                    } else if (fn == "cplx_winding_number") {
                        value = eval_cplx_winding_number(*matrix_m, arg1, arg2);
                    } else {
                        const int max_dim = static_cast<int>(arg2);
                        if (max_dim < 0 || arg2 != max_dim) {
                            return std::unexpected(DomainError{
                                "topo_vietoris_rips_betti0",
                                "expected non-negative integer max_dim"});
                        }
                        value = eval_topo_vietoris_rips_betti0(*matrix_m, arg1, max_dim);
                    }
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    return std::to_string(*value) + "\n";
                }
                double rows_d = 0.0;
                double cols_d = 0.0;
                if (!parse_number(trim_copy(call_args->at(1)), rows_d) ||
                    !parse_number(trim_copy(call_args->at(2)), cols_d)) {
                    return std::unexpected(DomainError{
                        fn, "expected " + fn + "(joint, rows, cols)"});
                }
                const int rows = static_cast<int>(rows_d);
                const int cols = static_cast<int>(cols_d);
                if (rows < 1 || cols < 1 || rows_d != rows || cols_d != cols) {
                    return std::unexpected(DomainError{fn, "expected positive integer rows and cols"});
                }
                Result<double> value;
                if (fn == "info_joint_entropy") {
                    value = eval_info_joint_entropy(*matrix_m, rows, cols);
                } else {
                    value = eval_info_conditional_entropy(*matrix_m, rows, cols);
                }
                if (!value) {
                    return std::unexpected(value.error());
                }
                return std::to_string(*value) + "\n";
            }
        }
        if (fn == "geo_bezier_eval_x" || fn == "geo_bezier_eval_y" || fn == "bwt_decode_vec" ||
            fn == "combo_rank_combination" || fn == "combo_next_comb" ||
            fn == "quantum_time_evolution" ||
            fn == "signal_moving_average" || fn == "graph_bfs" || fn == "graph_dfs" ||
            fn == "stats_percentile" || fn == "stats_ttest" || fn == "stats_acf" || fn == "graph_dfs" ||
            fn == "stats_percentile" ||
            fn == "poly_eval" || fn == "fft_irfft" || fn == "poly_integ" ||
            fn == "cplx_blaschke_product") {
            const auto call_args = split_call_args(cmd);
            if (call_args && call_args->size() == 3 && fn == "cplx_blaschke_product") {
                double zre = 0.0;
                double zim = 0.0;
                if (!parse_number(trim_copy(call_args->at(0)), zre) ||
                    !parse_number(trim_copy(call_args->at(1)), zim)) {
                    return std::unexpected(DomainError{
                        "cplx_blaschke_product",
                        "expected cplx_blaschke_product(zre,zim,zeros)"});
                }
                auto resolve_arg = [this](const std::string& text) -> Result<Matrix<double>> {
                    auto matrix = parse_matrix(text);
                    if (!matrix) {
                        matrix = resolve_matrix(text);
                    }
                    return matrix;
                };
                auto zeros_m = resolve_arg(call_args->at(2));
                if (!zeros_m) {
                    return std::unexpected(zeros_m.error());
                }
                auto value = eval_cplx_blaschke_product(zre, zim, *zeros_m);
                if (!value) {
                    return std::unexpected(value.error());
                }
                return std::to_string(*value) + "\n";
            }
            if (call_args && call_args->size() == 2) {
                auto resolve_arg = [this](const std::string& text) -> Result<Matrix<double>> {
                    auto matrix = parse_matrix(text);
                    if (!matrix) {
                        matrix = resolve_matrix(text);
                    }
                    return matrix;
                };
                auto ctrl = resolve_arg(call_args->at(0));
                if (!ctrl) {
                    return std::unexpected(ctrl.error());
                }
                double t = 0.0;
                if (!parse_number(trim_copy(call_args->at(1)), t)) {
                    auto scalar_expr = eval_scalar_expr(state_, trim_copy(call_args->at(1)));
                    if (!scalar_expr) {
                        return std::unexpected(scalar_expr.error());
                    }
                    t = *scalar_expr;
                }
                if (fn == "bwt_decode_vec") {
                    auto value = eval_bwt_decode_vec(*ctrl, t);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    std::ostringstream out;
                    out << "decoded =\n";
                    print_matrix(out, *value);
                    return out.str();
                }
                if (fn == "quantum_time_evolution") {
                    auto value = eval_quantum_time_evolution_matrix(*ctrl, t);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    std::ostringstream out;
                    out << "U =\n";
                    print_matrix(out, *value);
                    return out.str();
                }
                if (fn == "combo_rank_combination") {
                    const int n = static_cast<int>(t);
                    if (n < 0 || t != n) {
                        return std::unexpected(DomainError{
                            "combo_rank_combination", "expected non-negative integer n"});
                    }
                    auto value = eval_combo_rank_combination(*ctrl, n);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    return std::to_string(*value) + "\n";
                }
                if (fn == "combo_next_comb") {
                    const int n = static_cast<int>(t);
                    if (n < 0 || t != n) {
                        return std::unexpected(DomainError{
                            "combo_next_comb", "expected non-negative integer n"});
                    }
                    auto value = eval_combo_next_comb(*ctrl, n);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    std::ostringstream out;
                    out << "comb =\n";
                    print_matrix(out, *value);
                    return out.str();
                }
                if (fn == "signal_moving_average") {
                    const int window = static_cast<int>(t);
                    if (window < 1 || t != window) {
                        return std::unexpected(DomainError{
                            "signal_moving_average", "expected positive integer window"});
                    }
                    auto value = eval_signal_moving_average(*ctrl, static_cast<size_t>(window));
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    std::ostringstream out;
                    out << "smoothed =\n";
                    print_matrix(out, *value);
                    return out.str();
                }
                if (fn == "graph_bfs") {
                    const int source = static_cast<int>(t);
                    if (source < 0 || t != source) {
                        return std::unexpected(DomainError{
                            "graph_bfs", "expected non-negative integer source"});
                    }
                    auto value = eval_graph_bfs(*ctrl, source);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    std::ostringstream out;
                    out << "order =\n";
                    print_matrix(out, *value);
                    return out.str();
                }
                if (fn == "graph_dfs") {
                    const int source = static_cast<int>(t);
                    if (source < 0 || t != source) {
                        return std::unexpected(DomainError{
                            "graph_dfs", "expected non-negative integer source"});
                    }
                    auto value = eval_graph_dfs(*ctrl, source);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    std::ostringstream out;
                    out << "order =\n";
                    print_matrix(out, *value);
                    return out.str();
                }
                if (fn == "stats_percentile") {
                    auto value = eval_stats_percentile(*ctrl, t);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    return std::to_string(*value) + "\n";
                }
                if (fn == "stats_ttest") {
                    auto value = eval_stats_ttest(*ctrl, t);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    return std::to_string(*value) + "\n";
                }
                if (fn == "stats_acf") {
                    const int max_lag = static_cast<int>(t);
                    if (max_lag < 0 || t != max_lag) {
                        return std::unexpected(DomainError{
                            "stats_acf", "expected non-negative integer max_lag"});
                    }
                    auto value = eval_stats_acf(*ctrl, max_lag);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    std::ostringstream out;
                    out << "acf =\n";
                    print_matrix(out, *value);
                    return out.str();
                }
                if (fn == "fft_irfft") {
                    const int n = static_cast<int>(t);
                    if (n < 1 || t != n) {
                        return std::unexpected(DomainError{
                            "fft_irfft", "expected positive integer n"});
                    }
                    auto value = eval_fft_irfft(*ctrl, static_cast<size_t>(n));
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    std::ostringstream out;
                    out << "signal =\n";
                    print_matrix(out, *value);
                    return out.str();
                }
                if (fn == "poly_integ") {
                    auto value = eval_poly_integ(*ctrl, t);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    std::ostringstream out;
                    out << "integ =\n";
                    print_matrix(out, *value);
                    return out.str();
                }
                if (fn == "poly_eval") {
                    auto value = eval_poly_eval(*ctrl, t);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    return std::to_string(*value) + "\n";
                }
                Result<double> value;
                if (fn == "geo_bezier_eval_x") {
                    value = eval_geo_bezier_eval_x(*ctrl, t);
                } else {
                    value = eval_geo_bezier_eval_y(*ctrl, t);
                }
                if (!value) {
                    return std::unexpected(value.error());
                }
                return std::to_string(*value) + "\n";
            }
        }
    }

    if (const auto open = cmd.find('('); open != std::string::npos) {
        const std::string fn = lower(trim_copy(cmd.substr(0, open)));
        if (is_matrix_dual_matrix_call_callee(fn)) {
            const auto call_args = split_call_args(cmd);
            if (call_args && call_args->size() == 2) {
                auto resolve_arg = [this](const std::string& text) -> Result<Matrix<double>> {
                    auto matrix = parse_matrix(text);
                    if (!matrix) {
                        matrix = resolve_matrix(text);
                    }
                    return matrix;
                };
                auto arg_a_m = resolve_arg(call_args->at(0));
                if (!arg_a_m) {
                    return std::unexpected(arg_a_m.error());
                }
                auto arg_b_m = resolve_arg(call_args->at(1));
                if (!arg_b_m) {
                    return std::unexpected(arg_b_m.error());
                }
                if (fn == "control_lyap") {
                    auto value = eval_control_lyap(*arg_a_m, *arg_b_m);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    std::ostringstream out;
                    out << "X =\n";
                    print_matrix(out, *value);
                    return out.str();
                }
                if (fn == "control_dlyap") {
                    auto value = eval_control_dlyap(*arg_a_m, *arg_b_m);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    std::ostringstream out;
                    out << "X =\n";
                    print_matrix(out, *value);
                    return out.str();
                }
                if (fn == "huffman_decode_vec") {
                    auto value = eval_huffman_decode_vec(*arg_a_m, *arg_b_m);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    std::ostringstream out;
                    out << "decoded =\n";
                    print_matrix(out, *value);
                    return out.str();
                }
                if (fn == "quantum_op_apply") {
                    auto value = eval_quantum_op_apply(*arg_a_m, *arg_b_m);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    std::ostringstream out;
                    out << "psi =\n";
                    print_matrix(out, *value);
                    return out.str();
                }
                if (fn == "topo_persistence_diagram") {
                    auto value = eval_topo_persistence_diagram(*arg_a_m, *arg_b_m);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    std::ostringstream out;
                    out << "dgm =\n";
                    print_matrix(out, *value);
                    return out.str();
                }
                if (fn == "control_margins") {
                    auto value = eval_control_margins(*arg_a_m, *arg_b_m);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    std::ostringstream out;
                    out << "margins =\n";
                    print_matrix(out, *value);
                    return out.str();
                }
                if (fn == "quantum_commutator") {
                    auto value = eval_quantum_commutator(*arg_a_m, *arg_b_m);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    std::ostringstream out;
                    out << "comm =\n";
                    print_matrix(out, *value);
                    return out.str();
                }
                if (fn == "signal_convolve") {
                    auto value = eval_signal_convolve(*arg_a_m, *arg_b_m);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    std::ostringstream out;
                    out << "conv =\n";
                    print_matrix(out, *value);
                    return out.str();
                }
                if (fn == "signal_correlate") {
                    auto value = eval_signal_correlate(*arg_a_m, *arg_b_m);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    std::ostringstream out;
                    out << "corr =\n";
                    print_matrix(out, *value);
                    return out.str();
                }
                if (fn == "poly_add") {
                    auto value = eval_poly_add(*arg_a_m, *arg_b_m);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    std::ostringstream out;
                    out << "sum =\n";
                    print_matrix(out, *value);
                    return out.str();
                }
                if (fn == "ml_mat_mul") {
                    auto value = eval_ml_mat_mul(*arg_a_m, *arg_b_m);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    std::ostringstream out;
                    out << "C =\n";
                    print_matrix(out, *value);
                    return out.str();
                }
                if (fn == "poly_mul") {
                    auto value = eval_poly_mul(*arg_a_m, *arg_b_m);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    std::ostringstream out;
                    out << "prod =\n";
                    print_matrix(out, *value);
                    return out.str();
                }
                if (fn == "poly_sub") {
                    auto value = eval_poly_sub(*arg_a_m, *arg_b_m);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    std::ostringstream out;
                    out << "diff =\n";
                    print_matrix(out, *value);
                    return out.str();
                }
                if (fn == "poly_compose") {
                    auto value = eval_poly_compose(*arg_a_m, *arg_b_m);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    std::ostringstream out;
                    out << "composed =\n";
                    print_matrix(out, *value);
                    return out.str();
                }
                if (fn == "quantum_tensor_product") {
                    auto value = eval_quantum_tensor_product(*arg_a_m, *arg_b_m);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    std::ostringstream out;
                    out << "op =\n";
                    print_matrix(out, *value);
                    return out.str();
                }
            }
        }
        if (is_matrix_triple_matrix_call_callee(fn)) {
            const auto call_args = split_call_args(cmd);
            if (call_args && call_args->size() == 3) {
                auto resolve_arg = [this](const std::string& text) -> Result<Matrix<double>> {
                    auto matrix = parse_matrix(text);
                    if (!matrix) {
                        matrix = resolve_matrix(text);
                    }
                    return matrix;
                };
                auto arg_a_m = resolve_arg(call_args->at(0));
                if (!arg_a_m) {
                    return std::unexpected(arg_a_m.error());
                }
                auto arg_b_m = resolve_arg(call_args->at(1));
                if (!arg_b_m) {
                    return std::unexpected(arg_b_m.error());
                }
                auto arg_c_m = resolve_arg(call_args->at(2));
                if (!arg_c_m) {
                    return std::unexpected(arg_c_m.error());
                }
                if (fn == "control_place") {
                    auto value = eval_control_place(*arg_a_m, *arg_b_m, *arg_c_m);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    std::ostringstream out;
                    out << "K =\n";
                    print_matrix(out, *value);
                    return out.str();
                }
            }
        }
        if (is_matrix_quad_matrix_call_callee(fn)) {
            const auto call_args = split_call_args(cmd);
            if (call_args && call_args->size() == 4) {
                auto resolve_arg = [this](const std::string& text) -> Result<Matrix<double>> {
                    auto matrix = parse_matrix(text);
                    if (!matrix) {
                        matrix = resolve_matrix(text);
                    }
                    return matrix;
                };
                auto arg_a_m = resolve_arg(call_args->at(0));
                if (!arg_a_m) {
                    return std::unexpected(arg_a_m.error());
                }
                auto arg_b_m = resolve_arg(call_args->at(1));
                if (!arg_b_m) {
                    return std::unexpected(arg_b_m.error());
                }
                auto arg_c_m = resolve_arg(call_args->at(2));
                if (!arg_c_m) {
                    return std::unexpected(arg_c_m.error());
                }
                auto arg_d_m = resolve_arg(call_args->at(3));
                if (!arg_d_m) {
                    return std::unexpected(arg_d_m.error());
                }
                if (fn == "control_lqr") {
                    auto value = eval_control_lqr(*arg_a_m, *arg_b_m, *arg_c_m, *arg_d_m);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    std::ostringstream out;
                    out << "K =\n";
                    print_matrix(out, *value);
                    return out.str();
                }
                if (fn == "control_riccati") {
                    auto value = eval_control_riccati(*arg_a_m, *arg_b_m, *arg_c_m, *arg_d_m);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    std::ostringstream out;
                    out << "X =\n";
                    print_matrix(out, *value);
                    return out.str();
                }
                if (fn == "control_dare") {
                    auto value = eval_control_dare(*arg_a_m, *arg_b_m, *arg_c_m, *arg_d_m);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    std::ostringstream out;
                    out << "X =\n";
                    print_matrix(out, *value);
                    return out.str();
                }
            }
        }
        if (is_scalar_dual_matrix_call_callee(fn)) {
            const auto call_args = split_call_args(cmd);
            if (call_args && call_args->size() == 2) {
                auto resolve_arg = [this](const std::string& text) -> Result<Matrix<double>> {
                    auto matrix = parse_matrix(text);
                    if (!matrix) {
                        matrix = resolve_matrix(text);
                    }
                    return matrix;
                };
                auto arg_a_m = resolve_arg(call_args->at(0));
                if (!arg_a_m) {
                    return std::unexpected(arg_a_m.error());
                }
                auto arg_b_m = resolve_arg(call_args->at(1));
                if (!arg_b_m) {
                    return std::unexpected(arg_b_m.error());
                }
                if (fn == "control_step_final") {
                    auto value = eval_control_step_final(*arg_a_m, *arg_b_m);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    return std::to_string(*value) + "\n";
                }
                if (fn == "control_pidtune_kp") {
                    auto value = eval_control_pidtune_kp(*arg_a_m, *arg_b_m);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    return std::to_string(*value) + "\n";
                }
                if (fn == "control_pidtune_ki") {
                    auto value = eval_control_pidtune_ki(*arg_a_m, *arg_b_m);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    return std::to_string(*value) + "\n";
                }
                if (fn == "control_pidtune_kd") {
                    auto value = eval_control_pidtune_kd(*arg_a_m, *arg_b_m);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    return std::to_string(*value) + "\n";
                }
                if (fn == "control_impulse_final") {
                    auto value = eval_control_impulse_final(*arg_a_m, *arg_b_m);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    return std::to_string(*value) + "\n";
                }
                if (fn == "control_dcgain") {
                    auto value = eval_control_dcgain(*arg_a_m, *arg_b_m);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    return std::to_string(*value) + "\n";
                }
                if (fn == "control_phase_margin") {
                    auto value = eval_control_phase_margin(*arg_a_m, *arg_b_m);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    return std::to_string(*value) + "\n";
                }
                if (fn == "control_gain_margin") {
                    auto value = eval_control_gain_margin(*arg_a_m, *arg_b_m);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    return std::to_string(*value) + "\n";
                }
                if (fn == "control_is_stable") {
                    auto value = eval_control_is_stable(*arg_a_m, *arg_b_m);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    return std::to_string(*value) + "\n";
                }
                if (fn == "control_is_controllable") {
                    auto value = eval_control_is_controllable(*arg_a_m, *arg_b_m);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    return std::to_string(*value) + "\n";
                }
                if (fn == "control_is_observable") {
                    auto value = eval_control_is_observable(*arg_a_m, *arg_b_m);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    return std::to_string(*value) + "\n";
                }
                if (fn == "numthy_crt") {
                    auto value = eval_numthy_crt(*arg_a_m, *arg_b_m);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    return std::to_string(*value) + "\n";
                }
                if (fn == "info_kl_divergence") {
                    auto value = eval_info_kl_divergence(*arg_a_m, *arg_b_m);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    return std::to_string(*value) + "\n";
                }
                if (fn == "tensorops_inner") {
                    auto value = eval_tensorops_inner(*arg_a_m, *arg_b_m);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    return std::to_string(*value) + "\n";
                }
                if (fn == "finance_portfolio_return") {
                    auto value = eval_finance_portfolio_return(*arg_a_m, *arg_b_m);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    return std::to_string(*value) + "\n";
                }
                if (fn == "finance_portfolio_variance") {
                    auto value = eval_finance_portfolio_variance(*arg_a_m, *arg_b_m);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    return std::to_string(*value) + "\n";
                }
                if (fn == "info_cross_entropy") {
                    auto value = eval_info_cross_entropy(*arg_a_m, *arg_b_m);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    return std::to_string(*value) + "\n";
                }
                if (fn == "info_js_divergence") {
                    auto value = eval_info_js_divergence(*arg_a_m, *arg_b_m);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    return std::to_string(*value) + "\n";
                }
                if (fn == "info_tv_distance") {
                    auto value = eval_info_tv_distance(*arg_a_m, *arg_b_m);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    return std::to_string(*value) + "\n";
                }
                if (fn == "info_hellinger_dist") {
                    auto value = eval_info_hellinger_dist(*arg_a_m, *arg_b_m);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    return std::to_string(*value) + "\n";
                }
                if (fn == "quantum_fidelity") {
                    auto value = eval_quantum_fidelity(*arg_a_m, *arg_b_m);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    return std::to_string(*value) + "\n";
                }
                if (fn == "stats_correlation") {
                    auto value = eval_stats_correlation(*arg_a_m, *arg_b_m);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    return std::to_string(*value) + "\n";
                }
                if (fn == "stats_spearman") {
                    auto value = eval_stats_spearman(*arg_a_m, *arg_b_m);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    return std::to_string(*value) + "\n";
                }
                if (fn == "stats_kendall") {
                    auto value = eval_stats_kendall(*arg_a_m, *arg_b_m);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    return std::to_string(*value) + "\n";
                }
                if (fn == "stats_two_sample_ttest") {
                    auto value = eval_stats_two_sample_ttest(*arg_a_m, *arg_b_m);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    return std::to_string(*value) + "\n";
                }
                if (fn == "stats_chi2_gof") {
                    auto value = eval_stats_chi2_gof(*arg_a_m, *arg_b_m);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    return std::to_string(*value) + "\n";
                }
                if (fn == "quantum_expectation_dm") {
                    auto value = eval_quantum_expectation_dm(*arg_a_m, *arg_b_m);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    return std::to_string(*value) + "\n";
                }
                if (fn == "quantum_expectation") {
                    auto value = eval_quantum_expectation(*arg_a_m, *arg_b_m);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    return std::to_string(*value) + "\n";
                }
                if (fn == "quantum_inner") {
                    auto value = eval_quantum_inner(*arg_a_m, *arg_b_m);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    return std::to_string(*value) + "\n";
                }
                if (fn == "quantum_trace_distance") {
                    auto value = eval_quantum_trace_distance(*arg_a_m, *arg_b_m);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    return std::to_string(*value) + "\n";
                }
                if (fn == "ml_categorical_crossentropy") {
                    auto value = eval_ml_categorical_crossentropy(*arg_a_m, *arg_b_m);
                    if (!value) {
                        return std::unexpected(value.error());
                    }
                    return std::to_string(*value) + "\n";
                }
                auto y_pred = matrix_to_ml_vec(*arg_a_m, fn.c_str());
                if (!y_pred) {
                    return std::unexpected(y_pred.error());
                }
                auto y_true = matrix_to_ml_vec(*arg_b_m, fn.c_str());
                if (!y_true) {
                    return std::unexpected(y_true.error());
                }
                auto metric = eval_ml_metric(fn, *y_pred, *y_true);
                if (!metric) {
                    return std::unexpected(metric.error());
                }
                return std::to_string(*metric) + "\n";
            }
        }
    }

    static const std::regex binary(R"((\w+)\(([^,]+),([^)]+)\))", std::regex::icase);
    if (std::regex_match(cmd, match, binary)) {
        const std::string fn = lower(match[1].str());
        const std::string arg_a = trim(match[2].str());
        const std::string arg_b = trim(match[3].str());

        if (fn == "spherical_jn") {
            double n = 0.0;
            double x = 0.0;
            if (!parse_number(arg_a, n) || !parse_number(arg_b, x)) {
                return std::unexpected(DomainError{"spherical_jn", "expected spherical_jn(n,x)"});
            }
            return std::to_string(spherical_jn(static_cast<int>(n), x)) + "\n";
        }

        if (fn == "kelvin_ber") {
            double nu = 0.0;
            double x = 0.0;
            if (!parse_number(arg_a, nu) || !parse_number(arg_b, x)) {
                return std::unexpected(DomainError{"kelvin_ber", "expected kelvin_ber(nu,x)"});
            }
            return std::to_string(kelvin_ber(static_cast<int>(nu), x)) + "\n";
        }

        if (fn == "struve_h") {
            double nu = 0.0;
            double x = 0.0;
            if (!parse_number(arg_a, nu) || !parse_number(arg_b, x)) {
                return std::unexpected(DomainError{"struve_h", "expected struve_h(nu,x)"});
            }
            return std::to_string(struve_h(static_cast<int>(nu), x)) + "\n";
        }

        if (fn == "bessel_zero_jnu") {
            double nu = 0.0;
            double n = 0.0;
            if (!parse_number(arg_a, nu) || !parse_number(arg_b, n)) {
                return std::unexpected(DomainError{"bessel_zero_jnu", "expected bessel_zero_jnu(nu,n)"});
            }
            return std::to_string(bessel_zero_jnu(static_cast<int>(nu), static_cast<int>(n))) + "\n";
        }

        if (fn == "jacobi_sn") {
            double u = 0.0;
            double k = 0.0;
            if (!parse_number(arg_a, u) || !parse_number(arg_b, k)) {
                return std::unexpected(DomainError{"jacobi_sn", "expected jacobi_sn(u,k)"});
            }
            return std::to_string(jacobi_sn(u, k)) + "\n";
        }

        if (fn == "theta3") {
            double z = 0.0;
            double q = 0.0;
            if (!parse_number(arg_a, z) || !parse_number(arg_b, q)) {
                return std::unexpected(DomainError{"theta3", "expected theta3(z,q)"});
            }
            return std::to_string(theta3(z, q)) + "\n";
        }

        if (fn == "polylog") {
            double n = 0.0;
            double z = 0.0;
            if (!parse_number(arg_a, n) || !parse_number(arg_b, z)) {
                return std::unexpected(DomainError{"polylog", "expected polylog(n,z)"});
            }
            return std::to_string(polylog(static_cast<int>(n), z)) + "\n";
        }

        if (fn == "debye") {
            double n = 0.0;
            double x = 0.0;
            if (!parse_number(arg_a, n) || !parse_number(arg_b, x)) {
                return std::unexpected(DomainError{"debye", "expected debye(n,x)"});
            }
            return std::to_string(debye(static_cast<int>(n), x)) + "\n";
        }

        if (fn == "beta") {
            double a = 0.0;
            double b = 0.0;
            if (!parse_number(arg_a, a) || !parse_number(arg_b, b)) {
                return std::unexpected(DomainError{"beta", "expected numeric arguments beta(a,b)"});
            }
            return std::to_string(beta_func(a, b)) + "\n";
        }

        if (fn == "prob_exp_cdf") {
            double x = 0.0;
            double lambda = 0.0;
            if (!parse_number(arg_a, x) || !parse_number(arg_b, lambda)) {
                return std::unexpected(
                    DomainError{"prob_exp_cdf", "expected prob_exp_cdf(x,lambda)"});
            }
            return std::to_string(exp_cdf(x, lambda)) + "\n";
        }

        if (fn == "prob_pois_pdf") {
            double k = 0.0;
            double lambda = 0.0;
            if (!parse_number(arg_a, k) || !parse_number(arg_b, lambda)) {
                return std::unexpected(
                    DomainError{"prob_pois_pdf", "expected prob_pois_pdf(k,lambda)"});
            }
            return std::to_string(pois_pdf(k, lambda)) + "\n";
        }

        if (fn == "prob_pois_cdf") {
            double k = 0.0;
            double lambda = 0.0;
            if (!parse_number(arg_a, k) || !parse_number(arg_b, lambda)) {
                return std::unexpected(
                    DomainError{"prob_pois_cdf", "expected prob_pois_cdf(k,lambda)"});
            }
            return std::to_string(pois_cdf(k, lambda)) + "\n";
        }

        if (fn == "prob_exp_pdf") {
            double x = 0.0;
            double lambda = 0.0;
            if (!parse_number(arg_a, x) || !parse_number(arg_b, lambda)) {
                return std::unexpected(
                    DomainError{"prob_exp_pdf", "expected prob_exp_pdf(x,lambda)"});
            }
            return std::to_string(exp_pdf(x, lambda)) + "\n";
        }

        if (fn == "prob_chi2_cdf") {
            double x = 0.0;
            double df = 0.0;
            if (!parse_number(arg_a, x) || !parse_number(arg_b, df)) {
                return std::unexpected(
                    DomainError{"prob_chi2_cdf", "expected prob_chi2_cdf(x,df)"});
            }
            return std::to_string(chi2_cdf(x, df)) + "\n";
        }

        if (fn == "prob_t_cdf") {
            double x = 0.0;
            double df = 0.0;
            if (!parse_number(arg_a, x) || !parse_number(arg_b, df)) {
                return std::unexpected(
                    DomainError{"prob_t_cdf", "expected prob_t_cdf(x,df)"});
            }
            return std::to_string(t_cdf(x, df)) + "\n";
        }

        if (fn == "prob_chi2_pdf") {
            double x = 0.0;
            double df = 0.0;
            if (!parse_number(arg_a, x) || !parse_number(arg_b, df)) {
                return std::unexpected(
                    DomainError{"prob_chi2_pdf", "expected prob_chi2_pdf(x,df)"});
            }
            return std::to_string(chi2_pdf(x, df)) + "\n";
        }

        if (fn == "combo_nchoosek") {
            double n_d = 0.0;
            double k_d = 0.0;
            if (!parse_number(arg_a, n_d) || !parse_number(arg_b, k_d)) {
                return std::unexpected(DomainError{"combo_nchoosek", "expected combo_nchoosek(n,k)"});
            }
            const int n = static_cast<int>(n_d);
            const int k = static_cast<int>(k_d);
            if (n < 0 || k < 0 || k > n || n_d != n || k_d != k) {
                return std::unexpected(DomainError{"combo_nchoosek", "expected 0 <= k <= n"});
            }
            return std::to_string(combo::binomial(static_cast<uint32_t>(n),
                                                  static_cast<uint32_t>(k))) +
                   "\n";
        }

        if (fn == "diffgeo_gaussian_curvature_sphere") {
            double u = 0.0;
            double v = 0.0;
            if (!parse_number(arg_a, u) || !parse_number(arg_b, v)) {
                return std::unexpected(DomainError{
                    "diffgeo_gaussian_curvature_sphere",
                    "expected diffgeo_gaussian_curvature_sphere(u,v)"});
            }
            auto value = eval_diffgeo_gaussian_curvature_sphere(u, v);
            if (!value) {
                return std::unexpected(value.error());
            }
            return std::to_string(*value) + "\n";
        }

        if (fn == "diffgeo_mean_curvature_sphere") {
            double u = 0.0;
            double v = 0.0;
            if (!parse_number(arg_a, u) || !parse_number(arg_b, v)) {
                return std::unexpected(DomainError{
                    "diffgeo_mean_curvature_sphere",
                    "expected diffgeo_mean_curvature_sphere(u,v)"});
            }
            auto value = eval_diffgeo_mean_curvature_sphere(u, v);
            if (!value) {
                return std::unexpected(value.error());
            }
            return std::to_string(*value) + "\n";
        }

        if (fn == "diffgeo_ricci_scalar_sphere") {
            double u = 0.0;
            double v = 0.0;
            if (!parse_number(arg_a, u) || !parse_number(arg_b, v)) {
                return std::unexpected(DomainError{
                    "diffgeo_ricci_scalar_sphere",
                    "expected diffgeo_ricci_scalar_sphere(u,v)"});
            }
            auto value = eval_diffgeo_ricci_scalar_sphere(u, v);
            if (!value) {
                return std::unexpected(value.error());
            }
            return std::to_string(*value) + "\n";
        }

        if (fn == "diffgeo_einstein_scalar_sphere") {
            double u = 0.0;
            double v = 0.0;
            if (!parse_number(arg_a, u) || !parse_number(arg_b, v)) {
                return std::unexpected(DomainError{
                    "diffgeo_einstein_scalar_sphere",
                    "expected diffgeo_einstein_scalar_sphere(u,v)"});
            }
            auto value = eval_diffgeo_einstein_scalar_sphere(u, v);
            if (!value) {
                return std::unexpected(value.error());
            }
            return std::to_string(*value) + "\n";
        }

        if (fn == "cplx_residue_inv") {
            double re = 0.0;
            double im = 0.0;
            if (!parse_number(arg_a, re) || !parse_number(arg_b, im)) {
                return std::unexpected(DomainError{
                    "cplx_residue_inv", "expected cplx_residue_inv(pole_re,pole_im)"});
            }
            auto value = eval_cplx_residue_inv(re, im);
            if (!value) {
                return std::unexpected(value.error());
            }
            return std::to_string(*value) + "\n";
        }

        if (fn == "boxfilter" || fn == "medfilt2" || fn == "imgaussfilt") {
            auto resolve_arg = [this](const std::string& text) -> Result<Matrix<double>> {
                auto matrix = parse_matrix(text);
                if (!matrix) {
                    matrix = resolve_matrix(text);
                }
                return matrix;
            };
            auto gray_m = resolve_arg(arg_a);
            if (!gray_m) {
                return std::unexpected(gray_m.error());
            }
            double param_d = 0.0;
            if (!parse_number(arg_b, param_d)) {
                return std::unexpected(DomainError{fn, "expected numeric second argument"});
            }
            auto gray = matrix_to_gray_image(*gray_m);
            if (!gray) {
                return std::unexpected(gray.error());
            }
            Matrix<double> filtered_m;
            if (fn == "boxfilter") {
                const int ksize = static_cast<int>(param_d);
                if (ksize < 1 || param_d != ksize || (ksize % 2) == 0) {
                    return std::unexpected(
                        DomainError{"boxfilter", "expected positive odd integer ksize"});
                }
                filtered_m = gray_image_to_matrix(image::boxfilter(*gray, ksize));
            } else if (fn == "medfilt2") {
                const int ksize = static_cast<int>(param_d);
                if (ksize < 1 || param_d != ksize || (ksize % 2) == 0) {
                    return std::unexpected(
                        DomainError{"medfilt2", "expected positive odd integer ksize"});
                }
                filtered_m = gray_image_to_matrix(image::medfilt2(*gray, ksize));
            } else {
                filtered_m = gray_image_to_matrix(
                    image::imgaussfilt(*gray, static_cast<float>(param_d)));
            }
            std::ostringstream out;
            out << "filtered =\n";
            print_matrix(out, filtered_m);
            return out.str();
        }

        if (fn == "combo_unrank_permutation") {
            double n_d = 0.0;
            double rank_d = 0.0;
            if (!parse_number(arg_a, n_d) || !parse_number(arg_b, rank_d)) {
                return std::unexpected(DomainError{
                    "combo_unrank_permutation", "expected combo_unrank_permutation(n,rank)"});
            }
            const int n = static_cast<int>(n_d);
            if (n < 0 || n_d != n || rank_d < 0.0 || std::floor(rank_d) != rank_d) {
                return std::unexpected(DomainError{
                    "combo_unrank_permutation", "expected non-negative integer n and rank"});
            }
            auto value = eval_combo_unrank_permutation(n, static_cast<uint64_t>(rank_d));
            if (!value) {
                return std::unexpected(value.error());
            }
            std::ostringstream out;
            out << "perm =\n";
            print_matrix(out, *value);
            return out.str();
        }

        if (fn == "numthy_continued_fraction") {
            double x = 0.0;
            double n_d = 0.0;
            if (!parse_number(arg_a, x) || !parse_number(arg_b, n_d)) {
                return std::unexpected(DomainError{
                    "numthy_continued_fraction", "expected numthy_continued_fraction(x,n)"});
            }
            const int n = static_cast<int>(n_d);
            if (n < 1 || n_d != n) {
                return std::unexpected(DomainError{
                    "numthy_continued_fraction", "expected positive integer n"});
            }
            auto value = eval_numthy_continued_fraction(x, n);
            if (!value) {
                return std::unexpected(value.error());
            }
            std::ostringstream out;
            out << "cf =\n";
            print_matrix(out, *value);
            return out.str();
        }

        if (fn == "numthy_primes") {
            double lo_d = 0.0;
            double hi_d = 0.0;
            if (!parse_number(arg_a, lo_d) || !parse_number(arg_b, hi_d)) {
                return std::unexpected(DomainError{
                    "numthy_primes", "expected numthy_primes(lo,hi)"});
            }
            if (lo_d < 0.0 || hi_d < 0.0 || std::floor(lo_d) != lo_d || std::floor(hi_d) != hi_d) {
                return std::unexpected(
                    DomainError{"numthy_primes", "expected non-negative integer bounds"});
            }
            auto value = eval_numthy_primes(static_cast<uint64_t>(lo_d), static_cast<uint64_t>(hi_d));
            if (!value) {
                return std::unexpected(value.error());
            }
            std::ostringstream out;
            out << "primes =\n";
            print_matrix(out, *value);
            return out.str();
        }

        if (fn == "combo_stirling2") {
            double n_d = 0.0;
            double k_d = 0.0;
            if (!parse_number(arg_a, n_d) || !parse_number(arg_b, k_d)) {
                return std::unexpected(DomainError{"combo_stirling2", "expected combo_stirling2(n,k)"});
            }
            const int n = static_cast<int>(n_d);
            const int k = static_cast<int>(k_d);
            if (n < 0 || k < 0 || k > n || n_d != n || k_d != k) {
                return std::unexpected(DomainError{"combo_stirling2", "expected 0 <= k <= n"});
            }
            return std::to_string(combo::stirling2(static_cast<uint32_t>(n),
                                                   static_cast<uint32_t>(k))) +
                   "\n";
        }

        if (fn == "combo_stirling1") {
            double n_d = 0.0;
            double k_d = 0.0;
            if (!parse_number(arg_a, n_d) || !parse_number(arg_b, k_d)) {
                return std::unexpected(DomainError{"combo_stirling1", "expected combo_stirling1(n,k)"});
            }
            const int n = static_cast<int>(n_d);
            const int k = static_cast<int>(k_d);
            if (n < 0 || k < 0 || k > n || n_d != n || k_d != k) {
                return std::unexpected(DomainError{"combo_stirling1", "expected 0 <= k <= n"});
            }
            return std::to_string(combo::stirling1(static_cast<uint32_t>(n),
                                                   static_cast<uint32_t>(k))) +
                   "\n";
        }

        if (fn == "combo_permutations") {
            double n_d = 0.0;
            double k_d = 0.0;
            if (!parse_number(arg_a, n_d) || !parse_number(arg_b, k_d)) {
                return std::unexpected(
                    DomainError{"combo_permutations", "expected combo_permutations(n,k)"});
            }
            const int n = static_cast<int>(n_d);
            const int k = static_cast<int>(k_d);
            if (n < 0 || k < 0 || k > n || n_d != n || k_d != k) {
                return std::unexpected(DomainError{"combo_permutations", "expected 0 <= k <= n"});
            }
            return std::to_string(combo::permutations(static_cast<uint32_t>(n),
                                                        static_cast<uint32_t>(k))) +
                   "\n";
        }

        if (fn == "numthy_gcd") {
            double a_d = 0.0;
            double b_d = 0.0;
            if (!parse_number(arg_a, a_d) || !parse_number(arg_b, b_d)) {
                return std::unexpected(DomainError{"numthy_gcd", "expected numthy_gcd(a,b)"});
            }
            return std::to_string(numthy::gcd(static_cast<uint64_t>(a_d),
                                              static_cast<uint64_t>(b_d))) +
                   "\n";
        }

        if (fn == "numthy_lcm") {
            double a_d = 0.0;
            double b_d = 0.0;
            if (!parse_number(arg_a, a_d) || !parse_number(arg_b, b_d)) {
                return std::unexpected(DomainError{"numthy_lcm", "expected numthy_lcm(a,b)"});
            }
            return std::to_string(numthy::lcm(static_cast<uint64_t>(a_d),
                                              static_cast<uint64_t>(b_d))) +
                   "\n";
        }

        if (fn == "combo_combinations_with_rep") {
            double n_d = 0.0;
            double k_d = 0.0;
            if (!parse_number(arg_a, n_d) || !parse_number(arg_b, k_d)) {
                return std::unexpected(DomainError{
                    "combo_combinations_with_rep",
                    "expected combo_combinations_with_rep(n,k)"});
            }
            const int n = static_cast<int>(n_d);
            const int k = static_cast<int>(k_d);
            if (n < 0 || k < 0 || n_d != n || k_d != k) {
                return std::unexpected(
                    DomainError{"combo_combinations_with_rep",
                                "expected non-negative integer n and k"});
            }
            return std::to_string(combo::combinations_with_rep(static_cast<uint32_t>(n),
                                                               static_cast<uint32_t>(k))) +
                   "\n";
        }

        if (fn == "numthy_legendre_symbol") {
            double a_d = 0.0;
            double p_d = 0.0;
            if (!parse_number(arg_a, a_d) || !parse_number(arg_b, p_d)) {
                return std::unexpected(DomainError{
                    "numthy_legendre_symbol", "expected numthy_legendre_symbol(a,p)"});
            }
            if (std::floor(a_d) != a_d || std::floor(p_d) != p_d) {
                return std::unexpected(
                    DomainError{"numthy_legendre_symbol", "expected integer arguments"});
            }
            if (p_d < 0.0) {
                return std::unexpected(
                    DomainError{"numthy_legendre_symbol", "expected odd prime p"});
            }
            return std::to_string(numthy::legendre_symbol(static_cast<int64_t>(a_d),
                                                          static_cast<uint64_t>(p_d))) +
                   "\n";
        }

        if (fn == "numthy_jacobi_symbol") {
            double a_d = 0.0;
            double n_d = 0.0;
            if (!parse_number(arg_a, a_d) || !parse_number(arg_b, n_d)) {
                return std::unexpected(DomainError{
                    "numthy_jacobi_symbol", "expected numthy_jacobi_symbol(a,n)"});
            }
            if (std::floor(a_d) != a_d || std::floor(n_d) != n_d) {
                return std::unexpected(
                    DomainError{"numthy_jacobi_symbol", "expected integer arguments"});
            }
            if (n_d <= 0.0 || static_cast<uint64_t>(n_d) % 2u == 0u) {
                return std::unexpected(
                    DomainError{"numthy_jacobi_symbol", "expected odd positive integer n"});
            }
            return std::to_string(numthy::jacobi_symbol(static_cast<int64_t>(a_d),
                                                        static_cast<uint64_t>(n_d))) +
                   "\n";
        }

        if (fn == "numthy_kronecker_symbol") {
            double a_d = 0.0;
            double n_d = 0.0;
            if (!parse_number(arg_a, a_d) || !parse_number(arg_b, n_d)) {
                return std::unexpected(DomainError{
                    "numthy_kronecker_symbol", "expected numthy_kronecker_symbol(a,n)"});
            }
            if (std::floor(a_d) != a_d || std::floor(n_d) != n_d) {
                return std::unexpected(
                    DomainError{"numthy_kronecker_symbol", "expected integer arguments"});
            }
            return std::to_string(numthy::kronecker_symbol(static_cast<int64_t>(a_d),
                                                           static_cast<int64_t>(n_d))) +
                   "\n";
        }

        if (fn == "numthy_tonelli_shanks") {
            double n_d = 0.0;
            double p_d = 0.0;
            if (!parse_number(arg_a, n_d) || !parse_number(arg_b, p_d)) {
                return std::unexpected(DomainError{
                    "numthy_tonelli_shanks", "expected numthy_tonelli_shanks(n,p)"});
            }
            auto root = eval_numthy_tonelli_shanks(n_d, p_d);
            if (!root) {
                return std::unexpected(root.error());
            }
            return std::to_string(*root) + "\n";
        }

        if (fn == "numthy_mod_inv") {
            double a_d = 0.0;
            double m_d = 0.0;
            if (!parse_number(arg_a, a_d) || !parse_number(arg_b, m_d)) {
                return std::unexpected(DomainError{
                    "numthy_mod_inv", "expected numthy_mod_inv(a,m)"});
            }
            auto inv = eval_numthy_mod_inv(a_d, m_d);
            if (!inv) {
                return std::unexpected(inv.error());
            }
            return std::to_string(*inv) + "\n";
        }

        if (fn == "numthy_extended_gcd") {
            double a_d = 0.0;
            double b_d = 0.0;
            if (!parse_number(arg_a, a_d) || !parse_number(arg_b, b_d)) {
                return std::unexpected(DomainError{
                    "numthy_extended_gcd", "expected numthy_extended_gcd(a,b)"});
            }
            auto g = eval_numthy_extended_gcd(a_d, b_d);
            if (!g) {
                return std::unexpected(g.error());
            }
            return std::to_string(*g) + "\n";
        }

        if (fn == "numthy_is_primitive_root") {
            double g_d = 0.0;
            double p_d = 0.0;
            if (!parse_number(arg_a, g_d) || !parse_number(arg_b, p_d)) {
                return std::unexpected(DomainError{
                    "numthy_is_primitive_root", "expected numthy_is_primitive_root(g,p)"});
            }
            if (std::floor(g_d) != g_d || std::floor(p_d) != p_d) {
                return std::unexpected(
                    DomainError{"numthy_is_primitive_root", "expected integer arguments"});
            }
            if (g_d < 0.0 || p_d <= 0.0) {
                return std::unexpected(
                    DomainError{"numthy_is_primitive_root", "expected g >= 0 and p > 0"});
            }
            return std::to_string(numthy::is_primitive_root(static_cast<uint64_t>(g_d),
                                                            static_cast<uint64_t>(p_d))
                                      ? 1
                                      : 0) +
                   "\n";
        }

        if (fn == "cplx_joukowski") {
            double re = 0.0;
            double im = 0.0;
            if (!parse_number(arg_a, re) || !parse_number(arg_b, im)) {
                return std::unexpected(DomainError{"cplx_joukowski", "expected cplx_joukowski(re,im)"});
            }
            const cplx::C w = cplx::joukowski(cplx::C{re, im});
            return std::to_string(std::abs(w)) + "\n";
        }

        if (fn == "cplx_joukowski_inv") {
            double re = 0.0;
            double im = 0.0;
            if (!parse_number(arg_a, re) || !parse_number(arg_b, im)) {
                return std::unexpected(
                    DomainError{"cplx_joukowski_inv", "expected cplx_joukowski_inv(re,im)"});
            }
            auto value = eval_cplx_joukowski_inv(re, im);
            if (!value) {
                return std::unexpected(value.error());
            }
            return std::to_string(*value) + "\n";
        }

        if (fn == "geo_vec2d_length") {
            double x = 0.0;
            double y = 0.0;
            if (!parse_number(arg_a, x) || !parse_number(arg_b, y)) {
                return std::unexpected(
                    DomainError{"geo_vec2d_length", "expected geo_vec2d_length(x,y)"});
            }
            return std::to_string(geo::length(geo::Vec2D{x, y})) + "\n";
        }

        if (fn == "finance_npv") {
            double rate = 0.0;
            if (!parse_number(arg_a, rate)) {
                return std::unexpected(DomainError{"finance_npv", "expected finance_npv(rate,cf)"});
            }
            auto resolve_arg = [this](const std::string& text) -> Result<Matrix<double>> {
                auto matrix = parse_matrix(text);
                if (!matrix) {
                    matrix = resolve_matrix(text);
                }
                return matrix;
            };
            auto cashflows = resolve_arg(arg_b);
            if (!cashflows) {
                return std::unexpected(cashflows.error());
            }
            auto value = eval_finance_npv(rate, *cashflows);
            if (!value) {
                return std::unexpected(value.error());
            }
            return std::to_string(*value) + "\n";
        }

        if (fn == "combo_multinomial") {
            double n_d = 0.0;
            if (!parse_number(arg_a, n_d)) {
                auto scalar_expr = eval_scalar_expr(state_, arg_a);
                if (!scalar_expr) {
                    return std::unexpected(scalar_expr.error());
                }
                n_d = *scalar_expr;
            }
            auto resolve_arg = [this](const std::string& text) -> Result<Matrix<double>> {
                auto matrix = parse_matrix(text);
                if (!matrix) {
                    matrix = resolve_matrix(text);
                }
                return matrix;
            };
            auto ks = resolve_arg(arg_b);
            if (!ks) {
                return std::unexpected(ks.error());
            }
            auto value = eval_combo_multinomial(n_d, *ks);
            if (!value) {
                return std::unexpected(value.error());
            }
            return std::to_string(*value) + "\n";
        }

        if (fn == "finance_kelly_fraction") {
            double win_prob = 0.0;
            double win_loss_ratio = 0.0;
            if (!parse_number(arg_a, win_prob) || !parse_number(arg_b, win_loss_ratio)) {
                return std::unexpected(
                    DomainError{"finance_kelly_fraction",
                                  "expected finance_kelly_fraction(win_prob,win_loss_ratio)"});
            }
            return std::to_string(finance::kelly_fraction(win_prob, win_loss_ratio)) + "\n";
        }

        if (fn == "info_shannon_hartley") {
            double bandwidth_hz = 0.0;
            double snr_linear = 0.0;
            if (!parse_number(arg_a, bandwidth_hz) || !parse_number(arg_b, snr_linear)) {
                return std::unexpected(
                    DomainError{"info_shannon_hartley",
                                  "expected info_shannon_hartley(bandwidth_hz,snr_linear)"});
            }
            return std::to_string(info::shannon_hartley(bandwidth_hz, snr_linear)) + "\n";
        }

        if (fn == "info_rate_distortion_gaussian") {
            double variance = 0.0;
            double distortion = 0.0;
            if (!parse_number(arg_a, variance) || !parse_number(arg_b, distortion)) {
                return std::unexpected(
                    DomainError{"info_rate_distortion_gaussian",
                                  "expected info_rate_distortion_gaussian(variance,distortion)"});
            }
            return std::to_string(info::rate_distortion_gaussian(variance, distortion)) + "\n";
        }

        if (fn == "info_differential_entropy_uniform") {
            double a = 0.0;
            double b = 0.0;
            if (!parse_number(arg_a, a) || !parse_number(arg_b, b)) {
                return std::unexpected(
                    DomainError{"info_differential_entropy_uniform",
                                  "expected info_differential_entropy_uniform(a,b)"});
            }
            return std::to_string(info::differential_entropy_uniform(a, b)) + "\n";
        }

        if (fn == "info_renyi_entropy") {
            double alpha = 0.0;
            if (!parse_number(arg_a, alpha)) {
                return std::unexpected(
                    DomainError{"info_renyi_entropy", "expected info_renyi_entropy(alpha,p)"});
            }
            auto resolve_arg = [this](const std::string& text) -> Result<Matrix<double>> {
                auto matrix = parse_matrix(text);
                if (!matrix) {
                    matrix = resolve_matrix(text);
                }
                return matrix;
            };
            auto probs = resolve_arg(arg_b);
            if (!probs) {
                return std::unexpected(probs.error());
            }
            auto value = eval_info_renyi_entropy(alpha, *probs);
            if (!value) {
                return std::unexpected(value.error());
            }
            return std::to_string(*value) + "\n";
        }

        if (fn == "info_tsallis_entropy") {
            double q_param = 0.0;
            if (!parse_number(arg_a, q_param)) {
                return std::unexpected(
                    DomainError{"info_tsallis_entropy", "expected info_tsallis_entropy(q,p)"});
            }
            auto resolve_arg = [this](const std::string& text) -> Result<Matrix<double>> {
                auto matrix = parse_matrix(text);
                if (!matrix) {
                    matrix = resolve_matrix(text);
                }
                return matrix;
            };
            auto probs = resolve_arg(arg_b);
            if (!probs) {
                return std::unexpected(probs.error());
            }
            auto value = eval_info_tsallis_entropy(q_param, *probs);
            if (!value) {
                return std::unexpected(value.error());
            }
            return std::to_string(*value) + "\n";
        }

        if (fn == "geo_bezier_eval_x" || fn == "geo_bezier_eval_y" || fn == "bwt_decode_vec" ||
            fn == "combo_rank_combination" || fn == "combo_next_comb" ||
            fn == "quantum_time_evolution" ||
            fn == "signal_moving_average" || fn == "graph_bfs" || fn == "graph_dfs" ||
            fn == "stats_percentile" || fn == "stats_ttest" || fn == "poly_eval" ||
            fn == "fft_irfft" || fn == "poly_integ") {
            auto resolve_arg = [this](const std::string& text) -> Result<Matrix<double>> {
                auto matrix = parse_matrix(text);
                if (!matrix) {
                    matrix = resolve_matrix(text);
                }
                return matrix;
            };
            auto ctrl = resolve_arg(arg_a);
            if (!ctrl) {
                return std::unexpected(ctrl.error());
            }
            double t = 0.0;
            if (!parse_number(arg_b, t)) {
                auto scalar_expr = eval_scalar_expr(state_, arg_b);
                if (!scalar_expr) {
                    return std::unexpected(scalar_expr.error());
                }
                t = *scalar_expr;
            }
            if (fn == "bwt_decode_vec") {
                auto value = eval_bwt_decode_vec(*ctrl, t);
                if (!value) {
                    return std::unexpected(value.error());
                }
                std::ostringstream out;
                out << "decoded =\n";
                print_matrix(out, *value);
                return out.str();
            }
            if (fn == "quantum_time_evolution") {
                auto value = eval_quantum_time_evolution_matrix(*ctrl, t);
                if (!value) {
                    return std::unexpected(value.error());
                }
                std::ostringstream out;
                out << "U =\n";
                print_matrix(out, *value);
                return out.str();
            }
            if (fn == "combo_rank_combination") {
                const int n = static_cast<int>(t);
                if (n < 0 || t != n) {
                    return std::unexpected(DomainError{
                        "combo_rank_combination", "expected non-negative integer n"});
                }
                auto value = eval_combo_rank_combination(*ctrl, n);
                if (!value) {
                    return std::unexpected(value.error());
                }
                return std::to_string(*value) + "\n";
            }
            if (fn == "signal_moving_average") {
                const int window = static_cast<int>(t);
                if (window < 1 || t != window) {
                    return std::unexpected(DomainError{
                        "signal_moving_average", "expected positive integer window"});
                }
                auto value = eval_signal_moving_average(*ctrl, static_cast<size_t>(window));
                if (!value) {
                    return std::unexpected(value.error());
                }
                std::ostringstream out;
                out << "smoothed =\n";
                print_matrix(out, *value);
                return out.str();
            }
            if (fn == "graph_bfs") {
                const int source = static_cast<int>(t);
                if (source < 0 || t != source) {
                    return std::unexpected(DomainError{
                        "graph_bfs", "expected non-negative integer source"});
                }
                auto value = eval_graph_bfs(*ctrl, source);
                if (!value) {
                    return std::unexpected(value.error());
                }
                std::ostringstream out;
                out << "order =\n";
                print_matrix(out, *value);
                return out.str();
            }
            if (fn == "graph_dfs") {
                const int source = static_cast<int>(t);
                if (source < 0 || t != source) {
                    return std::unexpected(DomainError{
                        "graph_dfs", "expected non-negative integer source"});
                }
                auto value = eval_graph_dfs(*ctrl, source);
                if (!value) {
                    return std::unexpected(value.error());
                }
                std::ostringstream out;
                out << "order =\n";
                print_matrix(out, *value);
                return out.str();
            }
            if (fn == "stats_percentile") {
                auto value = eval_stats_percentile(*ctrl, t);
                if (!value) {
                    return std::unexpected(value.error());
                }
                return std::to_string(*value) + "\n";
            }
            if (fn == "stats_ttest") {
                auto value = eval_stats_ttest(*ctrl, t);
                if (!value) {
                    return std::unexpected(value.error());
                }
                return std::to_string(*value) + "\n";
            }
            if (fn == "fft_irfft") {
                const int n = static_cast<int>(t);
                if (n < 1 || t != n) {
                    return std::unexpected(DomainError{
                        "fft_irfft", "expected positive integer n"});
                }
                auto value = eval_fft_irfft(*ctrl, static_cast<size_t>(n));
                if (!value) {
                    return std::unexpected(value.error());
                }
                std::ostringstream out;
                out << "signal =\n";
                print_matrix(out, *value);
                return out.str();
            }
            if (fn == "poly_integ") {
                auto value = eval_poly_integ(*ctrl, t);
                if (!value) {
                    return std::unexpected(value.error());
                }
                std::ostringstream out;
                out << "integ =\n";
                print_matrix(out, *value);
                return out.str();
            }
            if (fn == "poly_eval") {
                auto value = eval_poly_eval(*ctrl, t);
                if (!value) {
                    return std::unexpected(value.error());
                }
                return std::to_string(*value) + "\n";
            }
            Result<double> value =
                std::unexpected(DomainError{fn.c_str(), "unsupported bezier eval call"});
            if (fn == "geo_bezier_eval_x") {
                value = eval_geo_bezier_eval_x(*ctrl, t);
            } else {
                value = eval_geo_bezier_eval_y(*ctrl, t);
            }
            if (!value) {
                return std::unexpected(value.error());
            }
            return std::to_string(*value) + "\n";
        }

        if (fn == "bigint_gcd") {
            std::string a;
            std::string b;
            if (!parse_quoted_string(arg_a, a) || !parse_quoted_string(arg_b, b)) {
                return std::unexpected(
                    DomainError{"bigint_gcd", "expected bigint_gcd(\"a\", \"b\")"});
            }
            auto value = eval_bigint_gcd_strings(a, b);
            if (!value) {
                return std::unexpected(value.error());
            }
            return std::to_string(*value) + "\n";
        }

        if (fn == "sym_diff") {
            auto value = eval_sym_diff_strings(arg_a, arg_b);
            if (!value) {
                return std::unexpected(value.error());
            }
            return *value;
        }

        if (fn == "sym_integrate") {
            auto value = eval_sym_integrate_strings(arg_a, arg_b);
            if (!value) {
                return std::unexpected(value.error());
            }
            return *value;
        }

        if (fn == "sym_eval") {
            auto value = eval_sym_eval_strings(arg_a, arg_b);
            if (!value) {
                return std::unexpected(value.error());
            }
            return *value;
        }

        if (fn == "legendre_p") {
            double n = 0.0;
            double x = 0.0;
            if (!parse_number(arg_a, n) || !parse_number(arg_b, x)) {
                return std::unexpected(DomainError{"legendre_p", "expected legendre_p(n,x)"});
            }
            return std::to_string(legendre_p(static_cast<int>(n), x)) + "\n";
        }

        if (fn == "plot" || fn == "scatter" || fn == "solve" || fn == "matmul" ||
            fn == "tensorops_matmul" || fn == "tensorops_einsum") {
            auto A = resolve_matrix(arg_a);
            if (!A) {
                A = parse_matrix(arg_a);
            }
            auto B = resolve_matrix(arg_b);
            if (!B) {
                B = parse_matrix(arg_b);
            }
            if (A && B) {
                if (fn == "plot") {
                    return set_plot(*A, *B);
                }
                if (fn == "scatter") {
                    return set_plot(*A, *B, PlotSeries::Kind::Scatter);
                }
                if (fn == "solve") {
                    auto x = solve(*A, *B);
                    if (!x) {
                        return std::unexpected(x.error());
                    }
                    std::ostringstream out;
                    out << "x =\n";
                    print_matrix(out, *x);
                    return out.str();
                }
                Result<Matrix<double>> C;
                if (fn == "tensorops_matmul") {
                    C = eval_tensorops_matmul(*A, *B);
                } else if (fn == "tensorops_einsum") {
                    C = eval_tensorops_einsum(*A, *B);
                } else {
                    C = matmul(*A, *B);
                }
                if (!C) {
                    return std::unexpected(C.error());
                }
                std::ostringstream out;
                out << "C =\n";
                print_matrix(out, *C);
                return out.str();
            }
        }
    }

    if (const auto nullary = parse_nullary_scalar_call(cmd)) {
        auto value = eval_nullary_scalar_call(*nullary);
        if (!value) {
            return std::unexpected(value.error());
        }
        return std::to_string(*value) + "\n";
    }

    static const std::regex unary(R"((\w+)\(([^)]+)\))", std::regex::icase);
    if (std::regex_match(cmd, match, unary)) {
        const std::string fn = lower(match[1].str());
        const std::string arg = trim(match[2].str());

        if (fn == "plot") {
            auto ys = parse_matrix(arg);
            if (!ys) {
                ys = resolve_matrix(arg);
            }
            if (!ys) {
                return std::unexpected(ys.error());
            }
            Matrix<double> empty(0, 0);
            return set_plot(empty, *ys);
        }

        if (fn == "bigint_factorial" || fn == "bigint_fib") {
            double n_d = 0.0;
            if (!parse_number(arg, n_d) || n_d < 0.0 || std::floor(n_d) != n_d) {
                return std::unexpected(
                    DomainError{fn, "expected non-negative integer argument"});
            }
            auto value = eval_bigint_unary(fn, static_cast<int>(n_d));
            if (!value) {
                return std::unexpected(value.error());
            }
            return std::to_string(*value) + "\n";
        }

        if (fn == "numthy_prime_nth") {
            double n_d = 0.0;
            if (!parse_number(arg, n_d) || n_d < 1.0 || std::floor(n_d) != n_d) {
                return std::unexpected(
                    DomainError{"numthy_prime_nth", "expected integer n >= 1"});
            }
            return std::to_string(numthy::prime_nth(static_cast<uint64_t>(n_d))) + "\n";
        }

        if (fn == "numthy_primitive_root") {
            double p_d = 0.0;
            if (!parse_number(arg, p_d) || p_d < 2.0 || std::floor(p_d) != p_d) {
                return std::unexpected(
                    DomainError{"numthy_primitive_root", "expected prime p >= 2"});
            }
            const auto p = static_cast<uint64_t>(p_d);
            if (!numthy::isprime(p)) {
                return std::unexpected(
                    DomainError{"numthy_primitive_root", "expected prime p"});
            }
            return std::to_string(numthy::primitive_root(p)) + "\n";
        }

        if (fn == "combo_factorial" || fn == "combo_catalan" || fn == "combo_bell" ||
            fn == "combo_motzkin" || fn == "combo_subfactorial" ||
            fn == "combo_double_factorial" || fn == "numthy_partition" ||
            fn == "numthy_num_divisors" || fn == "numthy_factor_count" ||
            fn == "numthy_sum_divisors" ||
            fn == "numthy_isprime" || fn == "numthy_euler_phi" || fn == "numthy_mobius" ||
            fn == "numthy_nextprime" || fn == "numthy_prevprime" ||
            fn == "numthy_liouville" || fn == "numthy_prime_pi") {
            double n_d = 0.0;
            if (!parse_number(arg, n_d) || n_d < 0.0 || std::floor(n_d) != n_d) {
                return std::unexpected(
                    DomainError{fn, "expected non-negative integer argument"});
            }
            if (fn == "combo_factorial") {
                return std::to_string(
                           combo::factorial(static_cast<uint32_t>(n_d))) +
                       "\n";
            }
            if (fn == "combo_catalan") {
                return std::to_string(
                           combo::catalan_num(static_cast<uint32_t>(n_d))) +
                       "\n";
            }
            if (fn == "combo_bell") {
                return std::to_string(
                           combo::bell_num(static_cast<uint32_t>(n_d))) +
                       "\n";
            }
            if (fn == "combo_motzkin") {
                return std::to_string(
                           combo::motzkin_num(static_cast<uint32_t>(n_d))) +
                       "\n";
            }
            if (fn == "combo_subfactorial") {
                return std::to_string(
                           combo::subfactorial(static_cast<uint32_t>(n_d))) +
                       "\n";
            }
            if (fn == "combo_double_factorial") {
                return std::to_string(
                           combo::double_factorial(static_cast<uint32_t>(n_d))) +
                       "\n";
            }
            if (fn == "combo_double_factorial") {
                return std::to_string(
                           combo::double_factorial(static_cast<uint32_t>(n_d))) +
                       "\n";
            }
            if (fn == "numthy_isprime") {
                return std::to_string(numthy::isprime(static_cast<uint64_t>(n_d)) ? 1 : 0) + "\n";
            }
            if (fn == "numthy_euler_phi") {
                return std::to_string(numthy::euler_phi(static_cast<uint64_t>(n_d))) + "\n";
            }
            if (fn == "numthy_mobius") {
                return std::to_string(
                           static_cast<double>(numthy::mobius(static_cast<uint64_t>(n_d)))) +
                       "\n";
            }
            if (fn == "numthy_nextprime") {
                return std::to_string(
                           numthy::nextprime(static_cast<uint64_t>(n_d))) +
                       "\n";
            }
            if (fn == "numthy_prevprime") {
                return std::to_string(
                           numthy::prevprime(static_cast<uint64_t>(n_d))) +
                       "\n";
            }
            if (fn == "numthy_prevprime") {
                return std::to_string(
                           numthy::prevprime(static_cast<uint64_t>(n_d))) +
                       "\n";
            }
            if (fn == "numthy_liouville") {
                return std::to_string(
                           static_cast<double>(numthy::liouville(static_cast<uint64_t>(n_d)))) +
                       "\n";
            }
            if (fn == "numthy_prime_pi") {
                return std::to_string(numthy::prime_pi(static_cast<uint64_t>(n_d))) + "\n";
            }
            if (fn == "numthy_num_divisors") {
                return std::to_string(
                           numthy::num_divisors(static_cast<uint64_t>(n_d))) +
                       "\n";
            }
            if (fn == "numthy_factor_count") {
                return std::to_string(
                           numthy::factor(static_cast<uint64_t>(n_d)).size()) +
                       "\n";
            }
            if (fn == "numthy_sum_divisors") {
                return std::to_string(
                           numthy::sum_divisors(static_cast<uint64_t>(n_d))) +
                       "\n";
            }
            return std::to_string(numthy::partition(static_cast<uint32_t>(n_d))) + "\n";
        }

        if (fn == "info_channel_capacity_bsc") {
            double p_error = 0.0;
            if (!parse_number(arg, p_error)) {
                return std::unexpected(
                    DomainError{"info_channel_capacity_bsc", "expected numeric p_error"});
            }
            return std::to_string(info::channel_capacity_bsc(p_error)) + "\n";
        }

        if (fn == "info_channel_capacity_bec") {
            double epsilon = 0.0;
            if (!parse_number(arg, epsilon)) {
                return std::unexpected(
                    DomainError{"info_channel_capacity_bec", "expected numeric epsilon"});
            }
            return std::to_string(info::channel_capacity_bec(epsilon)) + "\n";
        }

        if (fn == "info_differential_entropy_gaussian") {
            double sigma = 0.0;
            if (!parse_number(arg, sigma)) {
                return std::unexpected(
                    DomainError{"info_differential_entropy_gaussian", "expected numeric sigma"});
            }
            return std::to_string(info::differential_entropy_gaussian(sigma)) + "\n";
        }

        if (fn == "clausen") {
            double theta = 0.0;
            if (!parse_number(arg, theta)) {
                return std::unexpected(DomainError{"clausen", "expected numeric theta"});
            }
            return std::to_string(clausen(theta)) + "\n";
        }

        if (fn == "sym_simplify") {
            auto value = eval_sym_simplify_string(arg);
            if (!value) {
                return std::unexpected(value.error());
            }
            return *value;
        }

        if (fn == "eta_dirichlet") {
            double s = 0.0;
            if (!parse_number(arg, s)) {
                return std::unexpected(DomainError{"eta_dirichlet", "expected numeric s"});
            }
            return std::to_string(eta_dirichlet(s)) + "\n";
        }

        if (fn == "erf" || fn == "erfc" || fn == "gamma" || fn == "bessel_j0" || fn == "fresnel_c" ||
            fn == "fresnel_s" || fn == "ellip_k" || fn == "zeta") {
            double value = 0.0;
            if (!parse_number(arg, value)) {
                return std::unexpected(DomainError{"special", "expected numeric argument"});
            }
            std::ostringstream out;
            if (fn == "erf") {
                out << erf(value);
            } else if (fn == "erfc") {
                out << erfc(value);
            } else if (fn == "gamma") {
                out << gamma_func(value);
            } else if (fn == "bessel_j0") {
                out << bessel_j0(value);
            } else if (fn == "fresnel_c") {
                out << fresnel_c(value);
            } else if (fn == "fresnel_s") {
                out << fresnel_s(value);
            } else if (fn == "ellip_k") {
                out << ellip_k(value);
            } else {
                out << zeta(value);
            }
            out << "\n";
            return out.str();
        }

        if (fn == "gria") {
            auto data = parse_matrix(arg);
            if (!data) {
                data = resolve_matrix(arg);
            }
            if (!data) {
                return std::unexpected(data.error());
            }
            std::vector<double> flat;
            for (size_t i = 0; i < data->rows(); ++i) {
                for (size_t j = 0; j < data->cols(); ++j) {
                    flat.push_back((*data)(i, j));
                }
            }
            const double alpha = gria::compute_alpha<double>(flat, [](std::span<const double> input) {
                std::vector<double> out;
                out.reserve(input.size());
                for (double v : input) {
                    out.push_back(v * v);
                }
                return out;
            });
            std::ostringstream out;
            out << "alpha=" << alpha << " class="
                << (gria::classify(alpha) == gria::ComputeClass::Critical ? "critical"
                    : gria::classify(alpha) == gria::ComputeClass::Irreversible ? "irreversible"
                                                                                  : "reversible")
                << "\n";
            return out.str();
        }

        if (fn == "imshow") {
            auto data = parse_matrix(arg);
            if (!data) {
                data = resolve_matrix(arg);
            }
            if (!data) {
                return std::unexpected(data.error());
            }
            return set_plot_heatmap(*data);
        }

        if (fn == "spy") {
            auto data = parse_matrix(arg);
            if (!data) {
                data = resolve_matrix(arg);
            }
            if (!data) {
                return std::unexpected(data.error());
            }
            return set_plot_spy(*data);
        }

        if (fn == "hist") {
            auto data = parse_matrix(arg);
            if (!data) {
                data = resolve_matrix(arg);
            }
            if (!data) {
                return std::unexpected(data.error());
            }
            std::vector<double> values;
            values.reserve(data->rows() * data->cols());
            for (size_t i = 0; i < data->rows(); ++i) {
                for (size_t j = 0; j < data->cols(); ++j) {
                    values.push_back((*data)(i, j));
                }
            }
            if (values.empty()) {
                return std::unexpected(DomainError{"hist", "empty data"});
            }
            const auto [min_it, max_it] = std::minmax_element(values.begin(), values.end());
            const double vmin = *min_it;
            const double vmax = *max_it;
            constexpr size_t bins = 10;
            std::vector<double> counts(bins, 0.0);
            const double width = (vmax - vmin) > 0.0 ? (vmax - vmin) / static_cast<double>(bins) : 1.0;
            for (double value : values) {
                size_t idx = static_cast<size_t>((value - vmin) / width);
                if (idx >= bins) {
                    idx = bins - 1;
                }
                counts[idx] += 1.0;
            }
            std::vector<double> centers(bins);
            for (size_t i = 0; i < bins; ++i) {
                centers[i] = vmin + (static_cast<double>(i) + 0.5) * width;
            }
            return set_plot_bars(std::move(centers), std::move(counts));
        }

        if (fn == "fft") {
            auto vec = parse_matrix(arg);
            if (!vec || vec->cols() != 1) {
                if (vec && vec->rows() == 1) {
                    Matrix<double> col(vec->cols(), 1);
                    for (size_t i = 0; i < vec->cols(); ++i) {
                        col(i, 0) = (*vec)(0, i);
                    }
                    vec = col;
                } else if (!vec) {
                    return std::unexpected(vec.error());
                } else {
                    return std::unexpected(DomainError{"fft", "expected vector [a,b,c,...]"});
                }
            }
            std::vector<double> data(vec->rows());
            for (size_t i = 0; i < vec->rows(); ++i) {
                data[i] = (*vec)(i, 0);
            }
            auto spectrum = fft(data);
            if (!spectrum) {
                return std::unexpected(spectrum.error());
            }
            std::ostringstream out;
            out << "fft magnitudes:\n";
            for (size_t i = 0; i < spectrum->size(); ++i) {
                out << "  [" << i << "] " << std::abs((*spectrum)[i]) << "\n";
            }
            return out.str();
        }

        auto matrix = parse_matrix(arg);
        if (matrix) {
            state_.matrices["_"] = *matrix;
        } else {
            matrix = resolve_matrix(arg);
            if (!matrix) {
                return std::unexpected(matrix.error());
            }
        }

        std::ostringstream out;
        if (fn == "det") {
            auto result = det(*matrix);
            if (!result) {
                return std::unexpected(result.error());
            }
            out << *result << "\n";
        } else if (fn == "trace") {
            auto result = trace(*matrix);
            if (!result) {
                return std::unexpected(result.error());
            }
            out << *result << "\n";
        } else if (fn == "norm") {
            auto result = norm(*matrix);
            if (!result) {
                return std::unexpected(result.error());
            }
            out << *result << "\n";
        } else if (fn == "rank") {
            auto result = rank(*matrix);
            if (!result) {
                return std::unexpected(result.error());
            }
            out << *result << "\n";
        } else if (fn == "cond") {
            auto result = cond(*matrix);
            if (!result) {
                return std::unexpected(result.error());
            }
            out << *result << "\n";
        } else if (fn == "geo_convex_hull_area") {
            auto points = matrix_to_points2d(*matrix, "geo_convex_hull_area");
            if (!points) {
                return std::unexpected(points.error());
            }
            if (points->size() < 3) {
                return std::unexpected(
                    DomainError{"geo_convex_hull_area", "expected at least 3 points"});
            }
            const auto hull = geo::convex_hull_2d(*points);
            if (hull.size() < 3) {
                return std::unexpected(
                    DomainError{"geo_convex_hull_area", "degenerate convex hull"});
            }
            out << geo::area(hull) << "\n";
        } else if (fn == "geo_polygon_area") {
            auto value = eval_geo_polygon_area(*matrix);
            if (!value) {
                return std::unexpected(value.error());
            }
            out << *value << "\n";
        } else if (fn == "geo_polygon_perimeter") {
            auto value = eval_geo_polygon_perimeter(*matrix);
            if (!value) {
                return std::unexpected(value.error());
            }
            out << *value << "\n";
        } else if (fn == "geo_signed_area") {
            auto value = eval_geo_signed_area(*matrix);
            if (!value) {
                return std::unexpected(value.error());
            }
            out << *value << "\n";
        } else if (fn == "geo_moment_of_inertia") {
            auto value = eval_geo_moment_of_inertia(*matrix);
            if (!value) {
                return std::unexpected(value.error());
            }
            out << *value << "\n";
        } else if (fn == "geo_centroid_x") {
            auto value = eval_geo_centroid_x(*matrix);
            if (!value) {
                return std::unexpected(value.error());
            }
            out << *value << "\n";
        } else if (fn == "geo_centroid_y") {
            auto value = eval_geo_centroid_y(*matrix);
            if (!value) {
                return std::unexpected(value.error());
            }
            out << *value << "\n";
        } else if (fn == "bwt_primary_index") {
            auto value = eval_bwt_primary_index(*matrix);
            if (!value) {
                return std::unexpected(value.error());
            }
            out << *value << "\n";
        } else if (fn == "combo_rank_permutation") {
            auto value = eval_combo_rank_permutation(*matrix);
            if (!value) {
                return std::unexpected(value.error());
            }
            out << *value << "\n";
        } else if (fn == "ml_vec_norm") {
            auto value = eval_ml_vec_norm(*matrix);
            if (!value) {
                return std::unexpected(value.error());
            }
            out << *value << "\n";
        } else if (fn == "info_entropy") {
            auto value = eval_info_entropy(*matrix);
            if (!value) {
                return std::unexpected(value.error());
            }
            out << *value << "\n";
        } else if (fn == "info_lz_complexity") {
            auto value = eval_info_lz_complexity(*matrix);
            if (!value) {
                return std::unexpected(value.error());
            }
            out << *value << "\n";
        } else if (fn == "info_redundancy") {
            auto value = eval_info_redundancy(*matrix);
            if (!value) {
                return std::unexpected(value.error());
            }
            out << *value << "\n";
        } else if (fn == "info_efficiency") {
            auto value = eval_info_efficiency(*matrix);
            if (!value) {
                return std::unexpected(value.error());
            }
            out << *value << "\n";
        } else if (fn == "info_source_coding_rate") {
            auto value = eval_info_source_coding_rate(*matrix);
            if (!value) {
                return std::unexpected(value.error());
            }
            out << *value << "\n";
        } else if (fn == "info_mutual_info") {
            auto value = eval_info_mutual_info(*matrix);
            if (!value) {
                return std::unexpected(value.error());
            }
            out << *value << "\n";
        } else if (fn == "finance_irr") {
            auto value = eval_finance_irr(*matrix);
            if (!value) {
                return std::unexpected(value.error());
            }
            out << *value << "\n";
        } else if (fn == "finance_sharpe") {
            auto value = eval_finance_sharpe(*matrix);
            if (!value) {
                return std::unexpected(value.error());
            }
            out << *value << "\n";
        } else if (fn == "finance_sortino") {
            auto value = eval_finance_sortino(*matrix);
            if (!value) {
                return std::unexpected(value.error());
            }
            out << *value << "\n";
        } else if (fn == "finance_var") {
            auto value = eval_finance_var(*matrix);
            if (!value) {
                return std::unexpected(value.error());
            }
            out << *value << "\n";
        } else if (fn == "finance_cvar") {
            auto value = eval_finance_cvar(*matrix);
            if (!value) {
                return std::unexpected(value.error());
            }
            out << *value << "\n";
        } else if (fn == "finance_max_drawdown") {
            auto value = eval_finance_max_drawdown(*matrix);
            if (!value) {
                return std::unexpected(value.error());
            }
            out << *value << "\n";
        } else if (fn == "quantum_von_neumann_entropy") {
            auto value = eval_quantum_von_neumann_entropy(*matrix);
            if (!value) {
                return std::unexpected(value.error());
            }
            out << *value << "\n";
        } else if (fn == "quantum_concurrence") {
            auto value = eval_quantum_concurrence(*matrix);
            if (!value) {
                return std::unexpected(value.error());
            }
            out << *value << "\n";
        } else if (fn == "tensorops_norm") {
            auto value = eval_tensorops_norm(*matrix);
            if (!value) {
                return std::unexpected(value.error());
            }
            out << *value << "\n";
        } else if (fn == "graph_diameter") {
            auto value = eval_graph_diameter(*matrix);
            if (!value) {
                return std::unexpected(value.error());
            }
            out << *value << "\n";
        } else if (fn == "graph_radius") {
            auto value = eval_graph_radius(*matrix);
            if (!value) {
                return std::unexpected(value.error());
            }
            out << *value << "\n";
        } else if (fn == "graph_is_bipartite") {
            auto value = eval_graph_is_bipartite(*matrix);
            if (!value) {
                return std::unexpected(value.error());
            }
            out << *value << "\n";
        } else if (fn == "graph_is_dag") {
            auto value = eval_graph_is_dag(*matrix);
            if (!value) {
                return std::unexpected(value.error());
            }
            out << *value << "\n";
        } else if (fn == "stats_mean") {
            auto value = eval_stats_mean(*matrix);
            if (!value) {
                return std::unexpected(value.error());
            }
            out << *value << "\n";
        } else if (fn == "stats_median") {
            auto value = eval_stats_median(*matrix);
            if (!value) {
                return std::unexpected(value.error());
            }
            out << *value << "\n";
        } else if (fn == "stats_stddev") {
            auto value = eval_stats_stddev(*matrix);
            if (!value) {
                return std::unexpected(value.error());
            }
            out << *value << "\n";
        } else if (fn == "stats_skewness") {
            auto value = eval_stats_skewness(*matrix);
            if (!value) {
                return std::unexpected(value.error());
            }
            out << *value << "\n";
        } else if (fn == "stats_kurtosis") {
            auto value = eval_stats_kurtosis(*matrix);
            if (!value) {
                return std::unexpected(value.error());
            }
            out << *value << "\n";
        } else if (fn == "stats_var") {
            auto value = eval_stats_var(*matrix);
            if (!value) {
                return std::unexpected(value.error());
            }
            out << *value << "\n";
        } else if (fn == "stats_mode") {
            auto value = eval_stats_mode(*matrix);
            if (!value) {
                return std::unexpected(value.error());
            }
            out << *value << "\n";
        } else if (fn == "stats_geometric_mean") {
            auto value = eval_stats_geometric_mean(*matrix);
            if (!value) {
                return std::unexpected(value.error());
            }
            out << *value << "\n";
        } else if (fn == "stats_harmonic_mean") {
            auto value = eval_stats_harmonic_mean(*matrix);
            if (!value) {
                return std::unexpected(value.error());
            }
            out << *value << "\n";
        } else if (fn == "stats_rms") {
            auto value = eval_stats_rms(*matrix);
            if (!value) {
                return std::unexpected(value.error());
            }
            out << *value << "\n";
        } else if (fn == "stats_mad") {
            auto value = eval_stats_mad(*matrix);
            if (!value) {
                return std::unexpected(value.error());
            }
            out << *value << "\n";
        } else if (fn == "stats_iqr") {
            auto value = eval_stats_iqr(*matrix);
            if (!value) {
                return std::unexpected(value.error());
            }
            out << *value << "\n";
        } else if (fn == "stats_min_value") {
            auto value = eval_stats_min_value(*matrix);
            if (!value) {
                return std::unexpected(value.error());
            }
            out << *value << "\n";
        } else if (fn == "count_components") {
            auto value = eval_count_components(*matrix);
            if (!value) {
                return std::unexpected(value.error());
            }
            out << *value << "\n";
        } else if (fn == "graph_is_connected") {
            auto value = eval_graph_is_connected(*matrix);
            if (!value) {
                return std::unexpected(value.error());
            }
            out << *value << "\n";
        } else if (fn == "graph_is_tree") {
            auto value = eval_graph_is_tree(*matrix);
            if (!value) {
                return std::unexpected(value.error());
            }
            out << *value << "\n";
        } else if (fn == "graph_pagerank") {
            auto G = graph_from_adjacency(*matrix, "graph_pagerank");
            if (!G) {
                return std::unexpected(G.error());
            }
            const auto scores = graph::pagerank(*G);
            out << "pagerank:\n";
            for (size_t i = 0; i < scores.size(); ++i) {
                out << "  [" << i << "] " << scores[i] << "\n";
            }
        } else if (fn == "graph_betweenness") {
            auto bc = eval_graph_betweenness(*matrix);
            if (!bc) {
                return std::unexpected(bc.error());
            }
            out << "betweenness:\n";
            for (size_t i = 0; i < bc->rows(); ++i) {
                out << "  [" << i << "] " << (*bc)(i, 0) << "\n";
            }
        } else if (fn == "graph_closeness") {
            auto cc = eval_graph_closeness(*matrix);
            if (!cc) {
                return std::unexpected(cc.error());
            }
            out << "closeness:\n";
            for (size_t i = 0; i < cc->rows(); ++i) {
                out << "  [" << i << "] " << (*cc)(i, 0) << "\n";
            }
        } else if (fn == "graph_degree_centrality") {
            auto dc = eval_graph_degree_centrality(*matrix);
            if (!dc) {
                return std::unexpected(dc.error());
            }
            out << "degree_centrality:\n";
            for (size_t i = 0; i < dc->rows(); ++i) {
                out << "  [" << i << "] " << (*dc)(i, 0) << "\n";
            }
        } else if (fn == "fft_rfft") {
            auto spectrum = eval_fft_rfft(*matrix);
            if (!spectrum) {
                return std::unexpected(spectrum.error());
            }
            out << "rfft =\n";
            print_matrix(out, *spectrum);
        } else if (fn == "fft_dft") {
            auto spectrum = eval_fft_dft(*matrix);
            if (!spectrum) {
                return std::unexpected(spectrum.error());
            }
            out << "dft =\n";
            print_matrix(out, *spectrum);
        } else if (fn == "fft_ifft") {
            auto signal = eval_fft_ifft(*matrix);
            if (!signal) {
                return std::unexpected(signal.error());
            }
            out << "ifft =\n";
            print_matrix(out, *signal);
        } else if (fn == "fft_fft2") {
            auto spectrum = eval_fft_fft2(*matrix);
            if (!spectrum) {
                return std::unexpected(spectrum.error());
            }
            out << "fft2 =\n";
            print_matrix(out, *spectrum);
        } else if (fn == "graph_topological_sort") {
            auto order = eval_graph_topological_sort(*matrix);
            if (!order) {
                return std::unexpected(order.error());
            }
            out << "order =\n";
            print_matrix(out, *order);
        } else if (fn == "graph_greedy_colour") {
            auto colors = eval_graph_greedy_colour(*matrix);
            if (!colors) {
                return std::unexpected(colors.error());
            }
            out << "colors =\n";
            print_matrix(out, *colors);
        } else if (fn == "graph_euler_circuit") {
            auto circuit = eval_graph_euler_circuit(*matrix);
            if (!circuit) {
                return std::unexpected(circuit.error());
            }
            out << "circuit =\n";
            print_matrix(out, *circuit);
        } else if (fn == "graph_scc") {
            auto scc = eval_graph_scc(*matrix);
            if (!scc) {
                return std::unexpected(scc.error());
            }
            out << "scc =\n";
            print_matrix(out, *scc);
        } else if (fn == "geo_delaunay_2d") {
            auto tris = eval_geo_delaunay_2d(*matrix);
            if (!tris) {
                return std::unexpected(tris.error());
            }
            out << "triangles =\n";
            print_matrix(out, *tris);
        } else if (fn == "geo_voronoi") {
            auto verts = eval_geo_voronoi(*matrix);
            if (!verts) {
                return std::unexpected(verts.error());
            }
            out << "voronoi =\n";
            print_matrix(out, *verts);
        } else if (fn == "topo_pairwise_distances") {
            auto dist = eval_topo_pairwise_distances(*matrix);
            if (!dist) {
                return std::unexpected(dist.error());
            }
            out << "dist =\n";
            print_matrix(out, *dist);
        } else if (fn == "combo_next_perm") {
            auto perm = eval_combo_next_perm(*matrix);
            if (!perm) {
                return std::unexpected(perm.error());
            }
            out << "perm =\n";
            print_matrix(out, *perm);
        } else if (fn == "numthy_convergents") {
            auto conv = eval_numthy_convergents(*matrix);
            if (!conv) {
                return std::unexpected(conv.error());
            }
            out << "convergents =\n";
            print_matrix(out, *conv);
        } else if (fn == "ml_mat_transpose") {
            auto At = eval_ml_mat_transpose(*matrix);
            if (!At) {
                return std::unexpected(At.error());
            }
            out << "At =\n";
            print_matrix(out, *At);
        } else if (fn == "fftshift") {
            auto shifted = eval_fftshift(*matrix);
            if (!shifted) {
                return std::unexpected(shifted.error());
            }
            out << "shifted =\n";
            print_matrix(out, *shifted);
        } else if (fn == "prewitt") {
            auto edge = eval_prewitt(*matrix);
            if (!edge) {
                return std::unexpected(edge.error());
            }
            out << "edge =\n";
            print_matrix(out, *edge);
        } else if (fn == "poly_deriv") {
            auto deriv = eval_poly_deriv(*matrix);
            if (!deriv) {
                return std::unexpected(deriv.error());
            }
            out << "deriv =\n";
            print_matrix(out, *deriv);
        } else if (fn == "graph_floyd_warshall") {
            auto dist = eval_graph_floyd_warshall(*matrix);
            if (!dist) {
                return std::unexpected(dist.error());
            }
            out << "dist =\n";
            print_matrix(out, *dist);
        } else if (fn == "graph_mst_kruskal") {
            auto edges = eval_graph_mst_kruskal(*matrix);
            if (!edges) {
                return std::unexpected(edges.error());
            }
            out << "mst =\n";
            print_matrix(out, *edges);
        } else if (fn == "graph_mst_prim") {
            auto edges = eval_graph_mst_prim(*matrix);
            if (!edges) {
                return std::unexpected(edges.error());
            }
            out << "mst =\n";
            print_matrix(out, *edges);
        } else if (fn == "fft_dct2") {
            auto coeffs = eval_fft_dct2(*matrix);
            if (!coeffs) {
                return std::unexpected(coeffs.error());
            }
            out << "dct =\n";
            print_matrix(out, *coeffs);
        } else if (fn == "fft_idct2") {
            auto signal = eval_fft_idct2(*matrix);
            if (!signal) {
                return std::unexpected(signal.error());
            }
            out << "idct =\n";
            print_matrix(out, *signal);
        } else if (fn == "fft_dst2") {
            auto coeffs = eval_fft_dst2(*matrix);
            if (!coeffs) {
                return std::unexpected(coeffs.error());
            }
            out << "dst =\n";
            print_matrix(out, *coeffs);
        } else if (fn == "quantum_hadamard") {
            auto psi = matrix_to_ket2(*matrix, "quantum_hadamard");
            if (!psi) {
                return std::unexpected(psi.error());
            }
            const auto applied = quantum::op_apply(quantum::hadamard(), *psi);
            out << "state =\n";
            out << "  [" << applied[0].real() << "]\n";
            out << "  [" << applied[1].real() << "]\n";
        } else if (fn == "quantum_ket_normalise") {
            auto normalized = eval_quantum_ket_normalise_matrix(*matrix);
            if (!normalized) {
                return std::unexpected(normalized.error());
            }
            out << "state =\n";
            for (size_t i = 0; i < normalized->rows(); ++i) {
                out << "  [" << (*normalized)(i, 0) << "]\n";
            }
        } else if (fn == "quantum_density_matrix") {
            auto rho = eval_quantum_density_matrix(*matrix);
            if (!rho) {
                return std::unexpected(rho.error());
            }
            out << "rho =\n";
            print_matrix(out, *rho);
        } else if (fn == "quantum_ket_superposition") {
            auto state = eval_quantum_ket_superposition_matrix(*matrix);
            if (!state) {
                return std::unexpected(state.error());
            }
            out << "state =\n";
            for (size_t i = 0; i < state->rows(); ++i) {
                out << "  [" << (*state)(i, 0) << "]\n";
            }
        } else if (fn == "eig_sym") {
            auto eig_result = eig_sym(*matrix);
            if (!eig_result) {
                return std::unexpected(eig_result.error());
            }
            out << "eigenvalues:\n";
            for (size_t i = 0; i < eig_result->values.rows(); ++i) {
                out << "  " << eig_result->values(i, 0) << "\n";
            }
        } else if (fn == "svd") {
            auto s = svd(*matrix);
            if (!s) {
                return std::unexpected(s.error());
            }
            out << "singular values:\n";
            for (size_t i = 0; i < s->S.rows(); ++i) {
                out << "  " << s->S(i, 0) << "\n";
            }
        } else if (fn == "lu") {
            auto result = lu(*matrix);
            if (!result) {
                return std::unexpected(result.error());
            }
            const auto& [L, U, P] = *result;
            out << "L =\n";
            print_matrix(out, L);
            out << "U =\n";
            print_matrix(out, U);
            out << "P =\n";
            print_matrix(out, P);
        } else if (fn == "qr") {
            auto result = qr(*matrix);
            if (!result) {
                return std::unexpected(result.error());
            }
            const auto& [Q, R] = *result;
            out << "Q =\n";
            print_matrix(out, Q);
            out << "R =\n";
            print_matrix(out, R);
        } else if (fn == "chol") {
            auto result = chol(*matrix);
            if (!result) {
                return std::unexpected(result.error());
            }
            out << "L =\n";
            print_matrix(out, *result);
        } else {
            return std::unexpected(DomainError{"repl", "unknown function: " + fn});
        }
        return out.str();
    }

    return std::unexpected(DomainError{"repl", "could not parse: " + cmd});
}

} // namespace ms::interp
