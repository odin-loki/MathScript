#include "ms/distributed/dist_matrix.hpp"
#include <functional>
#include "ms/distributed/iterative.hpp"
#include "ms/distributed/matmul.hpp"
#include "ms/distributed/mpi_context.hpp"
#include "ms/distributed/solve.hpp"
#include "ms/frameworks/axiom/axiom.hpp"
#include "ms/frameworks/cellai/cellai.hpp"
#include "ms/frameworks/cypha/cypha.hpp"
#include "ms/frameworks/gria/gria.hpp"
#include "ms/frameworks/izaac/izaac.hpp"
#include "ms/interp/repl_engine.hpp"
#include "ms/version.hpp"
#include "ms/interp/plot_console.hpp"
#include "ms/core/operations.hpp"
#include "ms/core/sparse.hpp"
#include "ms/fft/fft.hpp"
#include "ms/linalg/linalg.hpp"
#include "ms/cuda/nccl.hpp"
#include "ms/cuda/nvml.hpp"
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
#include "ms/optim/optim.hpp"
#include "ms/crypto/crypto.hpp"
#include "ms/fem/fem.hpp"
#include "ms/cfd/cfd.hpp"
#include "ms/cuda/elementwise.hpp"
#include "ms/cuda/solver.hpp"
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <complex>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <memory>
#include <optional>
#include <regex>
#include <span>
#include <span>
#include <sstream>
#include <string_view>
#include <type_traits>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ms::interp::detail {


thread_local std::atomic<bool>* g_repl_cancel_flag = nullptr;

ColMatrix<double> matrix_to_col_matrix(const Matrix<double>& matrix) {
    ColMatrix<double> out(matrix.rows(), matrix.cols());
    for (size_t i = 0; i < matrix.rows(); ++i) {
        for (size_t j = 0; j < matrix.cols(); ++j) {
            out(i, j) = matrix(i, j);
        }
    }
    return out;
}

ms::distributed::MPIContext& repl_mpi_context() {
    static ms::distributed::MPIContext ctx = ms::distributed::init(0, nullptr);
    return ctx;
}

Result<Matrix<double>> eval_dist_solve(const Matrix<double>& A, const Matrix<double>& b) {
    auto& ctx = repl_mpi_context();
    auto dA = ms::distributed::scatter(A, ctx);
    if (!dA) {
        return std::unexpected(dA.error());
    }
    auto db = ms::distributed::scatter(b, ctx);
    if (!db) {
        return std::unexpected(db.error());
    }
    return ms::distributed::solve(*dA, *db, ctx);
}

Result<Matrix<double>> eval_dist_cg(const Matrix<double>& A, const Matrix<double>& b) {
    auto& ctx = repl_mpi_context();
    auto dA = ms::distributed::scatter(A, ctx);
    if (!dA) {
        return std::unexpected(dA.error());
    }
    auto db = ms::distributed::scatter(b, ctx);
    if (!db) {
        return std::unexpected(db.error());
    }
    return ms::distributed::dist_cg(*dA, *db, ctx);
}

Result<Matrix<double>> eval_dist_gmres(const Matrix<double>& A, const Matrix<double>& b) {
    auto& ctx = repl_mpi_context();
    auto dA = ms::distributed::scatter(A, ctx);
    if (!dA) {
        return std::unexpected(dA.error());
    }
    auto db = ms::distributed::scatter(b, ctx);
    if (!db) {
        return std::unexpected(db.error());
    }
    return ms::distributed::dist_gmres(*dA, *db, ctx);
}

Result<Matrix<double>> eval_dist_jacobi(const Matrix<double>& A, const Matrix<double>& b) {
    auto& ctx = repl_mpi_context();
    auto dA = ms::distributed::scatter(A, ctx);
    if (!dA) {
        return std::unexpected(dA.error());
    }
    auto db = ms::distributed::scatter(b, ctx);
    if (!db) {
        return std::unexpected(db.error());
    }
    return ms::distributed::dist_jacobi(*dA, *db, ctx);
}

Result<Matrix<double>> eval_dist_bicgstab(const Matrix<double>& A, const Matrix<double>& b) {
    auto& ctx = repl_mpi_context();
    auto dA = ms::distributed::scatter(A, ctx);
    if (!dA) {
        return std::unexpected(dA.error());
    }
    auto db = ms::distributed::scatter(b, ctx);
    if (!db) {
        return std::unexpected(db.error());
    }
    return ms::distributed::dist_bicgstab(*dA, *db, ctx);
}

Result<Matrix<double>> eval_dist_minres(const Matrix<double>& A, const Matrix<double>& b) {
    auto& ctx = repl_mpi_context();
    auto dA = ms::distributed::scatter(A, ctx);
    if (!dA) {
        return std::unexpected(dA.error());
    }
    auto db = ms::distributed::scatter(b, ctx);
    if (!db) {
        return std::unexpected(db.error());
    }
    return ms::distributed::dist_minres(*dA, *db, ctx);
}

Result<Matrix<double>> eval_dist_qmr(const Matrix<double>& A, const Matrix<double>& b) {
    auto& ctx = repl_mpi_context();
    auto dA = ms::distributed::scatter(A, ctx);
    if (!dA) {
        return std::unexpected(dA.error());
    }
    auto db = ms::distributed::scatter(b, ctx);
    if (!db) {
        return std::unexpected(db.error());
    }
    return ms::distributed::dist_qmr(*dA, *db, ctx);
}

Result<Matrix<double>> eval_dist_tfqmr(const Matrix<double>& A, const Matrix<double>& b) {
    auto& ctx = repl_mpi_context();
    auto dA = ms::distributed::scatter(A, ctx);
    if (!dA) {
        return std::unexpected(dA.error());
    }
    auto db = ms::distributed::scatter(b, ctx);
    if (!db) {
        return std::unexpected(db.error());
    }
    return ms::distributed::dist_tfqmr(*dA, *db, ctx);
}

Result<Matrix<double>> eval_dist_lsmr(const Matrix<double>& A, const Matrix<double>& b) {
    auto& ctx = repl_mpi_context();
    auto dA = ms::distributed::scatter(A, ctx);
    if (!dA) {
        return std::unexpected(dA.error());
    }
    auto db = ms::distributed::scatter(b, ctx);
    if (!db) {
        return std::unexpected(db.error());
    }
    return ms::distributed::dist_lsmr(*dA, *db, ctx);
}

Result<Matrix<double>> eval_dist_lsqr(const Matrix<double>& A, const Matrix<double>& b) {
    auto& ctx = repl_mpi_context();
    auto dA = ms::distributed::scatter(A, ctx);
    if (!dA) {
        return std::unexpected(dA.error());
    }
    auto db = ms::distributed::scatter(b, ctx);
    if (!db) {
        return std::unexpected(db.error());
    }
    return ms::distributed::dist_lsqr(*dA, *db, ctx);
}

Result<Matrix<double>> eval_dist_matmul(const Matrix<double>& A, const Matrix<double>& B) {
    auto& ctx = repl_mpi_context();
    auto dA = ms::distributed::scatter(A, ctx);
    if (!dA) {
        return std::unexpected(dA.error());
    }
    auto dB = ms::distributed::scatter(B, ctx);
    if (!dB) {
        return std::unexpected(dB.error());
    }
    return ms::distributed::matmul(*dA, *dB, ctx);
}

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

bool parse_uint64(const std::string& text, uint64_t& value) {
    if (text.empty()) {
        return false;
    }
    char* end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(text.c_str(), &end, 0);
    if (end != text.c_str() + text.size() || errno == ERANGE) {
        return false;
    }
    value = static_cast<uint64_t>(parsed);
    return true;
}

std::string_view trim_view(std::string_view s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) {
        s.remove_prefix(1);
    }
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) {
        s.remove_suffix(1);
    }
    return s;
}

bool iequals(std::string_view a, std::string_view b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::tolower(static_cast<unsigned char>(a[i])) !=
            std::tolower(static_cast<unsigned char>(b[i]))) {
            return false;
        }
    }
    return true;
}

bool parse_number_view(std::string_view text, double& value) {
    text = trim_view(text);
    if (text.empty()) {
        return false;
    }
    std::string scratch(text);
    return parse_number(scratch, value);
}

bool is_literal_arith_char(char c) {
    return std::isspace(static_cast<unsigned char>(c)) ||
           std::isdigit(static_cast<unsigned char>(c)) || c == '.' || c == '+' || c == '-' ||
           c == '*' || c == '/' || c == '(' || c == ')';
}

bool is_literal_arith_expr(std::string_view expr) {
    expr = trim_view(expr);
    if (expr.empty()) {
        return false;
    }
    for (char c : expr) {
        if (!is_literal_arith_char(c)) {
            return false;
        }
    }
    return true;
}

std::string_view strip_outer_parens_view(std::string_view expr) {
    while (true) {
        expr = trim_view(expr);
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

bool is_binary_minus_view(std::string_view expr, size_t index) {
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

std::optional<std::pair<size_t, char>> find_top_level_op_view(std::string_view expr,
                                                              const char* ops) {
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
                    if (c == '-' && !is_binary_minus_view(expr, i)) {
                        continue;
                    }
                    last = std::pair{i, *p};
                }
            }
        }
    }
    return last;
}

std::optional<std::pair<size_t, char>> find_scalar_binop_view(std::string_view rhs) {
    if (auto add_sub = find_top_level_op_view(rhs, "+-")) {
        return add_sub;
    }
    return find_top_level_op_view(rhs, "*/");
}

Result<double> eval_literal_arith(std::string_view expr_text);

Result<double> eval_literal_arith(std::string_view expr_text) {
    std::string_view expr = strip_outer_parens_view(expr_text);
    if (expr.empty()) {
        return std::unexpected(DomainError{"eval", "empty expression"});
    }

    if (expr.front() == '-') {
        auto inner = eval_literal_arith(expr.substr(1));
        if (!inner) {
            return std::unexpected(inner.error());
        }
        return -(*inner);
    }
    if (expr.front() == '+') {
        return eval_literal_arith(expr.substr(1));
    }

    double value = 0.0;
    if (parse_number_view(expr, value)) {
        return value;
    }

    const auto op_pos = find_scalar_binop_view(expr);
    if (!op_pos) {
        return std::unexpected(DomainError{"eval", "invalid scalar expression"});
    }

    auto left = eval_literal_arith(trim_view(expr.substr(0, op_pos->first)));
    if (!left) {
        return std::unexpected(left.error());
    }
    auto right = eval_literal_arith(trim_view(expr.substr(op_pos->first + 1)));
    if (!right) {
        return std::unexpected(right.error());
    }
    return Interpreter::eval_scalar_op(op_pos->second, *left, *right);
}

struct ScalarFnCache {
    static constexpr size_t kCapacity = 32;
    std::array<std::string, kCapacity> keys{};
    std::array<std::string, kCapacity> lowered{};
    size_t count = 0;

    const std::string& lookup_lowered(std::string_view name) {
        const size_t search_limit = count < kCapacity ? count : kCapacity;
        for (size_t i = 0; i < search_limit; ++i) {
            if (iequals(keys[i], name)) {
                return lowered[i];
            }
        }
        std::string key;
        key.reserve(name.size());
        key.assign(name);
        std::string value = lower(key);
        if (count < kCapacity) {
            keys[count] = std::move(key);
            lowered[count] = std::move(value);
            return lowered[count++];
        }
        const size_t slot = count % kCapacity;
        keys[slot] = std::move(key);
        lowered[slot] = std::move(value);
        ++count;
        return lowered[slot];
    }

    std::string_view lookup(std::string_view name) {
        return lookup_lowered(name);
    }
};

thread_local ScalarFnCache g_scalar_fn_cache;
thread_local std::vector<double> g_scalar_call_arg_buf;

Result<double> eval_scalar_call_cached(std::string_view fn_name, std::span<const double> args);
Result<double> eval_scalar_expr_impl(const SessionState& state, std::string_view expr_text);
Result<void> require_session_rng(const char* fn);

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

Result<std::vector<std::vector<float>>> matrix_to_filter_kernel(const Matrix<double>& k,
                                                                const char* fn) {
    if (k.rows() == 0 || k.cols() == 0) {
        return std::unexpected(DomainError{fn, "empty kernel matrix"});
    }
    std::vector<std::vector<float>> kernel(k.rows(), std::vector<float>(k.cols()));
    for (size_t r = 0; r < k.rows(); ++r) {
        for (size_t c = 0; c < k.cols(); ++c) {
            kernel[r][c] = static_cast<float>(k(r, c));
        }
    }
    return kernel;
}

Matrix<double> rgb_image_to_matrix(const image::Image& img) {
    const size_t rows = static_cast<size_t>(img.rows * img.cols);
    Matrix<double> out(rows, static_cast<size_t>(img.channels));
    size_t idx = 0;
    for (int r = 0; r < img.rows; ++r) {
        for (int c = 0; c < img.cols; ++c) {
            for (int ch = 0; ch < img.channels; ++ch) {
                out(idx, static_cast<size_t>(ch)) = img.at(r, c, ch);
            }
            ++idx;
        }
    }
    return out;
}

Result<int> parse_morph_ksize(double ksize_d, const char* fn) {
    const int ksize = static_cast<int>(ksize_d);
    if (ksize < 1 || ksize_d != ksize || (ksize % 2) == 0) {
        return std::unexpected(DomainError{fn, "expected positive odd integer ksize"});
    }
    return ksize;
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

Result<bignum::BigInt> bigint_from_scalar(double arg, const char* fn) {
    if (!std::isfinite(arg) || std::floor(arg) != arg) {
        return std::unexpected(DomainError{fn, "expected integer argument"});
    }
    return bignum::BigInt(static_cast<long long>(arg));
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

Result<Matrix<double>> eval_cellai_boltzmann_weights(const std::vector<double>& energies,
                                                      double temperature) {
    return vector_to_column(cellai::boltzmann_weights(energies, temperature));
}

Result<Matrix<double>> eval_cellai_hebbian_update(const Matrix<double>& w_m,
                                                  const Matrix<double>& x_m,
                                                  const Matrix<double>& y_m,
                                                  double learning_rate) {
    return cellai::hebbian_update(w_m, x_m, y_m, learning_rate);
}

Result<Matrix<double>> eval_cellai_cell_to_cypha_features(
    const cellai::CellMemory& memory, const std::vector<double>& time_scales) {
    return cellai::cell_to_cypha_features(memory, time_scales);
}

Matrix<double> psd_result_to_matrix(const PSDResult& psd) {
    Matrix<double> out(psd.frequencies.size(), 2);
    for (size_t i = 0; i < psd.frequencies.size(); ++i) {
        out(i, 0) = psd.frequencies[i];
        out(i, 1) = psd.power[i];
    }
    return out;
}

Matrix<double> coherence_result_to_matrix(const CoherenceResult& coh) {
    Matrix<double> out(coh.frequencies.size(), 2);
    for (size_t i = 0; i < coh.frequencies.size(); ++i) {
        out(i, 0) = coh.frequencies[i];
        out(i, 1) = coh.coherence[i];
    }
    return out;
}

Matrix<double> iir_coeffs_to_matrix(const IirCoeffs& coeffs) {
    const size_t n = std::max(coeffs.b.size(), coeffs.a.size());
    Matrix<double> out(2, n);
    for (size_t j = 0; j < n; ++j) {
        out(0, j) = j < coeffs.b.size() ? coeffs.b[j] : 0.0;
        out(1, j) = j < coeffs.a.size() ? coeffs.a[j] : 0.0;
    }
    return out;
}

Result<FilterType> parse_signal_filter_type(const std::string& text) {
    double type_d = 0.0;
    if (parse_number(trim_copy(text), type_d)) {
        if (type_d == 0.0) {
            return FilterType::Lowpass;
        }
        if (type_d == 1.0) {
            return FilterType::Highpass;
        }
        return std::unexpected(DomainError{
            "signal_cheby2", "expected type 0 (lowpass) or 1 (highpass)"});
    }
    const std::string type = lower(trim_copy(text));
    if (type == "lowpass" || type == "low") {
        return FilterType::Lowpass;
    }
    if (type == "highpass" || type == "high") {
        return FilterType::Highpass;
    }
    return std::unexpected(DomainError{
        "signal_cheby2", "expected filter type 'lowpass' or 'highpass'"});
}

Result<FirWindow> parse_fir_window(const std::string& text, const char* fn) {
    double window_d = 0.0;
    if (parse_number(trim_copy(text), window_d)) {
        if (window_d == 0.0) {
            return FirWindow::Rectangular;
        }
        if (window_d == 1.0) {
            return FirWindow::Hamming;
        }
        if (window_d == 2.0) {
            return FirWindow::Hann;
        }
        if (window_d == 3.0) {
            return FirWindow::Blackman;
        }
        return std::unexpected(DomainError{
            fn, "expected window 0..3 (rect/hamming/hann/blackman)"});
    }
    const std::string window = lower(trim_copy(text));
    if (window == "rectangular" || window == "rect") {
        return FirWindow::Rectangular;
    }
    if (window == "hamming") {
        return FirWindow::Hamming;
    }
    if (window == "hann" || window == "hanning") {
        return FirWindow::Hann;
    }
    if (window == "blackman") {
        return FirWindow::Blackman;
    }
    return std::unexpected(DomainError{
        fn, "expected window 'rectangular'|'hamming'|'hann'|'blackman'"});
}

Matrix<double> ml_model_to_matrix(const ml::Vec& coef, double intercept) {
    Matrix<double> out(coef.size() + 1, 1);
    for (size_t i = 0; i < coef.size(); ++i) {
        out(i, 0) = coef[i];
    }
    out(coef.size(), 0) = intercept;
    return out;
}

Result<std::pair<ml::Vec, double>> ml_model_from_matrix(const Matrix<double>& model,
                                                         const char* fn) {
    if (model.cols() != 1 || model.rows() < 2) {
        return std::unexpected(
            DomainError{fn, "expected (n_features+1) x 1 model column with intercept last"});
    }
    ml::Vec coef(model.rows() - 1);
    for (size_t i = 0; i < coef.size(); ++i) {
        coef[i] = model(i, 0);
    }
    return std::pair{coef, model(model.rows() - 1, 0)};
}

Result<Matrix<double>> eval_ml_linear_fit(const Matrix<double>& X_m, const Matrix<double>& y_m) {
    auto X = matrix_to_ml_mat(X_m, "ml_linear_fit");
    if (!X) {
        return std::unexpected(X.error());
    }
    auto y = matrix_to_ml_vec(y_m, "ml_linear_fit");
    if (!y) {
        return std::unexpected(y.error());
    }
    if (y->size() != X->size()) {
        return std::unexpected(DimensionMismatch{y->size(), X->size()});
    }
    ml::LinearRegression lr;
    lr.fit(*X, *y);
    return ml_model_to_matrix(lr.coef, lr.intercept);
}

Result<Matrix<double>> eval_ml_linear_predict(const Matrix<double>& X_m,
                                              const Matrix<double>& model_m) {
    auto X = matrix_to_ml_mat(X_m, "ml_linear_predict");
    if (!X) {
        return std::unexpected(X.error());
    }
    auto params = ml_model_from_matrix(model_m, "ml_linear_predict");
    if (!params) {
        return std::unexpected(params.error());
    }
    ml::LinearRegression lr;
    lr.coef = params->first;
    lr.intercept = params->second;
    return vector_to_column(lr.predict(*X));
}

Result<Matrix<double>> eval_ml_ridge_fit(const Matrix<double>& X_m, const Matrix<double>& y_m,
                                         double alpha) {
    auto X = matrix_to_ml_mat(X_m, "ml_ridge_fit");
    if (!X) {
        return std::unexpected(X.error());
    }
    auto y = matrix_to_ml_vec(y_m, "ml_ridge_fit");
    if (!y) {
        return std::unexpected(y.error());
    }
    if (y->size() != X->size()) {
        return std::unexpected(DimensionMismatch{y->size(), X->size()});
    }
    ml::RidgeRegression rr(alpha);
    rr.fit(*X, *y);
    return ml_model_to_matrix(rr.coef, rr.intercept);
}

Result<Matrix<double>> eval_ml_ridge_predict(const Matrix<double>& X_m,
                                             const Matrix<double>& model_m) {
    auto X = matrix_to_ml_mat(X_m, "ml_ridge_predict");
    if (!X) {
        return std::unexpected(X.error());
    }
    auto params = ml_model_from_matrix(model_m, "ml_ridge_predict");
    if (!params) {
        return std::unexpected(params.error());
    }
    ml::RidgeRegression rr(1.0);
    rr.coef = params->first;
    rr.intercept = params->second;
    return vector_to_column(rr.predict(*X));
}

Result<Matrix<double>> eval_ml_logistic_fit(const Matrix<double>& X_m, const Matrix<double>& y_m) {
    auto X = matrix_to_ml_mat(X_m, "ml_logistic_fit");
    if (!X) {
        return std::unexpected(X.error());
    }
    auto y = matrix_to_ml_vec(y_m, "ml_logistic_fit");
    if (!y) {
        return std::unexpected(y.error());
    }
    if (y->size() != X->size()) {
        return std::unexpected(DimensionMismatch{y->size(), X->size()});
    }
    ml::LogisticRegression lr;
    lr.fit(*X, *y);
    return ml_model_to_matrix(lr.coef, lr.intercept);
}

Result<Matrix<double>> eval_ml_logistic_predict(const Matrix<double>& X_m,
                                               const Matrix<double>& model_m) {
    auto X = matrix_to_ml_mat(X_m, "ml_logistic_predict");
    if (!X) {
        return std::unexpected(X.error());
    }
    auto params = ml_model_from_matrix(model_m, "ml_logistic_predict");
    if (!params) {
        return std::unexpected(params.error());
    }
    ml::LogisticRegression lr;
    lr.coef = params->first;
    lr.intercept = params->second;
    return vector_to_column(lr.predict(*X));
}

Matrix<double> ml_pca_model_to_matrix(const ml::PCA& pca) {
    const int nc = pca.n_components;
    const int nf = static_cast<int>(pca.mean_.size());
    Matrix<double> out(static_cast<size_t>(nc + 1), static_cast<size_t>(nf));
    for (int i = 0; i < nc; ++i) {
        for (int j = 0; j < nf; ++j) {
            out(static_cast<size_t>(i), static_cast<size_t>(j)) = pca.components[static_cast<size_t>(i)][static_cast<size_t>(j)];
        }
    }
    for (int j = 0; j < nf; ++j) {
        out(static_cast<size_t>(nc), static_cast<size_t>(j)) = pca.mean_[static_cast<size_t>(j)];
    }
    return out;
}

Result<ml::PCA> ml_pca_from_matrix(const Matrix<double>& model, const char* fn) {
    if (model.rows() < 2 || model.cols() < 1) {
        return std::unexpected(
            DomainError{fn, "expected PCA model (n_components+1) x n_features with mean row last"});
    }
    ml::PCA pca(static_cast<int>(model.rows() - 1));
    pca.n_components = static_cast<int>(model.rows() - 1);
    pca.components.resize(static_cast<size_t>(pca.n_components));
    for (int i = 0; i < pca.n_components; ++i) {
        pca.components[static_cast<size_t>(i)].resize(model.cols());
        for (size_t j = 0; j < model.cols(); ++j) {
            pca.components[static_cast<size_t>(i)][j] = model(static_cast<size_t>(i), j);
        }
    }
    pca.mean_.resize(model.cols());
    for (size_t j = 0; j < model.cols(); ++j) {
        pca.mean_[j] = model(static_cast<size_t>(pca.n_components), j);
    }
    return pca;
}

Result<ml::KMeans> ml_kmeans_from_matrix(const Matrix<double>& model, const char* fn) {
    auto centers = matrix_to_ml_mat(model, fn);
    if (!centers) {
        return std::unexpected(centers.error());
    }
    if (centers->empty()) {
        return std::unexpected(DomainError{fn, "expected non-empty KMeans centers matrix"});
    }
    ml::KMeans km(static_cast<int>(centers->size()));
    km.centers = std::move(*centers);
    return km;
}

Matrix<double> grid_to_matrix(const std::vector<std::vector<double>>& grid);
Matrix<double> int_vector_to_column(const std::vector<int>& values);

Result<Matrix<double>> eval_ml_pca_fit(const Matrix<double>& X_m, int n_components) {
    auto X = matrix_to_ml_mat(X_m, "ml_pca_fit");
    if (!X) {
        return std::unexpected(X.error());
    }
    if (n_components < 1) {
        return std::unexpected(DomainError{"ml_pca_fit", "expected n_components >= 1"});
    }
    ml::PCA pca(n_components);
    pca.fit(*X);
    return ml_pca_model_to_matrix(pca);
}

Result<Matrix<double>> eval_ml_pca_transform(const Matrix<double>& X_m,
                                             const Matrix<double>& model_m) {
    auto X = matrix_to_ml_mat(X_m, "ml_pca_transform");
    if (!X) {
        return std::unexpected(X.error());
    }
    auto pca = ml_pca_from_matrix(model_m, "ml_pca_transform");
    if (!pca) {
        return std::unexpected(pca.error());
    }
    return grid_to_matrix(pca->transform(*X));
}

Result<Matrix<double>> eval_ml_pca_fit_transform(const Matrix<double>& X_m, int n_components) {
    auto X = matrix_to_ml_mat(X_m, "ml_pca_fit_transform");
    if (!X) {
        return std::unexpected(X.error());
    }
    if (n_components < 1) {
        return std::unexpected(DomainError{"ml_pca_fit_transform", "expected n_components >= 1"});
    }
    ml::PCA pca(n_components);
    return grid_to_matrix(pca.fit_transform(*X));
}

Result<Matrix<double>> eval_ml_kmeans_fit(const Matrix<double>& X_m, int k) {
    auto X = matrix_to_ml_mat(X_m, "ml_kmeans_fit");
    if (!X) {
        return std::unexpected(X.error());
    }
    if (k < 1) {
        return std::unexpected(DomainError{"ml_kmeans_fit", "expected k >= 1"});
    }
    ml::KMeans km(k);
    km.fit(*X);
    return grid_to_matrix(km.centers);
}

Result<Matrix<double>> eval_ml_kmeans_predict(const Matrix<double>& X_m,
                                              const Matrix<double>& model_m) {
    auto X = matrix_to_ml_mat(X_m, "ml_kmeans_predict");
    if (!X) {
        return std::unexpected(X.error());
    }
    auto km = ml_kmeans_from_matrix(model_m, "ml_kmeans_predict");
    if (!km) {
        return std::unexpected(km.error());
    }
    return vector_to_column(km->predict(*X));
}

Result<double> eval_ml_kmeans_inertia(const Matrix<double>& X_m,
                                      const Matrix<double>& model_m) {
    auto X = matrix_to_ml_mat(X_m, "ml_kmeans_inertia");
    if (!X) {
        return std::unexpected(X.error());
    }
    auto km = ml_kmeans_from_matrix(model_m, "ml_kmeans_inertia");
    if (!km) {
        return std::unexpected(km.error());
    }
    km->labels_ = km->predict(*X);
    return km->inertia(*X);
}

Matrix<double> ml_gmm_to_matrix(const ml::GaussianMixture& gmm) {
    const int K = static_cast<int>(gmm.means.size());
    const int p = K > 0 ? static_cast<int>(gmm.means[0].size()) : 0;
    // Header uses columns 0..2 (K, p, log_likelihood); weights use columns 0..K-1.
    const int cols = std::max({p, K, 3});
    Matrix<double> out(static_cast<size_t>(2 * K + 2), static_cast<size_t>(std::max(cols, 1)));
    out(0, 0) = static_cast<double>(K);
    out(0, 1) = static_cast<double>(p);
    out(0, 2) = gmm.log_likelihood;
    for (int k = 0; k < K; ++k) {
        for (int j = 0; j < p; ++j) {
            out(static_cast<size_t>(1 + k), static_cast<size_t>(j)) =
                gmm.means[static_cast<size_t>(k)][static_cast<size_t>(j)];
            out(static_cast<size_t>(1 + K + k), static_cast<size_t>(j)) =
                gmm.variances[static_cast<size_t>(k)][static_cast<size_t>(j)];
        }
        out(static_cast<size_t>(2 * K + 1), static_cast<size_t>(k)) =
            gmm.weights[static_cast<size_t>(k)];
    }
    return out;
}

Matrix<double> ml_standard_scaler_to_matrix(const ml::StandardScaler& sc) {
    const size_t nf = sc.mean_.size();
    Matrix<double> out(2, nf);
    for (size_t j = 0; j < nf; ++j) {
        out(0, j) = sc.mean_[j];
        out(1, j) = sc.std_[j];
    }
    return out;
}

Result<ml::GaussianMixture> ml_gmm_from_matrix(const Matrix<double>& model, const char* fn) {
    if (model.rows() < 3 || model.cols() < 1) {
        return std::unexpected(DomainError{fn, "expected GMM model matrix"});
    }
    const int K = static_cast<int>(model(0, 0));
    const int p = static_cast<int>(model(0, 1));
    const int need_cols = std::max({p, K, 3});
    if (K < 1 || p < 1 || model.rows() != static_cast<size_t>(2 * K + 2) ||
        model.cols() < static_cast<size_t>(need_cols)) {
        return std::unexpected(DomainError{fn, "invalid GMM model layout"});
    }
    ml::GaussianMixture gmm;
    gmm.config.n_components = static_cast<size_t>(K);
    gmm.log_likelihood = model(0, 2);
    gmm.means.resize(static_cast<size_t>(K));
    gmm.variances.resize(static_cast<size_t>(K));
    gmm.weights.resize(static_cast<size_t>(K));
    for (int k = 0; k < K; ++k) {
        gmm.means[static_cast<size_t>(k)].resize(static_cast<size_t>(p));
        gmm.variances[static_cast<size_t>(k)].resize(static_cast<size_t>(p));
        for (int j = 0; j < p; ++j) {
            gmm.means[static_cast<size_t>(k)][static_cast<size_t>(j)] =
                model(static_cast<size_t>(1 + k), static_cast<size_t>(j));
            gmm.variances[static_cast<size_t>(k)][static_cast<size_t>(j)] =
                model(static_cast<size_t>(1 + K + k), static_cast<size_t>(j));
        }
        gmm.weights[static_cast<size_t>(k)] =
            model(static_cast<size_t>(2 * K + 1), static_cast<size_t>(k));
    }
    return gmm;
}

Result<Matrix<double>> eval_ml_gmm_fit(const Matrix<double>& X_m, int n_components) {
    auto X = matrix_to_ml_mat(X_m, "ml_gmm_fit");
    if (!X) {
        return std::unexpected(X.error());
    }
    if (X->empty()) {
        return std::unexpected(DomainError{"ml_gmm_fit", "expected non-empty X"});
    }
    if (n_components < 1) {
        return std::unexpected(DomainError{"ml_gmm_fit", "expected n_components >= 1"});
    }
    ml::GaussianMixture gmm;
    gmm.config.n_components = static_cast<size_t>(n_components);
    gmm.fit(*X);
    return ml_gmm_to_matrix(gmm);
}

Result<Matrix<double>> eval_ml_gmm_predict(const Matrix<double>& X_m,
                                         const Matrix<double>& model_m) {
    auto X = matrix_to_ml_mat(X_m, "ml_gmm_predict");
    if (!X) {
        return std::unexpected(X.error());
    }
    auto gmm = ml_gmm_from_matrix(model_m, "ml_gmm_predict");
    if (!gmm) {
        return std::unexpected(gmm.error());
    }
    return vector_to_column(gmm->predict(*X));
}

Result<Matrix<double>> eval_ml_gmm_predict_proba(const Matrix<double>& X_m,
                                               const Matrix<double>& model_m) {
    auto X = matrix_to_ml_mat(X_m, "ml_gmm_predict_proba");
    if (!X) {
        return std::unexpected(X.error());
    }
    auto gmm = ml_gmm_from_matrix(model_m, "ml_gmm_predict_proba");
    if (!gmm) {
        return std::unexpected(gmm.error());
    }
    return grid_to_matrix(gmm->predict_proba(*X));
}

Result<Matrix<double>> eval_ml_dbscan_fit(const Matrix<double>& X_m, double eps,
                                          int min_samples) {
    auto X = matrix_to_ml_mat(X_m, "ml_dbscan_fit");
    if (!X) {
        return std::unexpected(X.error());
    }
    if (X->empty()) {
        return std::unexpected(DomainError{"ml_dbscan_fit", "expected non-empty X"});
    }
    if (min_samples < 1) {
        return std::unexpected(DomainError{"ml_dbscan_fit", "expected min_samples >= 1"});
    }
    ml::DBSCAN db(eps, min_samples);
    db.fit(*X);
    return vector_to_column(db.labels_);
}

Result<Matrix<double>> eval_ml_spectral_clustering(const Matrix<double>& X_m, int k,
                                                  double sigma, int n_neighbors) {
    auto X = matrix_to_ml_mat(X_m, "ml_spectral_clustering");
    if (!X) {
        return std::unexpected(X.error());
    }
    if (X->empty()) {
        return std::unexpected(
            DomainError{"ml_spectral_clustering", "expected non-empty X"});
    }
    if (k < 1) {
        return std::unexpected(DomainError{"ml_spectral_clustering", "expected k >= 1"});
    }
    auto labels = ml::spectral_clustering(*X, k, sigma, n_neighbors);
    return int_vector_to_column(labels);
}

Matrix<double> ml_isolation_forest_to_matrix(const ml::IsolationForestState& state) {
    size_t rows = 1;
    for (const auto& tree : state.trees) {
        rows += 1 + tree.nodes.size();
    }
    Matrix<double> out(rows, 5);
    out(0, 0) = static_cast<double>(state.n_trees);
    out(0, 1) = static_cast<double>(state.sample_size);
    out(0, 2) = static_cast<double>(state.seed);
    out(0, 3) = static_cast<double>(state.subsample_size);
    out(0, 4) = static_cast<double>(state.n_features);
    size_t row = 1;
    for (const auto& tree : state.trees) {
        out(row, 0) = static_cast<double>(tree.nodes.size());
        out(row, 1) = static_cast<double>(tree.height_limit);
        ++row;
        for (const auto& node : tree.nodes) {
            out(row, 0) = node.feature;
            out(row, 1) = node.threshold;
            out(row, 2) = node.left;
            out(row, 3) = node.right;
            out(row, 4) = static_cast<double>(node.size);
            ++row;
        }
    }
    return out;
}

Result<ml::IsolationForest> ml_isolation_forest_from_matrix(const Matrix<double>& model,
                                                            const char* fn) {
    if (model.rows() < 1 || model.cols() < 5) {
        return std::unexpected(DomainError{fn, "expected IsolationForest model matrix"});
    }
    ml::IsolationForestState state;
    state.n_trees = static_cast<size_t>(model(0, 0));
    state.sample_size = static_cast<size_t>(model(0, 1));
    state.seed = static_cast<unsigned>(model(0, 2));
    state.subsample_size = static_cast<size_t>(model(0, 3));
    state.n_features = static_cast<int>(model(0, 4));
    size_t row = 1;
    while (row < model.rows()) {
        const size_t n_nodes = static_cast<size_t>(model(row, 0));
        const size_t height_limit = static_cast<size_t>(model(row, 1));
        if (row + 1 + n_nodes > model.rows()) {
            return std::unexpected(DomainError{fn, "invalid IsolationForest model layout"});
        }
        ml::IsolationForestState::TreeState tree;
        tree.height_limit = height_limit;
        tree.nodes.reserve(n_nodes);
        ++row;
        for (size_t i = 0; i < n_nodes; ++i) {
            ml::IsolationForestState::NodeState node;
            node.feature = static_cast<int>(model(row, 0));
            node.threshold = model(row, 1);
            node.left = static_cast<int>(model(row, 2));
            node.right = static_cast<int>(model(row, 3));
            node.size = static_cast<size_t>(model(row, 4));
            tree.nodes.push_back(node);
            ++row;
        }
        state.trees.push_back(std::move(tree));
    }
    return ml::IsolationForest::from_state(state);
}

Result<std::string> parse_ml_linkage(const std::string& text, const char* fn) {
    std::string linkage;
    if (parse_quoted_string(text, linkage)) {
        linkage = lower(trim_copy(linkage));
    } else {
        linkage = lower(trim_copy(text));
    }
    if (linkage == "ward" || linkage == "single" || linkage == "average" ||
        linkage == "complete") {
        return linkage;
    }
    return std::unexpected(
        DomainError{fn, "expected linkage ward|single|average|complete"});
}

Result<Matrix<double>> eval_ml_isolation_forest_fit(const Matrix<double>& X_m, size_t n_trees,
                                                     size_t sample_size, unsigned seed) {
    auto X = matrix_to_ml_mat(X_m, "ml_isolation_forest_fit");
    if (!X) {
        return std::unexpected(X.error());
    }
    if (X->empty()) {
        return std::unexpected(DomainError{"ml_isolation_forest_fit", "expected non-empty X"});
    }
    if (n_trees < 1) {
        return std::unexpected(DomainError{"ml_isolation_forest_fit", "expected n_trees >= 1"});
    }
    if (sample_size < 1) {
        return std::unexpected(
            DomainError{"ml_isolation_forest_fit", "expected sample_size >= 1"});
    }
    ml::IsolationForest iso(n_trees, sample_size, seed);
    iso.fit(*X);
    return ml_isolation_forest_to_matrix(iso.export_state());
}

Result<Matrix<double>> eval_ml_isolation_forest_score(const Matrix<double>& X_m,
                                                      const Matrix<double>& model_m) {
    auto X = matrix_to_ml_mat(X_m, "ml_isolation_forest_score");
    if (!X) {
        return std::unexpected(X.error());
    }
    auto iso = ml_isolation_forest_from_matrix(model_m, "ml_isolation_forest_score");
    if (!iso) {
        return std::unexpected(iso.error());
    }
    return vector_to_column(iso->anomaly_scores(*X));
}

Result<Matrix<double>> eval_ml_agglomerative_fit(const Matrix<double>& X_m, int n_clusters,
                                                 const std::string& linkage) {
    auto X = matrix_to_ml_mat(X_m, "ml_agglomerative_fit");
    if (!X) {
        return std::unexpected(X.error());
    }
    if (X->empty()) {
        return std::unexpected(DomainError{"ml_agglomerative_fit", "expected non-empty X"});
    }
    if (n_clusters < 1) {
        return std::unexpected(DomainError{"ml_agglomerative_fit", "expected n_clusters >= 1"});
    }
    ml::AgglomerativeClustering ac(n_clusters, linkage);
    ac.fit(*X);
    return vector_to_column(ac.labels_);
}

Result<Matrix<double>> eval_ml_tsne_fit(const Matrix<double>& X_m, double perplexity, int n_iter,
                                        unsigned seed) {
    auto X = matrix_to_ml_mat(X_m, "ml_tsne_fit");
    if (!X) {
        return std::unexpected(X.error());
    }
    if (X->empty()) {
        return std::unexpected(DomainError{"ml_tsne_fit", "expected non-empty X"});
    }
    if (perplexity <= 0.0) {
        return std::unexpected(DomainError{"ml_tsne_fit", "expected perplexity > 0"});
    }
    if (n_iter < 1) {
        return std::unexpected(DomainError{"ml_tsne_fit", "expected n_iter >= 1"});
    }
    ml::TSNE tsne(2, perplexity, 200.0, n_iter, seed);
    return grid_to_matrix(tsne.fit_transform(*X));
}

Result<ml::StandardScaler> ml_standard_scaler_from_matrix(const Matrix<double>& model,
                                                          const char* fn) {
    if (model.rows() != 2 || model.cols() < 1) {
        return std::unexpected(
            DomainError{fn, "expected StandardScaler model (2 x n_features)"});
    }
    ml::StandardScaler sc;
    sc.mean_.resize(model.cols());
    sc.std_.resize(model.cols());
    for (size_t j = 0; j < model.cols(); ++j) {
        sc.mean_[j] = model(0, j);
        sc.std_[j] = model(1, j);
    }
    return sc;
}

Matrix<double> ml_minmax_scaler_to_matrix(const ml::MinMaxScaler& sc) {
    const size_t nf = sc.min_.size();
    Matrix<double> out(2, nf);
    for (size_t j = 0; j < nf; ++j) {
        out(0, j) = sc.min_[j];
        out(1, j) = sc.max_[j];
    }
    return out;
}

Result<ml::MinMaxScaler> ml_minmax_scaler_from_matrix(const Matrix<double>& model,
                                                      const char* fn) {
    if (model.rows() != 2 || model.cols() < 1) {
        return std::unexpected(
            DomainError{fn, "expected MinMaxScaler model (2 x n_features)"});
    }
    ml::MinMaxScaler sc;
    sc.min_.resize(model.cols());
    sc.max_.resize(model.cols());
    for (size_t j = 0; j < model.cols(); ++j) {
        sc.min_[j] = model(0, j);
        sc.max_[j] = model(1, j);
    }
    return sc;
}

Result<Matrix<double>> eval_ml_standard_scaler_fit(const Matrix<double>& X_m) {
    auto X = matrix_to_ml_mat(X_m, "ml_standard_scaler_fit");
    if (!X) {
        return std::unexpected(X.error());
    }
    ml::StandardScaler sc;
    sc.fit(*X);
    return ml_standard_scaler_to_matrix(sc);
}

Result<Matrix<double>> eval_ml_standard_scaler_transform(const Matrix<double>& X_m,
                                                         const Matrix<double>& model_m) {
    auto X = matrix_to_ml_mat(X_m, "ml_standard_scaler_transform");
    if (!X) {
        return std::unexpected(X.error());
    }
    auto sc = ml_standard_scaler_from_matrix(model_m, "ml_standard_scaler_transform");
    if (!sc) {
        return std::unexpected(sc.error());
    }
    return grid_to_matrix(sc->transform(*X));
}

Result<Matrix<double>> eval_ml_minmax_scaler_fit(const Matrix<double>& X_m) {
    auto X = matrix_to_ml_mat(X_m, "ml_minmax_scaler_fit");
    if (!X) {
        return std::unexpected(X.error());
    }
    ml::MinMaxScaler sc;
    sc.fit(*X);
    return ml_minmax_scaler_to_matrix(sc);
}

Result<Matrix<double>> eval_ml_minmax_scaler_transform(const Matrix<double>& X_m,
                                                       const Matrix<double>& model_m) {
    auto X = matrix_to_ml_mat(X_m, "ml_minmax_scaler_transform");
    if (!X) {
        return std::unexpected(X.error());
    }
    auto sc = ml_minmax_scaler_from_matrix(model_m, "ml_minmax_scaler_transform");
    if (!sc) {
        return std::unexpected(sc.error());
    }
    return grid_to_matrix(sc->transform(*X));
}

Matrix<double> ml_confusion_matrix_to_matrix(const ml::ConfusionMatrix& cm) {
    Matrix<double> out(2, 2);
    out(0, 0) = static_cast<double>(cm.tp);
    out(0, 1) = static_cast<double>(cm.fp);
    out(1, 0) = static_cast<double>(cm.fn);
    out(1, 1) = static_cast<double>(cm.tn);
    return out;
}

Matrix<double> ml_roc_curve_to_matrix(const std::vector<ml::ROCPoint>& curve) {
    Matrix<double> out(curve.size(), 3);
    for (size_t i = 0; i < curve.size(); ++i) {
        out(i, 0) = curve[i].threshold;
        out(i, 1) = curve[i].fpr;
        out(i, 2) = curve[i].tpr;
    }
    return out;
}

Matrix<double> ml_precision_recall_curve_to_matrix(const std::vector<ml::PRPoint>& curve) {
    Matrix<double> out(curve.size(), 3);
    for (size_t i = 0; i < curve.size(); ++i) {
        out(i, 0) = curve[i].threshold;
        out(i, 1) = curve[i].precision;
        out(i, 2) = curve[i].recall;
    }
    return out;
}

Result<Matrix<double>> eval_ml_confusion_matrix(const Matrix<double>& y_pred_m,
                                                 const Matrix<double>& y_true_m,
                                                 double threshold) {
    auto y_pred = matrix_to_ml_vec(y_pred_m, "ml_confusion_matrix");
    if (!y_pred) {
        return std::unexpected(y_pred.error());
    }
    auto y_true = matrix_to_ml_vec(y_true_m, "ml_confusion_matrix");
    if (!y_true) {
        return std::unexpected(y_true.error());
    }
    return ml_confusion_matrix_to_matrix(ml::confusion_matrix(*y_pred, *y_true, threshold));
}

Result<Matrix<double>> eval_ml_roc_curve(const Matrix<double>& y_pred_m,
                                         const Matrix<double>& y_true_m) {
    auto y_pred = matrix_to_ml_vec(y_pred_m, "ml_roc_curve");
    if (!y_pred) {
        return std::unexpected(y_pred.error());
    }
    auto y_true = matrix_to_ml_vec(y_true_m, "ml_roc_curve");
    if (!y_true) {
        return std::unexpected(y_true.error());
    }
    return ml_roc_curve_to_matrix(ml::roc_curve(*y_pred, *y_true));
}

Result<Matrix<double>> eval_ml_precision_recall_curve(const Matrix<double>& y_pred_m,
                                                      const Matrix<double>& y_true_m) {
    auto y_pred = matrix_to_ml_vec(y_pred_m, "ml_precision_recall_curve");
    if (!y_pred) {
        return std::unexpected(y_pred.error());
    }
    auto y_true = matrix_to_ml_vec(y_true_m, "ml_precision_recall_curve");
    if (!y_true) {
        return std::unexpected(y_true.error());
    }
    return ml_precision_recall_curve_to_matrix(ml::precision_recall_curve(*y_pred, *y_true));
}

Result<Matrix<double>> eval_ml_lasso_fit(const Matrix<double>& X_m, const Matrix<double>& y_m,
                                        double alpha) {
    auto X = matrix_to_ml_mat(X_m, "ml_lasso_fit");
    if (!X) {
        return std::unexpected(X.error());
    }
    auto y = matrix_to_ml_vec(y_m, "ml_lasso_fit");
    if (!y) {
        return std::unexpected(y.error());
    }
    if (y->size() != X->size()) {
        return std::unexpected(DimensionMismatch{y->size(), X->size()});
    }
    ml::LassoRegression lr(alpha);
    lr.fit(*X, *y);
    return ml_model_to_matrix(lr.coef, lr.intercept);
}

Result<Matrix<double>> eval_ml_lasso_predict(const Matrix<double>& X_m,
                                            const Matrix<double>& model_m) {
    auto X = matrix_to_ml_mat(X_m, "ml_lasso_predict");
    if (!X) {
        return std::unexpected(X.error());
    }
    auto params = ml_model_from_matrix(model_m, "ml_lasso_predict");
    if (!params) {
        return std::unexpected(params.error());
    }
    ml::LassoRegression lr(1.0);
    lr.coef = params->first;
    lr.intercept = params->second;
    return vector_to_column(lr.predict(*X));
}

Result<Matrix<double>> eval_ml_elastic_net_fit(const Matrix<double>& X_m,
                                               const Matrix<double>& y_m, double alpha,
                                               double l1_ratio) {
    auto X = matrix_to_ml_mat(X_m, "ml_elastic_net_fit");
    if (!X) {
        return std::unexpected(X.error());
    }
    auto y = matrix_to_ml_vec(y_m, "ml_elastic_net_fit");
    if (!y) {
        return std::unexpected(y.error());
    }
    if (y->size() != X->size()) {
        return std::unexpected(DimensionMismatch{y->size(), X->size()});
    }
    ml::ElasticNet en(alpha, l1_ratio);
    en.fit(*X, *y);
    return ml_model_to_matrix(en.coef, en.intercept);
}

Result<Matrix<double>> eval_ml_elastic_net_predict(const Matrix<double>& X_m,
                                                  const Matrix<double>& model_m) {
    auto X = matrix_to_ml_mat(X_m, "ml_elastic_net_predict");
    if (!X) {
        return std::unexpected(X.error());
    }
    auto params = ml_model_from_matrix(model_m, "ml_elastic_net_predict");
    if (!params) {
        return std::unexpected(params.error());
    }
    ml::ElasticNet en(1.0, 0.5);
    en.coef = params->first;
    en.intercept = params->second;
    return vector_to_column(en.predict(*X));
}

Matrix<double> ml_knn_to_matrix(const ml::KNN& knn) {
    const size_t n = knn.X_train.size();
    const size_t p = n > 0 ? knn.X_train[0].size() : 0;
    Matrix<double> out(n + 1, p + 1);
    out(0, 0) = static_cast<double>(knn.k);
    out(0, 1) = static_cast<double>(p);
    out(0, 2) = static_cast<double>(n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < p; ++j) {
            out(i + 1, j) = knn.X_train[i][j];
        }
        out(i + 1, p) = knn.y_train[i];
    }
    return out;
}

Result<ml::KNN> ml_knn_from_matrix(const Matrix<double>& model, const char* fn) {
    if (model.rows() < 2 || model.cols() < 2) {
        return std::unexpected(
            DomainError{fn, "expected KNN model with header row and training data"});
    }
    const int k = static_cast<int>(model(0, 0));
    const int p = static_cast<int>(model(0, 1));
    const int n = static_cast<int>(model(0, 2));
    if (k < 1 || p < 1 || n < 1 || model.rows() != static_cast<size_t>(n + 1) ||
        model.cols() != static_cast<size_t>(p + 1)) {
        return std::unexpected(DomainError{fn, "invalid KNN model layout"});
    }
    ml::KNN knn(k);
    knn.X_train.resize(static_cast<size_t>(n));
    knn.y_train.resize(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        knn.X_train[static_cast<size_t>(i)].resize(static_cast<size_t>(p));
        for (int j = 0; j < p; ++j) {
            knn.X_train[static_cast<size_t>(i)][static_cast<size_t>(j)] =
                model(static_cast<size_t>(i + 1), static_cast<size_t>(j));
        }
        knn.y_train[static_cast<size_t>(i)] = model(static_cast<size_t>(i + 1), static_cast<size_t>(p));
    }
    return knn;
}

Result<Matrix<double>> eval_ml_knn_fit(const Matrix<double>& X_m, const Matrix<double>& y_m,
                                       int k) {
    auto X = matrix_to_ml_mat(X_m, "ml_knn_fit");
    if (!X) {
        return std::unexpected(X.error());
    }
    auto y = matrix_to_ml_vec(y_m, "ml_knn_fit");
    if (!y) {
        return std::unexpected(y.error());
    }
    if (y->size() != X->size()) {
        return std::unexpected(DimensionMismatch{y->size(), X->size()});
    }
    ml::KNN knn(k);
    knn.fit(*X, *y);
    return ml_knn_to_matrix(knn);
}

Result<Matrix<double>> eval_ml_knn_predict(const Matrix<double>& X_m,
                                          const Matrix<double>& model_m) {
    auto X = matrix_to_ml_mat(X_m, "ml_knn_predict");
    if (!X) {
        return std::unexpected(X.error());
    }
    auto knn = ml_knn_from_matrix(model_m, "ml_knn_predict");
    if (!knn) {
        return std::unexpected(knn.error());
    }
    return vector_to_column(knn->predict(*X));
}

Matrix<double> ml_naive_bayes_to_matrix(const ml::NaiveBayes& nb) {
    const size_t C = nb.classes.size();
    const size_t p = C > 0 && !nb.mean.empty() ? nb.mean[0].size() : 0;
    Matrix<double> out(1 + 4 * C, std::max(p, size_t{1}));
    out(0, 0) = static_cast<double>(C);
    out(0, 1) = static_cast<double>(p);
    for (size_t c = 0; c < C; ++c) {
        out(1 + c, 0) = nb.classes[c];
        for (size_t j = 0; j < p; ++j) {
            out(1 + C + c, j) = nb.mean[c][j];
            out(1 + 2 * C + c, j) = nb.var[c][j];
        }
        out(1 + 3 * C + c, 0) = nb.class_prior[c];
    }
    return out;
}

Result<ml::NaiveBayes> ml_naive_bayes_from_matrix(const Matrix<double>& model, const char* fn) {
    if (model.rows() < 5) {
        return std::unexpected(DomainError{fn, "expected NaiveBayes model matrix"});
    }
    const int C = static_cast<int>(model(0, 0));
    const int p = static_cast<int>(model(0, 1));
    if (C < 1 || p < 1 || model.rows() != static_cast<size_t>(1 + 4 * C) ||
        model.cols() != static_cast<size_t>(p)) {
        return std::unexpected(DomainError{fn, "invalid NaiveBayes model layout"});
    }
    ml::NaiveBayes nb;
    nb.classes.resize(static_cast<size_t>(C));
    nb.mean.assign(static_cast<size_t>(C), ml::Vec(static_cast<size_t>(p)));
    nb.var.assign(static_cast<size_t>(C), ml::Vec(static_cast<size_t>(p)));
    nb.class_prior.resize(static_cast<size_t>(C));
    for (int c = 0; c < C; ++c) {
        nb.classes[static_cast<size_t>(c)] = model(static_cast<size_t>(1 + c), 0);
        for (int j = 0; j < p; ++j) {
            nb.mean[static_cast<size_t>(c)][static_cast<size_t>(j)] =
                model(static_cast<size_t>(1 + C + c), static_cast<size_t>(j));
            nb.var[static_cast<size_t>(c)][static_cast<size_t>(j)] =
                model(static_cast<size_t>(1 + 2 * C + c), static_cast<size_t>(j));
        }
        nb.class_prior[static_cast<size_t>(c)] = model(static_cast<size_t>(1 + 3 * C + c), 0);
    }
    return nb;
}

Result<Matrix<double>> eval_ml_naive_bayes_fit(const Matrix<double>& X_m,
                                               const Matrix<double>& y_m) {
    auto X = matrix_to_ml_mat(X_m, "ml_naive_bayes_fit");
    if (!X) {
        return std::unexpected(X.error());
    }
    auto y = matrix_to_ml_vec(y_m, "ml_naive_bayes_fit");
    if (!y) {
        return std::unexpected(y.error());
    }
    if (y->size() != X->size()) {
        return std::unexpected(DimensionMismatch{y->size(), X->size()});
    }
    ml::NaiveBayes nb;
    nb.fit(*X, *y);
    return ml_naive_bayes_to_matrix(nb);
}

Result<Matrix<double>> eval_ml_naive_bayes_predict(const Matrix<double>& X_m,
                                                  const Matrix<double>& model_m) {
    auto X = matrix_to_ml_mat(X_m, "ml_naive_bayes_predict");
    if (!X) {
        return std::unexpected(X.error());
    }
    auto nb = ml_naive_bayes_from_matrix(model_m, "ml_naive_bayes_predict");
    if (!nb) {
        return std::unexpected(nb.error());
    }
    return vector_to_column(nb->predict(*X));
}

Matrix<double> ml_lda_to_matrix(const ml::LDA& lda) {
    const size_t C = lda.classes.size();
    const size_t p = C > 0 && !lda.mean.empty() ? lda.mean[0].size() : 0;
    const size_t n_comp = lda.projection.size();
    const size_t n_cols = std::max(p, size_t{4});
    Matrix<double> out(1 + 4 * C + n_comp + 1, n_cols);
    out(0, 0) = static_cast<double>(C);
    out(0, 1) = static_cast<double>(p);
    out(0, 2) = static_cast<double>(n_comp);
    out(0, 3) = lda.reg_epsilon;
    for (size_t c = 0; c < C; ++c) {
        out(1 + c, 0) = lda.classes[c];
        for (size_t j = 0; j < p; ++j) {
            out(1 + C + c, j) = lda.mean[c][j];
            out(1 + 2 * C + c, j) = lda.discrim_coef[c][j];
        }
        out(1 + 3 * C + c, 0) = lda.discrim_const[c];
    }
    for (size_t r = 0; r < n_comp; ++r) {
        for (size_t j = 0; j < p; ++j) {
            out(1 + 4 * C + r, j) = lda.projection[r][j];
        }
    }
    for (size_t j = 0; j < p; ++j) {
        out(1 + 4 * C + n_comp, j) = lda.overall_mean[j];
    }
    return out;
}

Result<ml::LDA> ml_lda_from_matrix(const Matrix<double>& model, const char* fn) {
    if (model.rows() < 6) {
        return std::unexpected(DomainError{fn, "expected LDA model matrix"});
    }
    const int C = static_cast<int>(model(0, 0));
    const int p = static_cast<int>(model(0, 1));
    const int n_comp = static_cast<int>(model(0, 2));
    const double reg_eps = model(0, 3);
    if (C < 2 || p < 1 || n_comp < 1 || model.cols() < static_cast<size_t>(p) ||
        model.rows() != static_cast<size_t>(1 + 4 * C + n_comp + 1)) {
        return std::unexpected(DomainError{fn, "invalid LDA model layout"});
    }
    ml::LDA lda(reg_eps, n_comp);
    lda.classes.resize(static_cast<size_t>(C));
    lda.mean.assign(static_cast<size_t>(C), ml::Vec(static_cast<size_t>(p)));
    lda.discrim_coef.assign(static_cast<size_t>(C), ml::Vec(static_cast<size_t>(p)));
    lda.discrim_const.resize(static_cast<size_t>(C));
    lda.projection.assign(static_cast<size_t>(n_comp), ml::Vec(static_cast<size_t>(p)));
    lda.overall_mean.resize(static_cast<size_t>(p));
    for (int c = 0; c < C; ++c) {
        lda.classes[static_cast<size_t>(c)] = model(static_cast<size_t>(1 + c), 0);
        for (int j = 0; j < p; ++j) {
            lda.mean[static_cast<size_t>(c)][static_cast<size_t>(j)] =
                model(static_cast<size_t>(1 + C + c), static_cast<size_t>(j));
            lda.discrim_coef[static_cast<size_t>(c)][static_cast<size_t>(j)] =
                model(static_cast<size_t>(1 + 2 * C + c), static_cast<size_t>(j));
        }
        lda.discrim_const[static_cast<size_t>(c)] = model(static_cast<size_t>(1 + 3 * C + c), 0);
    }
    for (int r = 0; r < n_comp; ++r) {
        for (int j = 0; j < p; ++j) {
            lda.projection[static_cast<size_t>(r)][static_cast<size_t>(j)] =
                model(static_cast<size_t>(1 + 4 * C + r), static_cast<size_t>(j));
        }
    }
    for (int j = 0; j < p; ++j) {
        lda.overall_mean[static_cast<size_t>(j)] =
            model(static_cast<size_t>(1 + 4 * C + n_comp), static_cast<size_t>(j));
    }
    return lda;
}

Result<Matrix<double>> eval_ml_lda_fit(const Matrix<double>& X_m, const Matrix<double>& y_m,
                                       int n_components) {
    auto X = matrix_to_ml_mat(X_m, "ml_lda_fit");
    if (!X) {
        return std::unexpected(X.error());
    }
    auto y = matrix_to_ml_vec(y_m, "ml_lda_fit");
    if (!y) {
        return std::unexpected(y.error());
    }
    if (y->size() != X->size()) {
        return std::unexpected(DimensionMismatch{y->size(), X->size()});
    }
    ml::LDA lda(1e-6, n_components);
    lda.fit(*X, *y);
    return ml_lda_to_matrix(lda);
}

Result<Matrix<double>> eval_ml_lda_predict(const Matrix<double>& X_m,
                                          const Matrix<double>& model_m) {
    auto X = matrix_to_ml_mat(X_m, "ml_lda_predict");
    if (!X) {
        return std::unexpected(X.error());
    }
    auto lda = ml_lda_from_matrix(model_m, "ml_lda_predict");
    if (!lda) {
        return std::unexpected(lda.error());
    }
    return vector_to_column(lda->predict(*X));
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

Result<Matrix<double>> eval_ml_lda_transform(const Matrix<double>& X_m,
                                            const Matrix<double>& model_m) {
    auto X = matrix_to_ml_mat(X_m, "ml_lda_transform");
    if (!X) {
        return std::unexpected(X.error());
    }
    auto lda = ml_lda_from_matrix(model_m, "ml_lda_transform");
    if (!lda) {
        return std::unexpected(lda.error());
    }
    return grid_to_matrix(lda->transform(*X));
}

Matrix<double> ml_qda_to_matrix(const ml::QDA& qda) {
    const size_t C = qda.classes.size();
    const size_t p = C > 0 && !qda.mean.empty() ? qda.mean[0].size() : 0;
    const size_t pp = p * p;
    const size_t n_cols = std::max({p, pp, size_t{3}});
    Matrix<double> out(1 + 5 * C, n_cols);
    out(0, 0) = static_cast<double>(C);
    out(0, 1) = static_cast<double>(p);
    out(0, 2) = qda.reg_epsilon;
    for (size_t c = 0; c < C; ++c) {
        out(1 + c, 0) = qda.classes[c];
        for (size_t j = 0; j < p; ++j) {
            out(1 + C + c, j) = qda.mean[c][j];
            out(1 + 2 * C + c, j) = qda.linear_coef[c][j];
        }
        out(1 + 3 * C + c, 0) = qda.discrim_const[c];
        out(1 + 3 * C + c, 1) = qda.class_prior[c];
        for (size_t j = 0; j < pp; ++j) {
            out(1 + 4 * C + c, j) = qda.quad_coef[c][j];
        }
    }
    return out;
}

Result<ml::QDA> ml_qda_from_matrix(const Matrix<double>& model, const char* fn) {
    if (model.rows() < 6) {
        return std::unexpected(DomainError{fn, "expected QDA model matrix"});
    }
    const int C = static_cast<int>(model(0, 0));
    const int p = static_cast<int>(model(0, 1));
    const double reg_eps = model(0, 2);
    const int pp = p * p;
    if (C < 1 || p < 1 || model.rows() != static_cast<size_t>(1 + 5 * C) ||
        model.cols() < static_cast<size_t>(std::max(p, pp))) {
        return std::unexpected(DomainError{fn, "invalid QDA model layout"});
    }
    ml::QDA qda(reg_eps);
    qda.classes.resize(static_cast<size_t>(C));
    qda.mean.assign(static_cast<size_t>(C), ml::Vec(static_cast<size_t>(p)));
    qda.linear_coef.assign(static_cast<size_t>(C), ml::Vec(static_cast<size_t>(p)));
    qda.discrim_const.resize(static_cast<size_t>(C));
    qda.class_prior.resize(static_cast<size_t>(C));
    qda.quad_coef.assign(static_cast<size_t>(C), ml::Vec(static_cast<size_t>(pp)));
    for (int c = 0; c < C; ++c) {
        qda.classes[static_cast<size_t>(c)] = model(static_cast<size_t>(1 + c), 0);
        for (int j = 0; j < p; ++j) {
            qda.mean[static_cast<size_t>(c)][static_cast<size_t>(j)] =
                model(static_cast<size_t>(1 + C + c), static_cast<size_t>(j));
            qda.linear_coef[static_cast<size_t>(c)][static_cast<size_t>(j)] =
                model(static_cast<size_t>(1 + 2 * C + c), static_cast<size_t>(j));
        }
        qda.discrim_const[static_cast<size_t>(c)] = model(static_cast<size_t>(1 + 3 * C + c), 0);
        qda.class_prior[static_cast<size_t>(c)] = model(static_cast<size_t>(1 + 3 * C + c), 1);
        for (int j = 0; j < pp; ++j) {
            qda.quad_coef[static_cast<size_t>(c)][static_cast<size_t>(j)] =
                model(static_cast<size_t>(1 + 4 * C + c), static_cast<size_t>(j));
        }
    }
    return qda;
}

Result<Matrix<double>> eval_ml_qda_fit(const Matrix<double>& X_m, const Matrix<double>& y_m) {
    auto X = matrix_to_ml_mat(X_m, "ml_qda_fit");
    if (!X) {
        return std::unexpected(X.error());
    }
    auto y = matrix_to_ml_vec(y_m, "ml_qda_fit");
    if (!y) {
        return std::unexpected(y.error());
    }
    if (y->size() != X->size()) {
        return std::unexpected(DimensionMismatch{y->size(), X->size()});
    }
    ml::QDA qda;
    qda.fit(*X, *y);
    return ml_qda_to_matrix(qda);
}

Result<Matrix<double>> eval_ml_qda_predict(const Matrix<double>& X_m,
                                           const Matrix<double>& model_m) {
    auto X = matrix_to_ml_mat(X_m, "ml_qda_predict");
    if (!X) {
        return std::unexpected(X.error());
    }
    auto qda = ml_qda_from_matrix(model_m, "ml_qda_predict");
    if (!qda) {
        return std::unexpected(qda.error());
    }
    return vector_to_column(qda->predict(*X));
}

Matrix<double> ml_svm_to_matrix(const ml::SVM& svm) {
    const size_t n = svm.support_vectors.size();
    const size_t p = n > 0 ? svm.support_vectors[0].size() : 0;
    const size_t n_cols = std::max(p + 2, size_t{8});
    Matrix<double> out(n + 1, n_cols);
    out(0, 0) = svm.config.kernel == ml::SVMKernel::Linear ? 0.0 : 1.0;
    out(0, 1) = svm.config.C;
    out(0, 2) = svm.config.gamma;
    out(0, 3) = svm.b;
    out(0, 4) = svm.config.tol;
    out(0, 5) = static_cast<double>(svm.config.max_iter);
    out(0, 6) = static_cast<double>(p);
    out(0, 7) = static_cast<double>(n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < p; ++j) {
            out(i + 1, j) = svm.support_vectors[i][j];
        }
        out(i + 1, p) = svm.alphas[i];
        out(i + 1, p + 1) = svm.sv_labels[i];
    }
    return out;
}

Result<ml::SVM> ml_svm_from_matrix(const Matrix<double>& model, const char* fn) {
    if (model.rows() < 2 || model.cols() < 3) {
        return std::unexpected(
            DomainError{fn, "expected SVM model with header row and support vectors"});
    }
    const int kernel_code = static_cast<int>(model(0, 0));
    const double C = model(0, 1);
    const double gamma = model(0, 2);
    const double b = model(0, 3);
    const double tol = model(0, 4);
    const int max_iter = static_cast<int>(model(0, 5));
    const int p = static_cast<int>(model(0, 6));
    const int n = static_cast<int>(model(0, 7));
    const size_t min_cols = std::max(static_cast<size_t>(p + 2), size_t{8});
    if ((kernel_code != 0 && kernel_code != 1) || p < 1 || n < 1 ||
        model.rows() != static_cast<size_t>(n + 1) || model.cols() < min_cols) {
        return std::unexpected(DomainError{fn, "invalid SVM model layout"});
    }
    ml::SVM svm;
    svm.config.kernel = kernel_code == 0 ? ml::SVMKernel::Linear : ml::SVMKernel::RBF;
    svm.config.C = C;
    svm.config.gamma = gamma;
    svm.config.tol = tol;
    svm.config.max_iter = max_iter;
    svm.b = b;
    svm.support_vectors.resize(static_cast<size_t>(n));
    svm.alphas.resize(static_cast<size_t>(n));
    svm.sv_labels.resize(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        svm.support_vectors[static_cast<size_t>(i)].resize(static_cast<size_t>(p));
        for (int j = 0; j < p; ++j) {
            svm.support_vectors[static_cast<size_t>(i)][static_cast<size_t>(j)] =
                model(static_cast<size_t>(i + 1), static_cast<size_t>(j));
        }
        svm.alphas[static_cast<size_t>(i)] = model(static_cast<size_t>(i + 1), static_cast<size_t>(p));
        svm.sv_labels[static_cast<size_t>(i)] =
            model(static_cast<size_t>(i + 1), static_cast<size_t>(p + 1));
    }
    return svm;
}

Result<Matrix<double>> eval_ml_svm_fit(const Matrix<double>& X_m, const Matrix<double>& y_m,
                                      double C, double gamma, bool use_rbf) {
    auto X = matrix_to_ml_mat(X_m, "ml_svm_fit");
    if (!X) {
        return std::unexpected(X.error());
    }
    auto y = matrix_to_ml_vec(y_m, "ml_svm_fit");
    if (!y) {
        return std::unexpected(y.error());
    }
    if (y->size() != X->size()) {
        return std::unexpected(DimensionMismatch{y->size(), X->size()});
    }
    ml::SVM svm;
    svm.config.C = C;
    if (use_rbf) {
        svm.config.kernel = ml::SVMKernel::RBF;
        svm.config.gamma = gamma;
    }
    svm.fit(*X, *y);
    return ml_svm_to_matrix(svm);
}

Result<Matrix<double>> eval_ml_svm_predict(const Matrix<double>& X_m,
                                          const Matrix<double>& model_m) {
    auto X = matrix_to_ml_mat(X_m, "ml_svm_predict");
    if (!X) {
        return std::unexpected(X.error());
    }
    auto svm = ml_svm_from_matrix(model_m, "ml_svm_predict");
    if (!svm) {
        return std::unexpected(svm.error());
    }
    return vector_to_column(svm->predict(*X));
}


static double ml_tree_criterion_code(const std::string& criterion) {
    if (criterion == "entropy") {
        return 1.0;
    }
    if (criterion == "mse") {
        return 2.0;
    }
    return 0.0;
}

static std::string ml_tree_criterion_from_code(double code) {
    if (code == 1.0) {
        return "entropy";
    }
    if (code == 2.0) {
        return "mse";
    }
    return "gini";
}

Matrix<double> ml_decision_tree_to_matrix(const ml::DecisionTree& tree) {
    const size_t n_nodes = tree.nodes.size();
    Matrix<double> out(1 + n_nodes, 5);
    out(0, 0) = tree.max_depth;
    out(0, 1) = ml_tree_criterion_code(tree.criterion);
    out(0, 2) = static_cast<double>(n_nodes);
    for (size_t i = 0; i < n_nodes; ++i) {
        const auto& node = tree.nodes[i];
        out(1 + i, 0) = node.feature;
        out(1 + i, 1) = node.threshold;
        out(1 + i, 2) = node.value;
        out(1 + i, 3) = node.left;
        out(1 + i, 4) = node.right;
    }
    return out;
}

Result<ml::DecisionTree> ml_decision_tree_from_matrix(const Matrix<double>& model, const char* fn) {
    if (model.rows() < 2 || model.cols() < 5) {
        return std::unexpected(DomainError{fn, "expected DecisionTree model matrix"});
    }
    const int max_depth = static_cast<int>(model(0, 0));
    const std::string criterion = ml_tree_criterion_from_code(model(0, 1));
    const size_t n_nodes = static_cast<size_t>(model(0, 2));
    if (model.rows() != 1 + n_nodes) {
        return std::unexpected(DomainError{fn, "invalid DecisionTree model layout"});
    }
    ml::DecisionTree tree(max_depth, criterion);
    tree.nodes.resize(n_nodes);
    for (size_t i = 0; i < n_nodes; ++i) {
        auto& node = tree.nodes[i];
        node.feature = static_cast<int>(model(1 + i, 0));
        node.threshold = model(1 + i, 1);
        node.value = model(1 + i, 2);
        node.left = static_cast<int>(model(1 + i, 3));
        node.right = static_cast<int>(model(1 + i, 4));
    }
    return tree;
}

Matrix<double> ml_random_forest_to_matrix(const ml::RandomForest& rf) {
    size_t total_rows = 1;
    for (size_t t = 0; t < rf.trees.size(); ++t) {
        total_rows += 2 + rf.trees[t].nodes.size();
    }
    Matrix<double> out(total_rows, 5);
    out(0, 0) = static_cast<double>(rf.trees.size());
    out(0, 1) = static_cast<double>(rf.config.max_depth);
    out(0, 2) = rf.config.feature_subsample_ratio;
    out(0, 3) = rf.config.sample_subsample_ratio;
    out(0, 4) = static_cast<double>(rf.config.seed);
    size_t row = 1;
    for (size_t t = 0; t < rf.trees.size(); ++t) {
        const auto& tree = rf.trees[t];
        const auto& feat = rf.feature_indices[t];
        out(row, 0) = static_cast<double>(tree.nodes.size());
        out(row, 1) = static_cast<double>(feat.size());
        out(row, 2) = tree.max_depth;
        out(row, 3) = ml_tree_criterion_code(tree.criterion);
        ++row;
        for (size_t j = 0; j < feat.size(); ++j) {
            out(row, j) = feat[j];
        }
        ++row;
        for (size_t i = 0; i < tree.nodes.size(); ++i) {
            const auto& node = tree.nodes[i];
            out(row, 0) = node.feature;
            out(row, 1) = node.threshold;
            out(row, 2) = node.value;
            out(row, 3) = node.left;
            out(row, 4) = node.right;
            ++row;
        }
    }
    return out;
}

Result<ml::RandomForest> ml_random_forest_from_matrix(const Matrix<double>& model, const char* fn) {
    if (model.rows() < 2 || model.cols() < 5) {
        return std::unexpected(DomainError{fn, "expected RandomForest model matrix"});
    }
    ml::RandomForest rf;
    rf.config.n_trees = static_cast<size_t>(model(0, 0));
    rf.config.max_depth = static_cast<size_t>(model(0, 1));
    rf.config.feature_subsample_ratio = model(0, 2);
    rf.config.sample_subsample_ratio = model(0, 3);
    rf.config.seed = static_cast<unsigned>(model(0, 4));
    rf.trees.reserve(rf.config.n_trees);
    rf.feature_indices.reserve(rf.config.n_trees);
    size_t row = 1;
    for (size_t t = 0; t < rf.config.n_trees; ++t) {
        if (row >= model.rows()) {
            return std::unexpected(DomainError{fn, "invalid RandomForest model layout"});
        }
        const size_t n_nodes = static_cast<size_t>(model(row, 0));
        const size_t n_feat = static_cast<size_t>(model(row, 1));
        const int max_depth = static_cast<int>(model(row, 2));
        const std::string criterion = ml_tree_criterion_from_code(model(row, 3));
        ++row;
        if (row >= model.rows()) {
            return std::unexpected(DomainError{fn, "invalid RandomForest model layout"});
        }
        std::vector<int> feat(n_feat);
        for (size_t j = 0; j < n_feat; ++j) {
            feat[j] = static_cast<int>(model(row, j));
        }
        ++row;
        if (row + n_nodes > model.rows()) {
            return std::unexpected(DomainError{fn, "invalid RandomForest model layout"});
        }
        ml::DecisionTree tree(max_depth, criterion);
        tree.nodes.resize(n_nodes);
        for (size_t i = 0; i < n_nodes; ++i) {
            auto& node = tree.nodes[i];
            node.feature = static_cast<int>(model(row, 0));
            node.threshold = model(row, 1);
            node.value = model(row, 2);
            node.left = static_cast<int>(model(row, 3));
            node.right = static_cast<int>(model(row, 4));
            ++row;
        }
        rf.trees.push_back(std::move(tree));
        rf.feature_indices.push_back(std::move(feat));
    }
    if (row != model.rows()) {
        return std::unexpected(DomainError{fn, "invalid RandomForest model layout"});
    }
    return rf;
}

Matrix<double> ml_adaboost_to_matrix(const ml::AdaBoost& ab) {
    size_t total_rows = 1;
    for (const auto& est : ab.estimators) {
        total_rows += 1 + est.nodes.size();
    }
    Matrix<double> out(total_rows, 5);
    out(0, 0) = static_cast<double>(ab.config.n_estimators);
    out(0, 1) = static_cast<double>(ab.config.max_depth);
    out(0, 2) = static_cast<double>(ab.config.seed);
    out(0, 3) = static_cast<double>(ab.estimators.size());
    size_t row = 1;
    for (size_t m = 0; m < ab.estimators.size(); ++m) {
        const auto& tree = ab.estimators[m];
        out(row, 0) = ab.estimator_weights[m];
        out(row, 1) = static_cast<double>(tree.nodes.size());
        out(row, 2) = tree.max_depth;
        out(row, 3) = ml_tree_criterion_code(tree.criterion);
        ++row;
        for (size_t i = 0; i < tree.nodes.size(); ++i) {
            const auto& node = tree.nodes[i];
            out(row, 0) = node.feature;
            out(row, 1) = node.threshold;
            out(row, 2) = node.value;
            out(row, 3) = node.left;
            out(row, 4) = node.right;
            ++row;
        }
    }
    return out;
}

Result<ml::AdaBoost> ml_adaboost_from_matrix(const Matrix<double>& model, const char* fn) {
    if (model.rows() < 2 || model.cols() < 5) {
        return std::unexpected(DomainError{fn, "expected AdaBoost model matrix"});
    }
    ml::AdaBoost ab;
    ab.config.n_estimators = static_cast<size_t>(model(0, 0));
    ab.config.max_depth = static_cast<size_t>(model(0, 1));
    ab.config.seed = static_cast<unsigned>(model(0, 2));
    const size_t n_est = static_cast<size_t>(model(0, 3));
    ab.estimators.reserve(n_est);
    ab.estimator_weights.reserve(n_est);
    size_t row = 1;
    for (size_t m = 0; m < n_est; ++m) {
        if (row >= model.rows()) {
            return std::unexpected(DomainError{fn, "invalid AdaBoost model layout"});
        }
        const double weight = model(row, 0);
        const size_t n_nodes = static_cast<size_t>(model(row, 1));
        const int max_depth = static_cast<int>(model(row, 2));
        const std::string criterion = ml_tree_criterion_from_code(model(row, 3));
        ++row;
        if (row + n_nodes > model.rows()) {
            return std::unexpected(DomainError{fn, "invalid AdaBoost model layout"});
        }
        ml::DecisionTree tree(max_depth, criterion);
        tree.nodes.resize(n_nodes);
        for (size_t i = 0; i < n_nodes; ++i) {
            auto& node = tree.nodes[i];
            node.feature = static_cast<int>(model(row, 0));
            node.threshold = model(row, 1);
            node.value = model(row, 2);
            node.left = static_cast<int>(model(row, 3));
            node.right = static_cast<int>(model(row, 4));
            ++row;
        }
        ab.estimators.push_back(std::move(tree));
        ab.estimator_weights.push_back(weight);
    }
    if (row != model.rows()) {
        return std::unexpected(DomainError{fn, "invalid AdaBoost model layout"});
    }
    return ab;
}

Matrix<double> ml_gradient_boosting_to_matrix(const ml::GradientBoosting& gb) {
    size_t total_rows = 1;
    for (const auto& tree : gb.trees) {
        total_rows += 1 + tree.nodes.size();
    }
    Matrix<double> out(total_rows, 5);
    out(0, 0) = static_cast<double>(gb.trees.size());
    out(0, 1) = static_cast<double>(gb.config.max_depth);
    out(0, 2) = gb.config.learning_rate;
    out(0, 3) = static_cast<double>(gb.config.seed);
    out(0, 4) = gb.init_prediction;
    size_t row = 1;
    for (const auto& tree : gb.trees) {
        out(row, 0) = static_cast<double>(tree.nodes.size());
        out(row, 1) = tree.max_depth;
        out(row, 2) = ml_tree_criterion_code(tree.criterion);
        ++row;
        for (size_t i = 0; i < tree.nodes.size(); ++i) {
            const auto& node = tree.nodes[i];
            out(row, 0) = node.feature;
            out(row, 1) = node.threshold;
            out(row, 2) = node.value;
            out(row, 3) = node.left;
            out(row, 4) = node.right;
            ++row;
        }
    }
    return out;
}

Result<ml::GradientBoosting> ml_gradient_boosting_from_matrix(const Matrix<double>& model,
                                                               const char* fn) {
    if (model.rows() < 2 || model.cols() < 5) {
        return std::unexpected(DomainError{fn, "expected GradientBoosting model matrix"});
    }
    ml::GradientBoosting gb;
    gb.config.n_trees = static_cast<size_t>(model(0, 0));
    gb.config.max_depth = static_cast<size_t>(model(0, 1));
    gb.config.learning_rate = model(0, 2);
    gb.config.seed = static_cast<unsigned>(model(0, 3));
    gb.init_prediction = model(0, 4);
    const size_t n_trees = static_cast<size_t>(model(0, 0));
    gb.trees.reserve(n_trees);
    size_t row = 1;
    for (size_t t = 0; t < n_trees; ++t) {
        if (row >= model.rows()) {
            return std::unexpected(DomainError{fn, "invalid GradientBoosting model layout"});
        }
        const size_t n_nodes = static_cast<size_t>(model(row, 0));
        const int max_depth = static_cast<int>(model(row, 1));
        const std::string criterion = ml_tree_criterion_from_code(model(row, 2));
        ++row;
        if (row + n_nodes > model.rows()) {
            return std::unexpected(DomainError{fn, "invalid GradientBoosting model layout"});
        }
        ml::DecisionTree tree(max_depth, criterion);
        tree.nodes.resize(n_nodes);
        for (size_t i = 0; i < n_nodes; ++i) {
            auto& node = tree.nodes[i];
            node.feature = static_cast<int>(model(row, 0));
            node.threshold = model(row, 1);
            node.value = model(row, 2);
            node.left = static_cast<int>(model(row, 3));
            node.right = static_cast<int>(model(row, 4));
            ++row;
        }
        gb.trees.push_back(std::move(tree));
    }
    if (row != model.rows()) {
        return std::unexpected(DomainError{fn, "invalid GradientBoosting model layout"});
    }
    return gb;
}

Result<Matrix<double>> eval_ml_decision_tree_fit(const Matrix<double>& X_m, const Matrix<double>& y_m,
                                                 int max_depth) {
    auto X = matrix_to_ml_mat(X_m, "ml_decision_tree_fit");
    if (!X) {
        return std::unexpected(X.error());
    }
    auto y = matrix_to_ml_vec(y_m, "ml_decision_tree_fit");
    if (!y) {
        return std::unexpected(y.error());
    }
    if (y->size() != X->size()) {
        return std::unexpected(DimensionMismatch{y->size(), X->size()});
    }
    ml::DecisionTree tree(max_depth);
    tree.fit(*X, *y);
    return ml_decision_tree_to_matrix(tree);
}

Result<Matrix<double>> eval_ml_decision_tree_predict(const Matrix<double>& X_m,
                                                    const Matrix<double>& model_m) {
    auto X = matrix_to_ml_mat(X_m, "ml_decision_tree_predict");
    if (!X) {
        return std::unexpected(X.error());
    }
    auto tree = ml_decision_tree_from_matrix(model_m, "ml_decision_tree_predict");
    if (!tree) {
        return std::unexpected(tree.error());
    }
    return vector_to_column(tree->predict(*X));
}

Result<Matrix<double>> eval_ml_random_forest_fit(const Matrix<double>& X_m, const Matrix<double>& y_m,
                                                 size_t n_trees, size_t max_depth) {
    auto X = matrix_to_ml_mat(X_m, "ml_random_forest_fit");
    if (!X) {
        return std::unexpected(X.error());
    }
    auto y = matrix_to_ml_vec(y_m, "ml_random_forest_fit");
    if (!y) {
        return std::unexpected(y.error());
    }
    if (y->size() != X->size()) {
        return std::unexpected(DimensionMismatch{y->size(), X->size()});
    }
    ml::RandomForest rf;
    rf.config.n_trees = n_trees;
    rf.config.max_depth = max_depth;
    rf.config.seed = 42;
    rf.fit(*X, *y);
    return ml_random_forest_to_matrix(rf);
}

Result<Matrix<double>> eval_ml_random_forest_predict(const Matrix<double>& X_m,
                                                     const Matrix<double>& model_m) {
    auto X = matrix_to_ml_mat(X_m, "ml_random_forest_predict");
    if (!X) {
        return std::unexpected(X.error());
    }
    auto rf = ml_random_forest_from_matrix(model_m, "ml_random_forest_predict");
    if (!rf) {
        return std::unexpected(rf.error());
    }
    return vector_to_column(rf->predict(*X));
}

Result<Matrix<double>> eval_ml_adaboost_fit(const Matrix<double>& X_m, const Matrix<double>& y_m,
                                            size_t n_estimators, size_t max_depth) {
    auto X = matrix_to_ml_mat(X_m, "ml_adaboost_fit");
    if (!X) {
        return std::unexpected(X.error());
    }
    auto y = matrix_to_ml_vec(y_m, "ml_adaboost_fit");
    if (!y) {
        return std::unexpected(y.error());
    }
    if (y->size() != X->size()) {
        return std::unexpected(DimensionMismatch{y->size(), X->size()});
    }
    ml::AdaBoost ab;
    ab.config.n_estimators = n_estimators;
    ab.config.max_depth = max_depth;
    ab.config.seed = 42;
    ab.fit(*X, *y);
    return ml_adaboost_to_matrix(ab);
}

Result<Matrix<double>> eval_ml_adaboost_predict(const Matrix<double>& X_m,
                                                const Matrix<double>& model_m) {
    auto X = matrix_to_ml_mat(X_m, "ml_adaboost_predict");
    if (!X) {
        return std::unexpected(X.error());
    }
    auto ab = ml_adaboost_from_matrix(model_m, "ml_adaboost_predict");
    if (!ab) {
        return std::unexpected(ab.error());
    }
    return vector_to_column(ab->predict(*X));
}

Result<Matrix<double>> eval_ml_gradient_boosting_fit(const Matrix<double>& X_m,
                                                     const Matrix<double>& y_m,
                                                     size_t n_estimators, double learning_rate,
                                                     size_t max_depth) {
    auto X = matrix_to_ml_mat(X_m, "ml_gradient_boosting_fit");
    if (!X) {
        return std::unexpected(X.error());
    }
    auto y = matrix_to_ml_vec(y_m, "ml_gradient_boosting_fit");
    if (!y) {
        return std::unexpected(y.error());
    }
    if (y->size() != X->size()) {
        return std::unexpected(DimensionMismatch{y->size(), X->size()});
    }
    ml::GradientBoosting gb;
    gb.config.n_trees = n_estimators;
    gb.config.learning_rate = learning_rate;
    gb.config.max_depth = max_depth;
    gb.config.seed = 42;
    gb.fit(*X, *y);
    return ml_gradient_boosting_to_matrix(gb);
}

Result<Matrix<double>> eval_ml_gradient_boosting_predict(const Matrix<double>& X_m,
                                                        const Matrix<double>& model_m) {
    auto X = matrix_to_ml_mat(X_m, "ml_gradient_boosting_predict");
    if (!X) {
        return std::unexpected(X.error());
    }
    auto gb = ml_gradient_boosting_from_matrix(model_m, "ml_gradient_boosting_predict");
    if (!gb) {
        return std::unexpected(gb.error());
    }
    return vector_to_column(gb->predict(*X));
}

Matrix<double> grid3d_to_matrix(const std::vector<std::vector<std::vector<double>>>& grid) {
    if (grid.empty() || grid[0].empty()) {
        return Matrix<double>(0, 0);
    }
    const std::size_t nz = grid.size();
    const std::size_t ny = grid[0].size();
    const std::size_t nx = grid[0][0].size();
    Matrix<double> out(nz * ny, nx);
    for (std::size_t k = 0; k < nz; ++k) {
        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t i = 0; i < nx; ++i) {
                out(k * ny + j, i) = grid[k][j][i];
            }
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

Result<std::vector<geo::Point3D>> matrix_to_points3d(const Matrix<double>& m, const char* fn) {
    if (m.cols() != 3) {
        return std::unexpected(DomainError{fn, "expected Nx3 point matrix"});
    }
    std::vector<geo::Point3D> points;
    points.reserve(m.rows());
    for (size_t i = 0; i < m.rows(); ++i) {
        points.push_back({m(i, 0), m(i, 1), m(i, 2)});
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

Matrix<double> col_matrix_to_matrix(const ColMatrix<double>& m);
Result<ColMatrix<double>> sparse_vector_to_col_column(const Matrix<double>& m, size_t expected_len,
                                                      const char* fn);

Result<std::vector<int>> matrix_to_int_coeff_vector(const Matrix<double>& m, const char* fn);

Result<SymExpr> parse_sym_quoted_expr(const std::string& quoted_arg, const char* fn);

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

Result<tensorops::Tensor> matrix_to_tensor_shaped(const Matrix<double>& m,
                                                  const std::vector<int>& shape,
                                                  const char* fn) {
    if (shape.empty()) {
        return std::unexpected(DomainError{fn, "expected non-empty tensor shape"});
    }
    long numel = 1;
    for (int dim : shape) {
        if (dim < 1) {
            return std::unexpected(DomainError{fn, "expected positive tensor shape dimensions"});
        }
        numel *= dim;
    }
    if (static_cast<size_t>(numel) != m.rows() * m.cols()) {
        return std::unexpected(
            DomainError{fn, "matrix element count must match tensor shape product"});
    }
    std::vector<double> data;
    data.reserve(static_cast<size_t>(numel));
    for (size_t i = 0; i < m.rows(); ++i) {
        for (size_t j = 0; j < m.cols(); ++j) {
            data.push_back(m(i, j));
        }
    }
    return tensorops::Tensor(shape, std::move(data));
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

Result<double> eval_finance_historical_var(const Matrix<double>& returns_m, double confidence) {
    auto returns = matrix_to_coeff_vector(returns_m, "finance_historical_var");
    if (!returns) {
        return std::unexpected(returns.error());
    }
    if (returns->empty()) {
        return std::unexpected(
            DomainError{"finance_historical_var", "expected non-empty return vector"});
    }
    return finance::historical_var(*returns, confidence);
}

Result<double> eval_finance_historical_cvar(const Matrix<double>& returns_m, double confidence) {
    auto returns = matrix_to_coeff_vector(returns_m, "finance_historical_cvar");
    if (!returns) {
        return std::unexpected(returns.error());
    }
    if (returns->empty()) {
        return std::unexpected(
            DomainError{"finance_historical_cvar", "expected non-empty return vector"});
    }
    return finance::historical_cvar(*returns, confidence);
}

Result<double> eval_finance_treynor(const Matrix<double>& returns_m, double risk_free,
                                    double beta) {
    auto returns = matrix_to_coeff_vector(returns_m, "finance_treynor");
    if (!returns) {
        return std::unexpected(returns.error());
    }
    if (returns->empty()) {
        return std::unexpected(
            DomainError{"finance_treynor", "expected non-empty return vector"});
    }
    return finance::treynor_ratio(*returns, risk_free, beta);
}

Result<double> eval_finance_information_ratio(const Matrix<double>& returns_m,
                                              const Matrix<double>& benchmark_m) {
    auto returns = matrix_to_coeff_vector(returns_m, "finance_information_ratio");
    if (!returns) {
        return std::unexpected(returns.error());
    }
    auto benchmark = matrix_to_coeff_vector(benchmark_m, "finance_information_ratio");
    if (!benchmark) {
        return std::unexpected(benchmark.error());
    }
    if (returns->empty() || benchmark->empty()) {
        return std::unexpected(
            DomainError{"finance_information_ratio", "expected non-empty return vectors"});
    }
    if (returns->size() != benchmark->size()) {
        return std::unexpected(
            DomainError{"finance_information_ratio", "vector length mismatch"});
    }
    return finance::information_ratio(*returns, *benchmark);
}

Result<double> eval_finance_merton_distance_to_default(double asset_value, double asset_volatility,
                                                      double debt_face_value, double risk_free_rate,
                                                      double time_horizon) {
    const finance::MertonResult result = finance::merton_distance_to_default(
        asset_value, asset_volatility, debt_face_value, risk_free_rate, time_horizon);
    if (!result.converged) {
        return std::unexpected(DomainError{
            "finance_merton_distance_to_default",
            "expected positive asset value, debt face value, asset volatility, and time horizon"});
    }
    return result.distance_to_default;
}

Result<Matrix<double>> eval_finance_merton_implied_asset_params(double equity_value,
                                                                double equity_volatility,
                                                                double debt_face_value,
                                                                double risk_free_rate,
                                                                double time_horizon) {
    const finance::MertonResult result = finance::merton_implied_asset_params(
        equity_value, equity_volatility, debt_face_value, risk_free_rate, time_horizon);
    Matrix<double> out(1, 6);
    out(0, 0) = result.distance_to_default;
    out(0, 1) = result.probability_of_default;
    out(0, 2) = result.implied_asset_value;
    out(0, 3) = result.implied_asset_volatility;
    out(0, 4) = result.converged ? 1.0 : 0.0;
    out(0, 5) = static_cast<double>(result.iterations);
    return out;
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

Result<Matrix<double>> eval_finance_min_variance_portfolio(const Matrix<double>& cov_m) {
    auto cov = matrix_to_row_major_flat(cov_m, "finance_min_variance_portfolio");
    if (!cov) {
        return std::unexpected(cov.error());
    }
    const int n = static_cast<int>(cov_m.rows());
    if (cov_m.rows() != cov_m.cols()) {
        return std::unexpected(
            DomainError{"finance_min_variance_portfolio", "expected square covariance matrix"});
    }
    auto w = finance::min_variance_portfolio(*cov, n);
    if (!w) {
        return std::unexpected(w.error());
    }
    return vector_to_column(*w);
}

Result<Matrix<double>> eval_finance_max_sharpe_portfolio(const Matrix<double>& cov_m,
                                                         const Matrix<double>& mu_m,
                                                         double risk_free) {
    auto cov = matrix_to_row_major_flat(cov_m, "finance_max_sharpe_portfolio");
    if (!cov) {
        return std::unexpected(cov.error());
    }
    const int n = static_cast<int>(cov_m.rows());
    if (cov_m.rows() != cov_m.cols()) {
        return std::unexpected(
            DomainError{"finance_max_sharpe_portfolio", "expected square covariance matrix"});
    }
    auto mu = matrix_to_coeff_vector(mu_m, "finance_max_sharpe_portfolio");
    if (!mu) {
        return std::unexpected(mu.error());
    }
    if (mu->size() != static_cast<size_t>(n)) {
        return std::unexpected(
            DomainError{"finance_max_sharpe_portfolio", "expected mu length to match covariance size"});
    }
    auto w = finance::max_sharpe_portfolio(*cov, *mu, risk_free, n);
    if (!w) {
        return std::unexpected(w.error());
    }
    return vector_to_column(*w);
}

Result<Matrix<double>> eval_finance_efficient_frontier(const Matrix<double>& cov_m,
                                                       const Matrix<double>& mu_m,
                                                       double target_return) {
    auto cov = matrix_to_row_major_flat(cov_m, "finance_efficient_frontier");
    if (!cov) {
        return std::unexpected(cov.error());
    }
    const int n = static_cast<int>(cov_m.rows());
    if (cov_m.rows() != cov_m.cols()) {
        return std::unexpected(
            DomainError{"finance_efficient_frontier", "expected square covariance matrix"});
    }
    auto mu = matrix_to_coeff_vector(mu_m, "finance_efficient_frontier");
    if (!mu) {
        return std::unexpected(mu.error());
    }
    if (mu->size() != static_cast<size_t>(n)) {
        return std::unexpected(DomainError{
            "finance_efficient_frontier", "expected mu length to match covariance size"});
    }
    auto w = finance::efficient_frontier_portfolio(*cov, *mu, target_return, n);
    if (!w) {
        return std::unexpected(w.error());
    }
    return vector_to_column(*w);
}

Result<Matrix<double>> eval_finance_max_sharpe(const Matrix<double>& cov_m,
                                               const Matrix<double>& mu_m, double risk_free) {
    auto cov = matrix_to_row_major_flat(cov_m, "finance_max_sharpe");
    if (!cov) {
        return std::unexpected(cov.error());
    }
    const int n = static_cast<int>(cov_m.rows());
    if (cov_m.rows() != cov_m.cols()) {
        return std::unexpected(
            DomainError{"finance_max_sharpe", "expected square covariance matrix"});
    }
    auto mu = matrix_to_coeff_vector(mu_m, "finance_max_sharpe");
    if (!mu) {
        return std::unexpected(mu.error());
    }
    if (mu->size() != static_cast<size_t>(n)) {
        return std::unexpected(
            DomainError{"finance_max_sharpe", "expected mu length to match covariance size"});
    }
    auto w = finance::max_sharpe_portfolio(*cov, *mu, risk_free, n);
    if (!w) {
        return std::unexpected(w.error());
    }
    return vector_to_column(*w);
}

Result<Matrix<double>> eval_finance_bl_implied_returns(const Matrix<double>& cov_m,
                                                       const Matrix<double>& w_mkt_m,
                                                       double delta) {
    auto cov = matrix_to_row_major_flat(cov_m, "finance_bl_implied_returns");
    if (!cov) {
        return std::unexpected(cov.error());
    }
    const int n = static_cast<int>(cov_m.rows());
    if (cov_m.rows() != cov_m.cols()) {
        return std::unexpected(
            DomainError{"finance_bl_implied_returns", "expected square covariance matrix"});
    }
    auto w_mkt = matrix_to_coeff_vector(w_mkt_m, "finance_bl_implied_returns");
    if (!w_mkt) {
        return std::unexpected(w_mkt.error());
    }
    if (w_mkt->size() != static_cast<size_t>(n)) {
        return std::unexpected(DomainError{
            "finance_bl_implied_returns", "expected w_mkt length to match covariance size"});
    }
    auto pi = finance::bl_implied_returns(*cov, *w_mkt, delta, n);
    if (!pi) {
        return std::unexpected(pi.error());
    }
    return vector_to_column(*pi);
}

Result<Matrix<double>> eval_finance_bl_posterior_returns(
    const Matrix<double>& pi_m, const Matrix<double>& cov_m, const Matrix<double>& P_m,
    const Matrix<double>& Q_m, double tau) {
    auto pi = matrix_to_coeff_vector(pi_m, "finance_bl_posterior_returns");
    if (!pi) {
        return std::unexpected(pi.error());
    }
    auto cov = matrix_to_row_major_flat(cov_m, "finance_bl_posterior_returns");
    if (!cov) {
        return std::unexpected(cov.error());
    }
    const int n = static_cast<int>(cov_m.rows());
    if (cov_m.rows() != cov_m.cols()) {
        return std::unexpected(
            DomainError{"finance_bl_posterior_returns", "expected square covariance matrix"});
    }
    if (pi->size() != static_cast<size_t>(n)) {
        return std::unexpected(DomainError{
            "finance_bl_posterior_returns", "expected pi length to match covariance size"});
    }
    const int k = static_cast<int>(P_m.rows());
    if (P_m.cols() != static_cast<size_t>(n)) {
        return std::unexpected(DomainError{
            "finance_bl_posterior_returns",
            "expected P to have n columns matching covariance size"});
    }
    // P is kÃƒâ€”n (views Ãƒâ€” assets), not square Ã¢â‚¬â€� flatten row-major without the
    // square-matrix helper used for covariance.
    std::vector<double> P(static_cast<size_t>(k) * static_cast<size_t>(n));
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < n; ++j) {
            P[static_cast<size_t>(i) * static_cast<size_t>(n) + static_cast<size_t>(j)] =
                P_m(static_cast<size_t>(i), static_cast<size_t>(j));
        }
    }
    auto Q = matrix_to_coeff_vector(Q_m, "finance_bl_posterior_returns");
    if (!Q) {
        return std::unexpected(Q.error());
    }
    if (Q->size() != static_cast<size_t>(k)) {
        return std::unexpected(DomainError{
            "finance_bl_posterior_returns",
            "expected Q length to match number of views (P rows)"});
    }
    auto post = finance::bl_posterior_returns_default_omega(*pi, *cov, tau, P, *Q, n, k);
    if (!post) {
        return std::unexpected(post.error());
    }
    return vector_to_column(*post);
}

Result<Matrix<double>> eval_finance_bl_posterior_returns_default_omega(
    const Matrix<double>& pi_m, const Matrix<double>& cov_m, const Matrix<double>& P_m,
    const Matrix<double>& Q_m, double tau) {
    auto pi = matrix_to_coeff_vector(pi_m, "finance_bl_posterior_returns_default_omega");
    if (!pi) {
        return std::unexpected(pi.error());
    }
    auto cov = matrix_to_row_major_flat(cov_m, "finance_bl_posterior_returns_default_omega");
    if (!cov) {
        return std::unexpected(cov.error());
    }
    const int n = static_cast<int>(cov_m.rows());
    if (cov_m.rows() != cov_m.cols()) {
        return std::unexpected(DomainError{
            "finance_bl_posterior_returns_default_omega", "expected square covariance matrix"});
    }
    if (pi->size() != static_cast<size_t>(n)) {
        return std::unexpected(DomainError{
            "finance_bl_posterior_returns_default_omega",
            "expected pi length to match covariance size"});
    }
    const int k = static_cast<int>(P_m.rows());
    if (P_m.cols() != static_cast<size_t>(n)) {
        return std::unexpected(DomainError{
            "finance_bl_posterior_returns_default_omega",
            "expected P to have n columns matching covariance size"});
    }
    std::vector<double> P(static_cast<size_t>(k) * static_cast<size_t>(n));
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < n; ++j) {
            P[static_cast<size_t>(i) * static_cast<size_t>(n) + static_cast<size_t>(j)] =
                P_m(static_cast<size_t>(i), static_cast<size_t>(j));
        }
    }
    auto Q = matrix_to_coeff_vector(Q_m, "finance_bl_posterior_returns_default_omega");
    if (!Q) {
        return std::unexpected(Q.error());
    }
    if (Q->size() != static_cast<size_t>(k)) {
        return std::unexpected(DomainError{
            "finance_bl_posterior_returns_default_omega",
            "expected Q length to match number of views (P rows)"});
    }
    auto post = finance::bl_posterior_returns_default_omega(*pi, *cov, tau, P, *Q, n, k);
    if (!post) {
        return std::unexpected(post.error());
    }
    return vector_to_column(*post);
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

Result<double> eval_info_blahut_arimoto(const Matrix<double>& W_m) {
    if (W_m.rows() == 0 || W_m.cols() == 0) {
        return std::unexpected(
            DomainError{"info_blahut_arimoto", "expected non-empty channel matrix"});
    }
    std::vector<std::vector<double>> W(W_m.rows(), std::vector<double>(W_m.cols()));
    for (size_t i = 0; i < W_m.rows(); ++i) {
        for (size_t j = 0; j < W_m.cols(); ++j) {
            W[i][j] = W_m(i, j);
        }
    }
    return info::blahut_arimoto(W);
}

Result<double> eval_info_channel_capacity(const Matrix<double>& W_m) {
    if (W_m.rows() == 0 || W_m.cols() == 0) {
        return std::unexpected(
            DomainError{"info_channel_capacity", "expected non-empty channel matrix"});
    }
    std::vector<std::vector<double>> W(W_m.rows(), std::vector<double>(W_m.cols()));
    for (size_t i = 0; i < W_m.rows(); ++i) {
        for (size_t j = 0; j < W_m.cols(); ++j) {
            W[i][j] = W_m(i, j);
        }
    }
    return info::channel_capacity(W).capacity;
}

Result<Matrix<double>> eval_info_channel_capacity_input(const Matrix<double>& W_m) {
    if (W_m.rows() == 0 || W_m.cols() == 0) {
        return std::unexpected(
            DomainError{"info_channel_capacity_input", "expected non-empty channel matrix"});
    }
    std::vector<std::vector<double>> W(W_m.rows(), std::vector<double>(W_m.cols()));
    for (size_t i = 0; i < W_m.rows(); ++i) {
        for (size_t j = 0; j < W_m.cols(); ++j) {
            W[i][j] = W_m(i, j);
        }
    }
    return vector_to_column(info::channel_capacity(W).input_distribution);
}

Result<double> eval_info_normalized_entropy(const Matrix<double>& p_m) {
    auto p = matrix_to_coeff_vector(p_m, "info_normalized_entropy");
    if (!p) {
        return std::unexpected(p.error());
    }
    if (p->empty()) {
        return std::unexpected(
            DomainError{"info_normalized_entropy", "expected non-empty probability vector"});
    }
    return info::normalized_entropy(*p);
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

Result<double> eval_info_permutation_entropy(const Matrix<double>& x_m, int order, int delay) {
    auto x = matrix_to_coeff_vector(x_m, "info_permutation_entropy");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (x->empty()) {
        return std::unexpected(DomainError{
            "info_permutation_entropy", "expected non-empty time series vector"});
    }
    return info::permutation_entropy(*x, order, delay, true);
}

Result<double> eval_info_transfer_entropy(const Matrix<double>& x_m, const Matrix<double>& y_m,
                                          int bins, int lag) {
    auto x = matrix_to_coeff_vector(x_m, "info_transfer_entropy");
    if (!x) {
        return std::unexpected(x.error());
    }
    auto y = matrix_to_coeff_vector(y_m, "info_transfer_entropy");
    if (!y) {
        return std::unexpected(y.error());
    }
    if (x->empty() || y->empty()) {
        return std::unexpected(DomainError{
            "info_transfer_entropy", "expected non-empty time series vectors"});
    }
    if (x->size() != y->size()) {
        return std::unexpected(DomainError{"info_transfer_entropy", "vector length mismatch"});
    }
    return info::transfer_entropy(*x, *y, bins, lag);
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
    if (t.ndim() == 2) {
        const size_t rows = static_cast<size_t>(t.shape[0]);
        const size_t cols = static_cast<size_t>(t.shape[1]);
        Matrix<double> out(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                out(i, j) = t.at({static_cast<int>(i), static_cast<int>(j)});
            }
        }
        return out;
    }
    const size_t rows = t.ndim() >= 1 ? static_cast<size_t>(t.shape[0]) : 1u;
    const size_t cols =
        rows == 0 ? 0u : static_cast<size_t>(t.numel() / static_cast<long>(rows));
    Matrix<double> out(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            const long flat_idx =
                static_cast<long>(i) * static_cast<long>(cols) + static_cast<long>(j);
            std::vector<int> idx(static_cast<size_t>(t.ndim()), 0);
            long tmp = flat_idx;
            for (int d = t.ndim() - 1; d >= 0; --d) {
                idx[static_cast<size_t>(d)] = static_cast<int>(tmp % t.shape[d]);
                tmp /= t.shape[d];
            }
            out(i, j) = t.at(idx);
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

Result<Matrix<double>> eval_geo_bezier_eval(const Matrix<double>& ctrl_m, double t) {
    auto ctrl = matrix_to_points2d(ctrl_m, "geo_bezier_eval");
    if (!ctrl) {
        return std::unexpected(ctrl.error());
    }
    if (ctrl->size() < 3) {
        return std::unexpected(
            DomainError{"geo_bezier_eval", "expected at least 3 control points"});
    }
    const geo::Point2D p = geo::bezier_eval(*ctrl, t);
    Matrix<double> out(1, 2);
    out(0, 0) = p.x;
    out(0, 1) = p.y;
    return out;
}

Result<Matrix<double>> eval_geo_bezier_deriv(const Matrix<double>& ctrl_m, double t) {
    auto ctrl = matrix_to_points2d(ctrl_m, "geo_bezier_deriv");
    if (!ctrl) {
        return std::unexpected(ctrl.error());
    }
    if (ctrl->size() < 3) {
        return std::unexpected(
            DomainError{"geo_bezier_deriv", "expected at least 3 control points"});
    }
    const geo::Vec2D d = geo::bezier_deriv(*ctrl, t);
    Matrix<double> out(1, 2);
    out(0, 0) = d.x;
    out(0, 1) = d.y;
    return out;
}

Result<Matrix<double>> eval_geo_catmull_rom(const Matrix<double>& ctrl_m, double t) {
    auto ctrl = matrix_to_points2d(ctrl_m, "geo_catmull_rom");
    if (!ctrl) {
        return std::unexpected(ctrl.error());
    }
    if (ctrl->size() < 2) {
        return std::unexpected(
            DomainError{"geo_catmull_rom", "expected at least 2 control points"});
    }
    const geo::Point2D p = geo::catmull_rom(*ctrl, t);
    Matrix<double> out(1, 2);
    out(0, 0) = p.x;
    out(0, 1) = p.y;
    return out;
}

Result<Matrix<double>> eval_geo_bspline_eval(const Matrix<double>& ctrl_m,
                                             const Matrix<double>& knots_m, double degree_d,
                                             double t) {
    auto ctrl = matrix_to_points2d(ctrl_m, "geo_bspline_eval");
    if (!ctrl) {
        return std::unexpected(ctrl.error());
    }
    if (ctrl->empty()) {
        return std::unexpected(
            DomainError{"geo_bspline_eval", "expected non-empty Nx2 control polygon"});
    }
    auto knots = matrix_to_coeff_vector(knots_m, "geo_bspline_eval");
    if (!knots) {
        return std::unexpected(knots.error());
    }
    const int degree = static_cast<int>(degree_d);
    if (degree < 1 || degree_d != degree) {
        return std::unexpected(
            DomainError{"geo_bspline_eval", "expected positive integer degree"});
    }
    const int n = static_cast<int>(ctrl->size()) - 1;
    const int expected_knots = n + degree + 2;
    if (static_cast<int>(knots->size()) < expected_knots) {
        return std::unexpected(DomainError{
            "geo_bspline_eval", "expected knot vector length >= n+degree+2"});
    }
    const geo::Point2D p = geo::bspline_eval(*ctrl, *knots, degree, t);
    Matrix<double> out(1, 2);
    out(0, 0) = p.x;
    out(0, 1) = p.y;
    return out;
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

Result<Matrix<double>> eval_numthy_factor_exp(int n) {
    if (n < 2) {
        return std::unexpected(
            DomainError{"numthy_factor_exp", "expected integer n >= 2"});
    }
    const auto fe = numthy::factor_exp(static_cast<uint64_t>(n));
    Matrix<double> out(fe.size(), 2);
    for (size_t i = 0; i < fe.size(); ++i) {
        out(i, 0) = static_cast<double>(fe[i].first);
        out(i, 1) = static_cast<double>(fe[i].second);
    }
    return out;
}

Result<Matrix<double>> eval_numthy_farey(int n) {
    if (n < 1) {
        return std::unexpected(
            DomainError{"numthy_farey", "expected positive integer n"});
    }
    const auto fr = numthy::farey(static_cast<uint32_t>(n));
    Matrix<double> out(fr.size(), 2);
    for (size_t i = 0; i < fr.size(); ++i) {
        out(i, 0) = static_cast<double>(fr[i].first);
        out(i, 1) = static_cast<double>(fr[i].second);
    }
    return out;
}

Result<Matrix<double>> eval_numthy_lucas_sequence(int64_t k, int64_t P, int64_t Q) {
    const auto [u, v] = numthy::lucas_sequence(k, P, Q);
    Matrix<double> out(1, 2);
    out(0, 0) = static_cast<double>(u);
    out(0, 1) = static_cast<double>(v);
    return out;
}

Result<double> eval_numthy_multiplicative_order(double a_d, double n_d) {
    if (std::floor(a_d) != a_d || std::floor(n_d) != n_d) {
        return std::unexpected(
            DomainError{"numthy_multiplicative_order", "expected integer arguments"});
    }
    if (a_d < 0.0 || n_d < 0.0) {
        return std::unexpected(
            DomainError{"numthy_multiplicative_order", "expected a >= 0 and n >= 0"});
    }
    auto ord = numthy::multiplicative_order(static_cast<uint64_t>(a_d),
                                             static_cast<uint64_t>(n_d));
    if (!ord) {
        return std::unexpected(ord.error());
    }
    return static_cast<double>(*ord);
}

Matrix<double> codes_to_matrix_col(const std::vector<uint32_t>& codes) {
    Matrix<double> out(codes.size(), 1);
    for (size_t i = 0; i < codes.size(); ++i) {
        out(i, 0) = static_cast<double>(codes[i]);
    }
    return out;
}

Result<Matrix<double>> eval_numthy_stern_brocot(int n) {
    if (n < 0) {
        return std::unexpected(
            DomainError{"numthy_stern_brocot", "expected non-negative integer n"});
    }
    const auto sb = numthy::stern_brocot(static_cast<uint64_t>(n));
    Matrix<double> out(sb.size(), 2);
    for (size_t i = 0; i < sb.size(); ++i) {
        out(i, 0) = static_cast<double>(sb[i].first);
        out(i, 1) = static_cast<double>(sb[i].second);
    }
    return out;
}

Result<Matrix<double>> eval_numthy_pell_solve(int D) {
    if (D < 1) {
        return std::unexpected(
            DomainError{"numthy_pell_solve", "expected positive integer D"});
    }
    auto sol = numthy::pell_solve(static_cast<uint64_t>(D));
    if (!sol) {
        return std::unexpected(sol.error());
    }
    Matrix<double> out(1, 2);
    out(0, 0) = static_cast<double>(sol->first);
    out(0, 1) = static_cast<double>(sol->second);
    return out;
}

Result<Matrix<double>> eval_numthy_cornacchia(uint64_t d, uint64_t p) {
    if (d < 1 || d >= p) {
        return std::unexpected(
            DomainError{"numthy_cornacchia", "expected positive integers d,p with 0 < d < p"});
    }
    auto sol = numthy::cornacchia(d, p);
    if (!sol) {
        return std::unexpected(sol.error());
    }
    Matrix<double> out(1, 2);
    out(0, 0) = static_cast<double>(sol->first);
    out(0, 1) = static_cast<double>(sol->second);
    return out;
}

Result<Matrix<double>> eval_numthy_quadratic_residues(int p) {
    if (p < 3) {
        return std::unexpected(
            DomainError{"numthy_quadratic_residues", "expected odd prime p >= 3"});
    }
    const auto p_u = static_cast<uint64_t>(p);
    if (!numthy::isprime(p_u)) {
        return std::unexpected(
            DomainError{"numthy_quadratic_residues", "expected odd prime p >= 3"});
    }
    const auto qr = numthy::quadratic_residues(p_u);
    Matrix<double> out(qr.size(), 1);
    for (size_t i = 0; i < qr.size(); ++i) {
        out(i, 0) = static_cast<double>(qr[i]);
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

Result<Matrix<double>> eval_arithmetic_encode_vec(const Matrix<double>& m) {
    const compress::ArithmeticResult ar = compress::arithmetic_encode(matrix_to_bytes(m));
    return bytes_to_matrix_col(ar.encoded);
}

Result<Matrix<double>> eval_arithmetic_decode_vec(const Matrix<double>& orig_m,
                                                  const Matrix<double>& /*encoded_m*/) {
    const compress::Bytes bytes = matrix_to_bytes(orig_m);
    const compress::ArithmeticResult ar = compress::arithmetic_encode(bytes);
    return bytes_to_matrix_col(compress::arithmetic_decode(ar));
}

Result<Matrix<double>> eval_ans_encode_vec(const Matrix<double>& m) {
    const compress::AnsResult ar = compress::ans_encode(matrix_to_bytes(m));
    return bytes_to_matrix_col(ar.encoded);
}

Result<Matrix<double>> eval_ans_decode_vec(const Matrix<double>& orig_m,
                                           const Matrix<double>& /*encoded_m*/) {
    const compress::Bytes bytes = matrix_to_bytes(orig_m);
    const compress::AnsResult ar = compress::ans_encode(bytes);
    return bytes_to_matrix_col(compress::ans_decode(ar));
}

Result<std::vector<uint32_t>> matrix_col_to_u32(const Matrix<double>& m, const char* fn) {
    if (m.cols() != 1) {
        return std::unexpected(DomainError{fn, "expected Nx1 integer vector"});
    }
    std::vector<uint32_t> values;
    values.reserve(m.rows());
    for (size_t i = 0; i < m.rows(); ++i) {
        const double v = m(i, 0);
        if (v < 0.0 || std::floor(v) != v) {
            return std::unexpected(DomainError{fn, "values must be non-negative integers"});
        }
        values.push_back(static_cast<uint32_t>(v));
    }
    return values;
}

Result<Matrix<double>> eval_golomb_rice_encode_vec(const Matrix<double>& values_m, int m_bits) {
    auto values = matrix_col_to_u32(values_m, "golomb_rice_encode_vec");
    if (!values) {
        return std::unexpected(values.error());
    }
    return bytes_to_matrix_col(compress::golomb_rice_encode(*values, m_bits));
}

Result<Matrix<double>> eval_golomb_rice_decode_vec(const Matrix<double>& encoded_m, int m_bits,
                                                   size_t count) {
    auto bytes = matrix_col_to_bytes(encoded_m, "golomb_rice_decode_vec");
    if (!bytes) {
        return std::unexpected(bytes.error());
    }
    return codes_to_matrix_col(compress::golomb_rice_decode(*bytes, m_bits, count));
}

Result<Matrix<double>> eval_gria_ca_step(const Matrix<double>& state_m, int rule) {
    if (rule < 0 || rule > 255) {
        return std::unexpected(DomainError{"gria_ca_step", "expected rule in [0,255]"});
    }
    auto state = matrix_col_to_bytes(state_m, "gria_ca_step");
    if (!state) {
        return std::unexpected(state.error());
    }
    return bytes_to_matrix_col(gria::ca::step(*state, static_cast<uint8_t>(rule)));
}

Result<double> eval_gria_langton_lambda(int rule) {
    if (rule < 0 || rule > 255) {
        return std::unexpected(DomainError{"gria_langton_lambda", "expected rule in [0,255]"});
    }
    return gria::ca::langton_lambda(static_cast<uint8_t>(rule));
}

Result<double> eval_gria_alpha_ca(int rule, size_t steps, size_t width) {
    if (rule < 0 || rule > 255) {
        return std::unexpected(DomainError{"gria_alpha_ca", "expected rule in [0,255]"});
    }
    return gria::ca::alpha_ca(static_cast<uint8_t>(rule), steps, width);
}

Result<double> eval_gria_hamming_distance(const Matrix<double>& a_m, const Matrix<double>& b_m) {
    auto a = matrix_col_to_bytes(a_m, "gria_hamming_distance");
    if (!a) {
        return std::unexpected(a.error());
    }
    auto b = matrix_col_to_bytes(b_m, "gria_hamming_distance");
    if (!b) {
        return std::unexpected(b.error());
    }
    return static_cast<double>(gria::ca::hamming_distance(*a, *b));
}

Result<Matrix<double>> eval_gria_divergence_trajectory(const Matrix<double>& a_m,
                                                         const Matrix<double>& b_m, int rule,
                                                         int n_steps) {
    if (rule < 0 || rule > 255) {
        return std::unexpected(
            DomainError{"gria_divergence_trajectory", "expected rule in [0,255]"});
    }
    auto a = matrix_col_to_bytes(a_m, "gria_divergence_trajectory");
    if (!a) {
        return std::unexpected(a.error());
    }
    auto b = matrix_col_to_bytes(b_m, "gria_divergence_trajectory");
    if (!b) {
        return std::unexpected(b.error());
    }
    const auto trajectory =
        gria::ca::divergence_trajectory(*a, *b, static_cast<uint8_t>(rule), n_steps);
    Matrix<double> out(trajectory.size(), 1);
    for (size_t i = 0; i < trajectory.size(); ++i) {
        out(i, 0) = static_cast<double>(trajectory[i]);
    }
    return out;
}

Result<double> eval_gria_settling_time(const Matrix<double>& a_m, const Matrix<double>& b_m,
                                         int rule, int n_steps) {
    if (rule < 0 || rule > 255) {
        return std::unexpected(DomainError{"gria_settling_time", "expected rule in [0,255]"});
    }
    auto a = matrix_col_to_bytes(a_m, "gria_settling_time");
    if (!a) {
        return std::unexpected(a.error());
    }
    auto b = matrix_col_to_bytes(b_m, "gria_settling_time");
    if (!b) {
        return std::unexpected(b.error());
    }
    const auto settled =
        gria::ca::settling_time(*a, *b, static_cast<uint8_t>(rule), n_steps);
    if (!settled) {
        return -1.0;
    }
    return static_cast<double>(*settled);
}

Result<Matrix<double>> eval_wavelet_compress_vec(const Matrix<double>& m,
                                                 double threshold = 0.0) {
    return bytes_to_matrix_col(compress::wavelet_compress(matrix_to_bytes(m), threshold));
}

Result<Matrix<double>> eval_wavelet_decompress_vec(const Matrix<double>& compressed_m) {
    auto bytes = matrix_col_to_bytes(compressed_m, "wavelet_decompress_vec");
    if (!bytes) {
        return std::unexpected(bytes.error());
    }
    return bytes_to_matrix_col(compress::wavelet_decompress(*bytes));
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

Result<Matrix<double>> eval_control_ctrb(const Matrix<double>& A_m,
                                         const Matrix<double>& B_m) {
    constexpr const char* fn = "control_ctrb";
    auto A = matrix_to_square_nested(A_m, fn);
    if (!A) {
        return std::unexpected(A.error());
    }
    auto B = matrix_to_nested(B_m, fn);
    if (!B) {
        return std::unexpected(B.error());
    }
    if (B->size() != A->size()) {
        return std::unexpected(DomainError{fn, "expected B with same row count as A"});
    }
    return nested_to_matrix(control::ctrb(*A, *B));
}

Result<Matrix<double>> eval_control_obsv(const Matrix<double>& A_m,
                                         const Matrix<double>& C_m) {
    constexpr const char* fn = "control_obsv";
    auto A = matrix_to_square_nested(A_m, fn);
    if (!A) {
        return std::unexpected(A.error());
    }
    auto C = matrix_to_nested(C_m, fn);
    if (!C) {
        return std::unexpected(C.error());
    }
    if (C->empty() || C->front().size() != A->size()) {
        return std::unexpected(
            DomainError{fn, "expected C with column count equal to A size"});
    }
    return nested_to_matrix(control::obsv(*A, *C));
}

Result<Matrix<double>> eval_control_ctrb_gram(const Matrix<double>& A_m,
                                              const Matrix<double>& B_m) {
    constexpr const char* fn = "control_ctrb_gram";
    auto A = matrix_to_square_nested(A_m, fn);
    if (!A) {
        return std::unexpected(A.error());
    }
    auto B = matrix_to_nested(B_m, fn);
    if (!B) {
        return std::unexpected(B.error());
    }
    if (B->size() != A->size()) {
        return std::unexpected(DomainError{fn, "expected B with same row count as A"});
    }
    const size_t n = A->size();
    const size_t m = (*B)[0].size();
    std::vector<std::vector<double>> C(1, std::vector<double>(n, 0.0));
    if (!C[0].empty()) {
        C[0][0] = 1.0;
    }
    std::vector<std::vector<double>> D(1, std::vector<double>(m, 0.0));
    auto Wc = control::ctrb_gram(control::ss(*A, *B, C, D));
    if (!Wc) {
        return std::unexpected(Wc.error());
    }
    return nested_to_matrix(*Wc);
}

Result<Matrix<double>> eval_control_obsv_gram(const Matrix<double>& A_m,
                                              const Matrix<double>& C_m) {
    constexpr const char* fn = "control_obsv_gram";
    auto A = matrix_to_square_nested(A_m, fn);
    if (!A) {
        return std::unexpected(A.error());
    }
    auto C = matrix_to_nested(C_m, fn);
    if (!C) {
        return std::unexpected(C.error());
    }
    if (C->empty() || C->front().size() != A->size()) {
        return std::unexpected(
            DomainError{fn, "expected C with column count equal to A size"});
    }
    const size_t n = A->size();
    const size_t p = C->size();
    std::vector<std::vector<double>> B(n, std::vector<double>(1, 0.0));
    if (!B.empty()) {
        B[0][0] = 1.0;
    }
    std::vector<std::vector<double>> D(p, std::vector<double>(1, 0.0));
    auto Wo = control::obsv_gram(control::ss(*A, B, *C, D));
    if (!Wo) {
        return std::unexpected(Wo.error());
    }
    return nested_to_matrix(*Wo);
}

Result<control::KalmanState> kalman_state_from_matrices(const Matrix<double>& x_m,
                                                          const Matrix<double>& P_m,
                                                          const char* fn) {
    auto x = matrix_to_coeff_vector(x_m, fn);
    if (!x) {
        return std::unexpected(x.error());
    }
    auto P = matrix_to_square_nested(P_m, fn);
    if (!P) {
        return std::unexpected(P.error());
    }
    if (P->size() != x->size()) {
        return std::unexpected(DimensionMismatch{x->size(), P->size()});
    }
    return control::KalmanState{*x, *P};
}

Result<Matrix<double>> eval_control_kalman_predict(const Matrix<double>& x_m,
                                                   const Matrix<double>& P_m,
                                                   const Matrix<double>& A_m,
                                                   const Matrix<double>& Q_m) {
    constexpr const char* fn = "control_kalman_predict";
    auto state = kalman_state_from_matrices(x_m, P_m, fn);
    if (!state) {
        return std::unexpected(state.error());
    }
    auto A = matrix_to_square_nested(A_m, fn);
    if (!A) {
        return std::unexpected(A.error());
    }
    auto Q = matrix_to_square_nested(Q_m, fn);
    if (!Q) {
        return std::unexpected(Q.error());
    }
    const auto out = control::kalman_predict(*state, *A, *Q);
    return vector_to_column(out.x);
}

Result<Matrix<double>> eval_control_kalman_predict_cov(const Matrix<double>& x_m,
                                                         const Matrix<double>& P_m,
                                                         const Matrix<double>& A_m,
                                                         const Matrix<double>& Q_m) {
    constexpr const char* fn = "control_kalman_predict_cov";
    auto state = kalman_state_from_matrices(x_m, P_m, fn);
    if (!state) {
        return std::unexpected(state.error());
    }
    auto A = matrix_to_square_nested(A_m, fn);
    if (!A) {
        return std::unexpected(A.error());
    }
    auto Q = matrix_to_square_nested(Q_m, fn);
    if (!Q) {
        return std::unexpected(Q.error());
    }
    const auto out = control::kalman_predict(*state, *A, *Q);
    return nested_to_matrix(out.P);
}

Result<Matrix<double>> eval_control_kalman_update(const Matrix<double>& x_m,
                                                  const Matrix<double>& P_m,
                                                  const Matrix<double>& z_m,
                                                  const Matrix<double>& H_m,
                                                  const Matrix<double>& R_m) {
    constexpr const char* fn = "control_kalman_update";
    auto state = kalman_state_from_matrices(x_m, P_m, fn);
    if (!state) {
        return std::unexpected(state.error());
    }
    auto z = matrix_to_coeff_vector(z_m, fn);
    if (!z) {
        return std::unexpected(z.error());
    }
    auto H = matrix_to_nested(H_m, fn);
    if (!H) {
        return std::unexpected(H.error());
    }
    auto R = matrix_to_square_nested(R_m, fn);
    if (!R) {
        return std::unexpected(R.error());
    }
    const auto out = control::kalman_update(*state, *z, *H, *R);
    return vector_to_column(out.x);
}

Result<Matrix<double>> eval_control_kalman_update_cov(const Matrix<double>& x_m,
                                                       const Matrix<double>& P_m,
                                                       const Matrix<double>& z_m,
                                                       const Matrix<double>& H_m,
                                                       const Matrix<double>& R_m) {
    constexpr const char* fn = "control_kalman_update_cov";
    auto state = kalman_state_from_matrices(x_m, P_m, fn);
    if (!state) {
        return std::unexpected(state.error());
    }
    auto z = matrix_to_coeff_vector(z_m, fn);
    if (!z) {
        return std::unexpected(z.error());
    }
    auto H = matrix_to_nested(H_m, fn);
    if (!H) {
        return std::unexpected(H.error());
    }
    auto R = matrix_to_square_nested(R_m, fn);
    if (!R) {
        return std::unexpected(R.error());
    }
    const auto out = control::kalman_update(*state, *z, *H, *R);
    return nested_to_matrix(out.P);
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

Result<Matrix<double>> eval_control_lqe(const Matrix<double>& A_m,
                                        const Matrix<double>& C_m,
                                        const Matrix<double>& Q_m,
                                        const Matrix<double>& R_m) {
    auto A = matrix_to_square_nested(A_m, "control_lqe");
    if (!A) {
        return std::unexpected(A.error());
    }
    auto C = matrix_to_nested(C_m, "control_lqe");
    if (!C) {
        return std::unexpected(C.error());
    }
    auto Q = matrix_to_square_nested(Q_m, "control_lqe");
    if (!Q) {
        return std::unexpected(Q.error());
    }
    auto R = matrix_to_square_nested(R_m, "control_lqe");
    if (!R) {
        return std::unexpected(R.error());
    }
    if (C->empty() || C->front().size() != A->size()) {
        return std::unexpected(
            DomainError{"control_lqe", "expected C with column count equal to A size"});
    }
    if (Q->size() != A->size()) {
        return std::unexpected(
            DomainError{"control_lqe", "expected Q with same size as A"});
    }
    auto est = control::lqe(*A, *C, *Q, *R);
    if (!est) {
        return std::unexpected(est.error());
    }
    return nested_to_matrix(est->L);
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

Result<Matrix<double>> eval_control_poles(const Matrix<double>& num_m,
                                          const Matrix<double>& den_m) {
    auto num = matrix_to_coeff_vector(num_m, "control_poles");
    if (!num) {
        return std::unexpected(num.error());
    }
    auto den = matrix_to_coeff_vector(den_m, "control_poles");
    if (!den) {
        return std::unexpected(den.error());
    }
    if (den->empty() || den->back() == 0.0) {
        return std::unexpected(DomainError{"control_poles", "denominator must be non-zero"});
    }
    const control::TransferFunction sys(std::move(*num), std::move(*den));
    const auto roots = control::poles(sys);
    Matrix<double> out(roots.size(), 2);
    for (size_t i = 0; i < roots.size(); ++i) {
        out(i, 0) = roots[i].real();
        out(i, 1) = roots[i].imag();
    }
    return out;
}

Result<Matrix<double>> eval_control_zeros(const Matrix<double>& num_m,
                                          const Matrix<double>& den_m) {
    auto num = matrix_to_coeff_vector(num_m, "control_zeros");
    if (!num) {
        return std::unexpected(num.error());
    }
    auto den = matrix_to_coeff_vector(den_m, "control_zeros");
    if (!den) {
        return std::unexpected(den.error());
    }
    if (den->empty() || den->back() == 0.0) {
        return std::unexpected(DomainError{"control_zeros", "denominator must be non-zero"});
    }
    const control::TransferFunction sys(std::move(*num), std::move(*den));
    const auto roots = control::zeros(sys);
    Matrix<double> out(roots.size(), 2);
    for (size_t i = 0; i < roots.size(); ++i) {
        out(i, 0) = roots[i].real();
        out(i, 1) = roots[i].imag();
    }
    return out;
}

Result<Matrix<double>> eval_control_step_info(const Matrix<double>& num_m,
                                              const Matrix<double>& den_m) {
    auto num = matrix_to_coeff_vector(num_m, "control_step_info");
    if (!num) {
        return std::unexpected(num.error());
    }
    auto den = matrix_to_coeff_vector(den_m, "control_step_info");
    if (!den) {
        return std::unexpected(den.error());
    }
    if (den->empty() || den->back() == 0.0) {
        return std::unexpected(DomainError{"control_step_info", "denominator must be non-zero"});
    }
    const control::TransferFunction sys(std::move(*num), std::move(*den));
    const control::StepData response = control::step_response(sys);
    if (response.y.empty()) {
        return std::unexpected(DomainError{"control_step_info", "empty step response"});
    }
    const control::StepInfo info = control::step_info(response);
    Matrix<double> out(1, 5);
    out(0, 0) = info.rise_time;
    out(0, 1) = info.settling_time;
    out(0, 2) = info.overshoot_pct;
    out(0, 3) = info.peak_time;
    out(0, 4) = info.peak_value;
    return out;
}

Result<Matrix<double>> eval_control_nyquist(const Matrix<double>& num_m,
                                            const Matrix<double>& den_m) {
    auto num = matrix_to_coeff_vector(num_m, "control_nyquist");
    if (!num) {
        return std::unexpected(num.error());
    }
    auto den = matrix_to_coeff_vector(den_m, "control_nyquist");
    if (!den) {
        return std::unexpected(den.error());
    }
    if (den->empty() || den->back() == 0.0) {
        return std::unexpected(DomainError{"control_nyquist", "denominator must be non-zero"});
    }
    const control::TransferFunction sys(std::move(*num), std::move(*den));
    const auto pts = control::nyquist(sys);
    Matrix<double> out(pts.size(), 2);
    for (size_t i = 0; i < pts.size(); ++i) {
        out(i, 0) = pts[i].first;
        out(i, 1) = pts[i].second;
    }
    return out;
}

Result<Matrix<double>> step_data_to_matrix(const control::StepData& data, const char* fn) {
    if (data.t.empty() || data.y.empty() || data.t.size() != data.y.size()) {
        return std::unexpected(DomainError{fn, "empty time response"});
    }
    Matrix<double> out(data.t.size(), 2);
    for (size_t i = 0; i < data.t.size(); ++i) {
        out(i, 0) = data.t[i];
        out(i, 1) = data.y[i];
    }
    return out;
}

Result<Matrix<double>> eval_control_step_response(const Matrix<double>& num_m,
                                                  const Matrix<double>& den_m, double t_end,
                                                  int n_pts) {
    constexpr const char* fn = "control_step_response";
    if (!(t_end > 0.0)) {
        return std::unexpected(DomainError{fn, "expected positive t_end"});
    }
    if (n_pts < 2) {
        return std::unexpected(DomainError{fn, "expected n_pts >= 2"});
    }
    auto num = matrix_to_coeff_vector(num_m, fn);
    if (!num) {
        return std::unexpected(num.error());
    }
    auto den = matrix_to_coeff_vector(den_m, fn);
    if (!den) {
        return std::unexpected(den.error());
    }
    if (den->empty() || den->back() == 0.0) {
        return std::unexpected(DomainError{fn, "denominator must be non-zero"});
    }
    const control::TransferFunction sys(std::move(*num), std::move(*den));
    return step_data_to_matrix(control::step_response(sys, t_end, n_pts), fn);
}

Result<Matrix<double>> eval_control_impulse_response(const Matrix<double>& num_m,
                                                     const Matrix<double>& den_m, double t_end,
                                                     int n_pts) {
    constexpr const char* fn = "control_impulse_response";
    if (!(t_end > 0.0)) {
        return std::unexpected(DomainError{fn, "expected positive t_end"});
    }
    if (n_pts < 2) {
        return std::unexpected(DomainError{fn, "expected n_pts >= 2"});
    }
    auto num = matrix_to_coeff_vector(num_m, fn);
    if (!num) {
        return std::unexpected(num.error());
    }
    auto den = matrix_to_coeff_vector(den_m, fn);
    if (!den) {
        return std::unexpected(den.error());
    }
    if (den->empty() || den->back() == 0.0) {
        return std::unexpected(DomainError{fn, "denominator must be non-zero"});
    }
    const control::TransferFunction sys(std::move(*num), std::move(*den));
    return step_data_to_matrix(control::impulse_response(sys, t_end, n_pts), fn);
}


Result<Matrix<double>> pack_state_space(const control::StateSpace& sys, const char* fn) {
    const size_t n = static_cast<size_t>(sys.n);
    const size_t m = static_cast<size_t>(sys.m);
    const size_t p = static_cast<size_t>(sys.p);
    if (n == 0 || m == 0 || p == 0) {
        return std::unexpected(DomainError{fn, "empty state-space system"});
    }
    Matrix<double> out(n + p, n + m, 0.0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            out(i, j) = sys.A[i][j];
        }
        for (size_t j = 0; j < m; ++j) {
            out(i, n + j) = sys.B[i][j];
        }
    }
    for (size_t i = 0; i < p; ++i) {
        for (size_t j = 0; j < n; ++j) {
            out(n + i, j) = sys.C[i][j];
        }
        for (size_t j = 0; j < m; ++j) {
            out(n + i, n + j) = sys.D[i][j];
        }
    }
    return out;
}

Result<Matrix<double>> eval_control_tf2ss(const Matrix<double>& num_m,
                                          const Matrix<double>& den_m) {
    constexpr const char* fn = "control_tf2ss";
    auto num = matrix_to_coeff_vector(num_m, fn);
    if (!num) {
        return std::unexpected(num.error());
    }
    auto den = matrix_to_coeff_vector(den_m, fn);
    if (!den) {
        return std::unexpected(den.error());
    }
    if (den->empty() || den->back() == 0.0) {
        return std::unexpected(DomainError{fn, "denominator must be non-zero"});
    }
    const control::TransferFunction sys(std::move(*num), std::move(*den));
    return pack_state_space(control::tf2ss(sys), fn);
}

Result<Matrix<double>> eval_control_c2d(const Matrix<double>& A_m, const Matrix<double>& B_m,
                                        const Matrix<double>& C_m, const Matrix<double>& D_m,
                                        double Ts, control::DiscretizationMethod method,
                                        const char* fn) {
    if (!(Ts > 0.0)) {
        return std::unexpected(DomainError{fn, "expected positive Ts"});
    }
    auto A = matrix_to_square_nested(A_m, fn);
    if (!A) {
        return std::unexpected(A.error());
    }
    auto B = matrix_to_nested(B_m, fn);
    if (!B) {
        return std::unexpected(B.error());
    }
    auto C = matrix_to_nested(C_m, fn);
    if (!C) {
        return std::unexpected(C.error());
    }
    auto D = matrix_to_nested(D_m, fn);
    if (!D) {
        return std::unexpected(D.error());
    }
    if (B->size() != A->size()) {
        return std::unexpected(DomainError{fn, "expected B with same row count as A"});
    }
    if (C->empty() || (*C)[0].size() != A->size()) {
        return std::unexpected(DomainError{fn, "expected C with column count equal to A size"});
    }

    const control::StateSpace sys(std::move(*A), std::move(*B), std::move(*C), std::move(*D));
    const auto disc = control::c2d(sys, Ts, method);
    return nested_to_matrix(disc.A);
}

Result<Matrix<double>> eval_control_c2d_B(const Matrix<double>& A_m, const Matrix<double>& B_m,
                                          const Matrix<double>& C_m, const Matrix<double>& D_m,
                                          double Ts) {
    constexpr const char* fn = "control_c2d_b";
    if (!(Ts > 0.0)) {
        return std::unexpected(DomainError{fn, "expected positive Ts"});
    }
    auto A = matrix_to_square_nested(A_m, fn);
    if (!A) {
        return std::unexpected(A.error());
    }
    auto B = matrix_to_nested(B_m, fn);
    if (!B) {
        return std::unexpected(B.error());
    }
    auto C = matrix_to_nested(C_m, fn);
    if (!C) {
        return std::unexpected(C.error());
    }
    auto D = matrix_to_nested(D_m, fn);
    if (!D) {
        return std::unexpected(D.error());
    }
    if (B->size() != A->size()) {
        return std::unexpected(DomainError{fn, "expected B with same row count as A"});
    }
    if (C->empty() || (*C)[0].size() != A->size()) {
        return std::unexpected(DomainError{fn, "expected C with column count equal to A size"});
    }

    const control::StateSpace sys(std::move(*A), std::move(*B), std::move(*C), std::move(*D));
    const auto disc = control::c2d(sys, Ts, control::DiscretizationMethod::ZOH);
    return nested_to_matrix(disc.B);
}

Matrix<double> pack_transfer_function(const control::TransferFunction& sys) {
    const size_t n = std::max(sys.num.size(), sys.den.size());
    Matrix<double> out(2, n, 0.0);
    for (size_t j = 0; j < n; ++j) {
        out(0, j) = j < sys.num.size() ? sys.num[j] : 0.0;
        out(1, j) = j < sys.den.size() ? sys.den[j] : 0.0;
    }
    return out;
}

Result<control::TransferFunction> matrices_to_transfer_function(const Matrix<double>& num_m,
                                                                const Matrix<double>& den_m,
                                                                const char* fn) {
    auto num = matrix_to_coeff_vector(num_m, fn);
    if (!num) {
        return std::unexpected(num.error());
    }
    auto den = matrix_to_coeff_vector(den_m, fn);
    if (!den) {
        return std::unexpected(den.error());
    }
    if (den->empty() || den->front() == 0.0) {
        return std::unexpected(DomainError{fn, "denominator must be non-zero"});
    }
    return control::TransferFunction(std::move(*num), std::move(*den));
}

Result<control::StateSpace> packed_ss_to_state_space(const Matrix<double>& ss_m,
                                                     const char* fn) {
    if (ss_m.rows() < 2 || ss_m.cols() < 2 || ss_m.rows() != ss_m.cols()) {
        return std::unexpected(DomainError{
            fn, "expected square packed SS [A B; C D] from control_tf2ss"});
    }
    const size_t n = ss_m.rows() - 1;
    std::vector<std::vector<double>> A(n, std::vector<double>(n, 0.0));
    std::vector<std::vector<double>> B(n, std::vector<double>(1, 0.0));
    std::vector<std::vector<double>> C(1, std::vector<double>(n, 0.0));
    std::vector<std::vector<double>> D(1, std::vector<double>(1, 0.0));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            A[i][j] = ss_m(i, j);
        }
        B[i][0] = ss_m(i, n);
    }
    for (size_t j = 0; j < n; ++j) {
        C[0][j] = ss_m(n, j);
    }
    D[0][0] = ss_m(n, n);
    return control::StateSpace(std::move(A), std::move(B), std::move(C), std::move(D));
}

Result<Matrix<double>> eval_control_series(const Matrix<double>& num1_m,
                                           const Matrix<double>& den1_m,
                                           const Matrix<double>& num2_m,
                                           const Matrix<double>& den2_m) {
    constexpr const char* fn = "control_series";
    auto g = matrices_to_transfer_function(num1_m, den1_m, fn);
    if (!g) {
        return std::unexpected(g.error());
    }
    auto h = matrices_to_transfer_function(num2_m, den2_m, fn);
    if (!h) {
        return std::unexpected(h.error());
    }
    return pack_transfer_function(control::series(*g, *h));
}

Result<Matrix<double>> eval_control_parallel(const Matrix<double>& num1_m,
                                           const Matrix<double>& den1_m,
                                           const Matrix<double>& num2_m,
                                           const Matrix<double>& den2_m) {
    constexpr const char* fn = "control_parallel";
    auto g = matrices_to_transfer_function(num1_m, den1_m, fn);
    if (!g) {
        return std::unexpected(g.error());
    }
    auto h = matrices_to_transfer_function(num2_m, den2_m, fn);
    if (!h) {
        return std::unexpected(h.error());
    }
    return pack_transfer_function(control::parallel(*g, *h));
}

Result<Matrix<double>> eval_control_feedback(const Matrix<double>& numG_m,
                                             const Matrix<double>& denG_m,
                                             const Matrix<double>& numH_m,
                                             const Matrix<double>& denH_m, int sign) {
    constexpr const char* fn = "control_feedback";
    auto g = matrices_to_transfer_function(numG_m, denG_m, fn);
    if (!g) {
        return std::unexpected(g.error());
    }
    auto h = matrices_to_transfer_function(numH_m, denH_m, fn);
    if (!h) {
        return std::unexpected(h.error());
    }
    return pack_transfer_function(control::feedback(*g, *h, sign));
}

Result<Matrix<double>> eval_control_ss2tf(const Matrix<double>& ss_m) {
    constexpr const char* fn = "control_ss2tf";
    auto sys = packed_ss_to_state_space(ss_m, fn);
    if (!sys) {
        return std::unexpected(sys.error());
    }
    return pack_transfer_function(control::ss2tf(*sys));
}

Result<Matrix<double>> eval_control_d2c(const Matrix<double>& A_m, const Matrix<double>& B_m,
                                        const Matrix<double>& C_m, const Matrix<double>& D_m,
                                        double Ts, control::DiscretizationMethod method,
                                        const char* fn) {
    if (!(Ts > 0.0)) {
        return std::unexpected(DomainError{fn, "expected positive Ts"});
    }
    auto A = matrix_to_square_nested(A_m, fn);
    if (!A) {
        return std::unexpected(A.error());
    }
    auto B = matrix_to_nested(B_m, fn);
    if (!B) {
        return std::unexpected(B.error());
    }
    auto C = matrix_to_nested(C_m, fn);
    if (!C) {
        return std::unexpected(C.error());
    }
    auto D = matrix_to_nested(D_m, fn);
    if (!D) {
        return std::unexpected(D.error());
    }
    if (B->size() != A->size()) {
        return std::unexpected(DomainError{fn, "expected B with same row count as A"});
    }
    if (C->empty() || (*C)[0].size() != A->size()) {
        return std::unexpected(DomainError{fn, "expected C with column count equal to A size"});
    }

    const control::StateSpace sys(std::move(*A), std::move(*B), std::move(*C), std::move(*D));
    const auto cont = control::d2c(sys, Ts, method);
    return nested_to_matrix(cont.A);
}

Result<Matrix<double>> eval_control_c2d_tf(const Matrix<double>& num_m,
                                          const Matrix<double>& den_m, double Ts,
                                          control::DiscretizationMethod method,
                                          const char* fn) {
    if (!(Ts > 0.0)) {
        return std::unexpected(DomainError{fn, "expected positive Ts"});
    }
    auto sys = matrices_to_transfer_function(num_m, den_m, fn);
    if (!sys) {
        return std::unexpected(sys.error());
    }

    return pack_transfer_function(control::c2d(*sys, Ts, method));
}

Result<Matrix<double>> eval_control_d2c_tf(const Matrix<double>& num_m,
                                          const Matrix<double>& den_m, double Ts,
                                          control::DiscretizationMethod method,
                                          const char* fn) {
    if (!(Ts > 0.0)) {
        return std::unexpected(DomainError{fn, "expected positive Ts"});
    }
    auto sys = matrices_to_transfer_function(num_m, den_m, fn);
    if (!sys) {
        return std::unexpected(sys.error());
    }

    return pack_transfer_function(control::d2c(*sys, Ts, method));
}

Result<double> eval_quantum_purity(const Matrix<double>& rho_m) {
    auto rho = matrix_to_density_matrix(rho_m, "quantum_purity");
    if (!rho) {
        return std::unexpected(rho.error());
    }
    return quantum::purity(*rho);
}

Result<double> eval_quantum_wigner(const Matrix<double>& rho_m, double x, double p) {
    auto rho = matrix_to_density_matrix(rho_m, "quantum_wigner");
    if (!rho) {
        return std::unexpected(rho.error());
    }
    return quantum::wigner_function(*rho, x, p);
}

Result<double> eval_quantum_husimi(const Matrix<double>& rho_m, double alpha_re, double alpha_im) {
    auto rho = matrix_to_density_matrix(rho_m, "quantum_husimi");
    if (!rho) {
        return std::unexpected(rho.error());
    }
    return quantum::husimi_Q(*rho, quantum::C(alpha_re, alpha_im));
}

Result<Matrix<double>> eval_quantum_grover_search(int n_qubits, const Matrix<double>& marked_m,
                                                  int n_iterations) {
    if (n_qubits < 1) {
        return std::unexpected(
            DomainError{"quantum_grover_search", "expected positive integer n_qubits"});
    }
    auto marked = matrix_to_int_coeff_vector(marked_m, "quantum_grover_search");
    if (!marked) {
        return std::unexpected(marked.error());
    }
    const auto psi = quantum::grover_search(n_qubits, *marked, n_iterations);
    if (psi.empty()) {
        return std::unexpected(
            DomainError{"quantum_grover_search", "expected positive integer n_qubits"});
    }
    return ket_to_column_matrix(psi);
}

Result<double> eval_quantum_schmidt_rank(const Matrix<double>& psi_m, int dim_a, int dim_b) {
    auto psi = matrix_to_ket(psi_m, "quantum_schmidt_rank");
    if (!psi) {
        return std::unexpected(psi.error());
    }
    return static_cast<double>(quantum::schmidt_rank(*psi, dim_a, dim_b));
}

Result<double> eval_quantum_uncertainty(const Matrix<double>& psi_m, const Matrix<double>& A_m,
                                        const Matrix<double>& B_m) {
    auto psi = matrix_to_ket(psi_m, "quantum_uncertainty");
    if (!psi) {
        return std::unexpected(psi.error());
    }
    auto A = matrix_to_density_matrix(A_m, "quantum_uncertainty");
    if (!A) {
        return std::unexpected(A.error());
    }
    auto B = matrix_to_density_matrix(B_m, "quantum_uncertainty");
    if (!B) {
        return std::unexpected(B.error());
    }
    if (A->size() != psi->size() || B->size() != psi->size()) {
        return std::unexpected(DomainError{
            "quantum_uncertainty", "operators must match state vector dimension"});
    }
    return quantum::uncertainty(*psi, *A, *B);
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
            DomainError{"quantum_schrodinger_final", "empty SchrÃƒÂ¶dinger trajectory"});
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

// Peak/corner matrix converters.
Matrix<double> hough_lines_to_matrix(const std::vector<image::HoughLine>& lines) {
    Matrix<double> out(lines.size(), 3);
    for (size_t i = 0; i < lines.size(); ++i) {
        out(i, 0) = lines[i].rho;
        out(i, 1) = lines[i].theta;
        out(i, 2) = static_cast<double>(lines[i].votes);
    }
    return out;
}

Matrix<double> hough_circles_to_matrix(const std::vector<image::HoughCircle>& circles) {
    Matrix<double> out(circles.size(), 4);
    for (size_t i = 0; i < circles.size(); ++i) {
        out(i, 0) = circles[i].cx;
        out(i, 1) = circles[i].cy;
        out(i, 2) = circles[i].r;
        out(i, 3) = static_cast<double>(circles[i].votes);
    }
    return out;
}

Matrix<double> keypoints_to_matrix(const std::vector<image::KeyPoint>& kps) {
    Matrix<double> out(kps.size(), 3);
    for (size_t i = 0; i < kps.size(); ++i) {
        out(i, 0) = kps[i].x;
        out(i, 1) = kps[i].y;
        out(i, 2) = static_cast<double>(kps[i].response);
    }
    return out;
}

Result<Matrix<double>> eval_hough_lines(const Matrix<double>& m, double edge_threshold,
                                        int n_theta, int n_rho, int vote_threshold) {
    auto gray = matrix_to_gray_image(m);
    if (!gray) {
        return std::unexpected(gray.error());
    }
    return hough_lines_to_matrix(
        image::hough_lines(*gray, edge_threshold, n_theta, n_rho, vote_threshold));
}

Result<Matrix<double>> eval_hough_circles(const Matrix<double>& m, double edge_threshold,
                                          double r_min, double r_max, int r_step,
                                          int vote_threshold) {
    auto gray = matrix_to_gray_image(m);
    if (!gray) {
        return std::unexpected(gray.error());
    }
    return hough_circles_to_matrix(image::hough_circles(*gray, edge_threshold, r_min, r_max,
                                                        r_step, vote_threshold));
}

Result<Matrix<double>> eval_harris(const Matrix<double>& m, float k, float threshold) {
    auto gray = matrix_to_gray_image(m);
    if (!gray) {
        return std::unexpected(gray.error());
    }
    return keypoints_to_matrix(image::harris(*gray, k, threshold));
}

Result<Matrix<double>> eval_shi_tomasi(const Matrix<double>& m, int n, float quality_level) {
    auto gray = matrix_to_gray_image(m);
    if (!gray) {
        return std::unexpected(gray.error());
    }
    return keypoints_to_matrix(image::shi_tomasi(*gray, n, quality_level));
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

Result<double> eval_graph_min_cut(const Matrix<double>& adj_m, int source, int sink) {
    auto G = graph_from_adjacency(adj_m, "graph_min_cut");
    if (!G) {
        return std::unexpected(G.error());
    }
    if (source < 0 || sink < 0 || source >= G->n_vertices() || sink >= G->n_vertices()) {
        return std::unexpected(
            DomainError{"graph_min_cut", "source/sink out of range"});
    }
    auto cut = graph::min_cut(*G, source, sink);
    if (!cut) {
        return std::unexpected(cut.error());
    }
    return cut->value;
}

Result<Matrix<double>> eval_graph_bridges(const Matrix<double>& adj_m) {
    auto G = graph_from_adjacency_undirected(adj_m, "graph_bridges");
    if (!G) {
        return std::unexpected(G.error());
    }
    const auto edges = graph::bridges(*G);
    Matrix<double> out(edges.size(), 3);
    for (size_t i = 0; i < edges.size(); ++i) {
        out(i, 0) = static_cast<double>(edges[i].from);
        out(i, 1) = static_cast<double>(edges[i].to);
        out(i, 2) = edges[i].weight;
    }
    return out;
}

Result<Matrix<double>> eval_graph_maximum_matching(const Matrix<double>& adj_m) {
    auto G = graph_from_adjacency_undirected(adj_m, "graph_maximum_matching");
    if (!G) {
        return std::unexpected(G.error());
    }
    const auto edges = graph::maximum_matching(*G);
    Matrix<double> out(edges.size(), 2);
    for (size_t i = 0; i < edges.size(); ++i) {
        out(i, 0) = static_cast<double>(edges[i].first);
        out(i, 1) = static_cast<double>(edges[i].second);
    }
    return out;
}

Result<Matrix<double>> eval_graph_transitive_closure(const Matrix<double>& adj_m) {
    auto G = graph_from_adjacency(adj_m, "graph_transitive_closure");
    if (!G) {
        return std::unexpected(G.error());
    }
    const auto reach = graph::transitive_closure(*G);
    if (reach.empty()) {
        return Matrix<double>{0, 0};
    }
    Matrix<double> out(reach.size(), reach[0].size());
    for (size_t i = 0; i < reach.size(); ++i) {
        for (size_t j = 0; j < reach[i].size(); ++j) {
            out(i, j) = reach[i][j] ? 1.0 : 0.0;
        }
    }
    return out;
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

Result<Matrix<double>> eval_signal_upsample(const Matrix<double>& x_m, int n) {
    auto x = matrix_to_coeff_vector(x_m, "signal_upsample");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (x->empty()) {
        return std::unexpected(
            DomainError{"signal_upsample", "expected non-empty signal vector"});
    }
    if (n < 1) {
        return std::unexpected(
            DomainError{"signal_upsample", "expected n >= 1"});
    }
    return vector_to_column(upsample(*x, n));
}

Result<Matrix<double>> eval_signal_downsample(const Matrix<double>& x_m, int n) {
    auto x = matrix_to_coeff_vector(x_m, "signal_downsample");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (x->empty()) {
        return std::unexpected(
            DomainError{"signal_downsample", "expected non-empty signal vector"});
    }
    if (n < 1) {
        return std::unexpected(
            DomainError{"signal_downsample", "expected n >= 1"});
    }
    return vector_to_column(downsample(*x, n));
}

Result<int> require_positive_int_arg(double v, const char* fn, const char* arg_name) {
    const int n = static_cast<int>(v);
    if (n < 1 || v != n) {
        return std::unexpected(DomainError{
            fn, std::string("expected positive integer ") + arg_name});
    }
    return n;
}

Result<Matrix<double>> eval_signal_decimate(const Matrix<double>& x_m, int q) {
    auto x = matrix_to_coeff_vector(x_m, "signal_decimate");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (x->empty()) {
        return std::unexpected(
            DomainError{"signal_decimate", "expected non-empty signal vector"});
    }
    if (q < 1) {
        return std::unexpected(
            DomainError{"signal_decimate", "expected q >= 1"});
    }
    return vector_to_column(decimate(*x, q));
}

Result<Matrix<double>> eval_signal_interpolate(const Matrix<double>& x_m, int p) {
    auto x = matrix_to_coeff_vector(x_m, "signal_interpolate");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (x->empty()) {
        return std::unexpected(
            DomainError{"signal_interpolate", "expected non-empty signal vector"});
    }
    if (p < 1) {
        return std::unexpected(
            DomainError{"signal_interpolate", "expected p >= 1"});
    }
    return vector_to_column(interpolate(*x, p));
}

Result<Matrix<double>> eval_signal_resample(const Matrix<double>& x_m, int p, int q) {
    auto x = matrix_to_coeff_vector(x_m, "signal_resample");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (x->empty()) {
        return std::unexpected(
            DomainError{"signal_resample", "expected non-empty signal vector"});
    }
    if (p < 1) {
        return std::unexpected(
            DomainError{"signal_resample", "expected p >= 1"});
    }
    if (q < 1) {
        return std::unexpected(
            DomainError{"signal_resample", "expected q >= 1"});
    }
    return vector_to_column(resample(*x, p, q));
}

// Wave 249: shared matrix+integer-factor dispatcher (avoids deep nesting / MSVC C1061).
Result<Matrix<double>> eval_signal_integer_factor(const std::string& fn,
                                                 const Matrix<double>& x_m,
                                                 double factor) {
    const char* arg_name = "n";
    if (fn == "signal_decimate") {
        arg_name = "q";
    } else if (fn == "signal_interpolate") {
        arg_name = "p";
    }
    auto n = require_positive_int_arg(factor, fn.c_str(), arg_name);
    if (!n) {
        return std::unexpected(n.error());
    }
    if (fn == "signal_upsample") {
        return eval_signal_upsample(x_m, *n);
    }
    if (fn == "signal_downsample") {
        return eval_signal_downsample(x_m, *n);
    }
    if (fn == "signal_decimate") {
        return eval_signal_decimate(x_m, *n);
    }
    if (fn == "signal_interpolate") {
        return eval_signal_interpolate(x_m, *n);
    }
    return std::unexpected(DomainError{fn, "unsupported signal integer-factor call"});
}

std::string format_labeled_matrix(const std::string& label, const Matrix<double>& m) {
    std::ostringstream out;
    out << label << " =\n";
    print_matrix(out, m);
    return out.str();
}


Result<Matrix<double>> eval_signal_resample_pq(const Matrix<double>& x_m, double p_d,
                                              double q_d) {
    auto p = require_positive_int_arg(p_d, "signal_resample", "p");
    if (!p) {
        return std::unexpected(p.error());
    }
    auto q = require_positive_int_arg(q_d, "signal_resample", "q");
    if (!q) {
        return std::unexpected(q.error());
    }
    return eval_signal_resample(x_m, *p, *q);
}

// Wave 252: extracted helpers keep assignment/ternary paths shallow (MSVC C1061).
Result<Matrix<double>> eval_signal_savgol(const Matrix<double>& x_m, int window_length,
                                          int polyorder) {
    auto x = matrix_to_coeff_vector(x_m, "signal_savgol");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (x->empty()) {
        return std::unexpected(
            DomainError{"signal_savgol", "expected non-empty signal vector"});
    }
    if (window_length <= 0 || window_length % 2 == 0) {
        return std::unexpected(
            DomainError{"signal_savgol", "expected odd positive window_length"});
    }
    if (polyorder < 0 || polyorder >= window_length) {
        return std::unexpected(
            DomainError{"signal_savgol", "expected polyorder in [0, window_length)"});
    }
    if (x->size() < static_cast<size_t>(window_length)) {
        return std::unexpected(
            DomainError{"signal_savgol", "expected signal length >= window_length"});
    }
    return vector_to_column(savgol(*x, window_length, polyorder));
}

Result<Matrix<double>> eval_signal_savgol_wp(const Matrix<double>& x_m, double window_d,
                                            double poly_d) {
    const int window_length = static_cast<int>(window_d);
    if (window_length < 1 || window_d != window_length) {
        return std::unexpected(DomainError{
            "signal_savgol", "expected positive integer window_length"});
    }
    const int polyorder = static_cast<int>(poly_d);
    if (polyorder < 0 || poly_d != polyorder) {
        return std::unexpected(DomainError{
            "signal_savgol", "expected non-negative integer polyorder"});
    }
    return eval_signal_savgol(x_m, window_length, polyorder);
}

// Wave 253: extracted helpers keep assignment/binary paths shallow (MSVC C1061).
Result<Matrix<double>> eval_signal_median_filter(const Matrix<double>& x_m, int window_length) {
    auto x = matrix_to_coeff_vector(x_m, "signal_median_filter");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (x->empty()) {
        return std::unexpected(
            DomainError{"signal_median_filter", "expected non-empty signal vector"});
    }
    if (window_length <= 0 || window_length % 2 == 0) {
        return std::unexpected(
            DomainError{"signal_median_filter", "expected odd positive window_length"});
    }
    if (x->size() < static_cast<size_t>(window_length)) {
        return std::unexpected(
            DomainError{"signal_median_filter", "expected signal length >= window_length"});
    }
    return vector_to_column(median_filter(*x, window_length));
}

Result<Matrix<double>> eval_signal_median_filter_w(const Matrix<double>& x_m, double window_d) {
    const int window_length = static_cast<int>(window_d);
    if (window_length < 1 || window_d != window_length) {
        return std::unexpected(DomainError{
            "signal_median_filter", "expected positive integer window_length"});
    }
    return eval_signal_median_filter(x_m, window_length);
}

// Wave 254: extracted helpers keep assignment paths shallow (MSVC C1061).
Result<LMSResult> run_signal_lms(const Matrix<double>& x_m, const Matrix<double>& d_m,
                                 double filter_length_d, double mu, const char* fn) {
    auto x = matrix_to_coeff_vector(x_m, fn);
    if (!x) {
        return std::unexpected(x.error());
    }
    auto d = matrix_to_coeff_vector(d_m, fn);
    if (!d) {
        return std::unexpected(d.error());
    }
    if (x->empty() || d->empty()) {
        return std::unexpected(DomainError{fn, "expected non-empty signal vectors"});
    }
    if (x->size() != d->size()) {
        return std::unexpected(DomainError{fn, "expected x and d of equal length"});
    }
    const int filter_length = static_cast<int>(filter_length_d);
    if (filter_length < 1 || filter_length_d != filter_length) {
        return std::unexpected(DomainError{fn, "expected positive integer filter_length"});
    }
    if (x->size() < static_cast<size_t>(filter_length)) {
        return std::unexpected(
            DomainError{fn, "expected signal length >= filter_length"});
    }
    auto result = lms_adaptive_filter(*x, *d, filter_length, mu);
    if (result.output.empty() || result.error.size() != result.output.size() ||
        result.weights.size() != static_cast<size_t>(filter_length)) {
        return std::unexpected(DomainError{fn, "lms_adaptive_filter failed"});
    }
    return result;
}

Result<Matrix<double>> eval_signal_lms(const Matrix<double>& x_m, const Matrix<double>& d_m,
                                       double filter_length_d, double mu) {
    auto result = run_signal_lms(x_m, d_m, filter_length_d, mu, "signal_lms");
    if (!result) {
        return std::unexpected(result.error());
    }
    Matrix<double> out(result->output.size(), 2);
    for (size_t i = 0; i < result->output.size(); ++i) {
        out(i, 0) = result->output[i];
        out(i, 1) = result->error[i];
    }
    return out;
}

Result<Matrix<double>> eval_signal_lms_weights(const Matrix<double>& x_m,
                                               const Matrix<double>& d_m,
                                               double filter_length_d, double mu) {
    auto result = run_signal_lms(x_m, d_m, filter_length_d, mu, "signal_lms_weights");
    if (!result) {
        return std::unexpected(result.error());
    }
    return vector_to_column(result->weights);
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

Result<double> eval_geo_kdtree_3d_nearest(const Matrix<double>& P_m, double qx, double qy,
                                          double qz) {
    auto pts = matrix_to_points3d(P_m, "geo_kdtree_3d_nearest");
    if (!pts) {
        return std::unexpected(pts.error());
    }
    if (pts->empty()) {
        return std::unexpected(
            DomainError{"geo_kdtree_3d_nearest", "expected non-empty Nx3 point matrix"});
    }
    const geo::KDTree3D kd(*pts);
    return static_cast<double>(kd.nearest({qx, qy, qz}));
}

Result<Matrix<double>> eval_geo_kdtree_knn(const Matrix<double>& P_m, double qx, double qy,
                                           double k_d) {
    auto pts = matrix_to_points2d(P_m, "geo_kdtree_knn");
    if (!pts) {
        return std::unexpected(pts.error());
    }
    if (pts->empty()) {
        return std::unexpected(
            DomainError{"geo_kdtree_knn", "expected non-empty Nx2 point matrix"});
    }
    const int k = static_cast<int>(k_d);
    if (k < 1 || k_d != k) {
        return std::unexpected(DomainError{"geo_kdtree_knn", "expected positive integer k"});
    }
    const geo::KDTree2D kd(*pts);
    return int_vector_to_column(kd.knn({qx, qy}, k));
}

Result<Matrix<double>> eval_geo_kdtree_range(const Matrix<double>& P_m, double qx, double qy,
                                             double r) {
    auto pts = matrix_to_points2d(P_m, "geo_kdtree_range");
    if (!pts) {
        return std::unexpected(pts.error());
    }
    if (pts->empty()) {
        return std::unexpected(
            DomainError{"geo_kdtree_range", "expected non-empty Nx2 point matrix"});
    }
    if (!(r >= 0.0)) {
        return std::unexpected(DomainError{"geo_kdtree_range", "expected non-negative radius r"});
    }
    const geo::KDTree2D kd(*pts);
    return int_vector_to_column(kd.range({qx, qy}, r));
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

Result<Matrix<double>> eval_combo_prev_perm(const Matrix<double>& v_m) {
    auto v_vec = matrix_to_coeff_vector(v_m, "combo_prev_perm");
    if (!v_vec) {
        return std::unexpected(v_vec.error());
    }
    if (v_vec->empty()) {
        return std::unexpected(
            DomainError{"combo_prev_perm", "expected non-empty permutation vector"});
    }
    std::vector<int> v;
    v.reserve(v_vec->size());
    for (const double entry : *v_vec) {
        if (entry < 0.0 || std::floor(entry) != entry) {
            return std::unexpected(
                DomainError{"combo_prev_perm", "expected non-negative integer entries"});
        }
        v.push_back(static_cast<int>(entry));
    }
    combo::prev_perm(v);
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

Result<Matrix<double>> eval_geo_convex_hull(const Matrix<double>& P_m) {
    auto pts = matrix_to_points2d(P_m, "geo_convex_hull");
    if (!pts) {
        return std::unexpected(pts.error());
    }
    if (pts->size() < 3) {
        return std::unexpected(
            DomainError{"geo_convex_hull", "expected at least 3 points"});
    }
    const auto hull = geo::convex_hull_2d(*pts);
    if (hull.size() < 3) {
        return std::unexpected(
            DomainError{"geo_convex_hull", "degenerate convex hull"});
    }
    return points2d_to_matrix(hull);
}

Result<Matrix<double>> eval_geo_upper_hull(const Matrix<double>& P_m) {
    auto pts = matrix_to_points2d(P_m, "geo_upper_hull");
    if (!pts) {
        return std::unexpected(pts.error());
    }
    if (pts->size() < 2) {
        return std::unexpected(
            DomainError{"geo_upper_hull", "expected at least 2 points"});
    }
    return points2d_to_matrix(geo::upper_hull(*pts));
}

Result<Matrix<double>> eval_geo_lower_hull(const Matrix<double>& P_m) {
    auto pts = matrix_to_points2d(P_m, "geo_lower_hull");
    if (!pts) {
        return std::unexpected(pts.error());
    }
    if (pts->size() < 2) {
        return std::unexpected(
            DomainError{"geo_lower_hull", "expected at least 2 points"});
    }
    return points2d_to_matrix(geo::lower_hull(*pts));
}

Result<Matrix<double>> eval_geo_bezier_subdivide(const Matrix<double>& ctrl_m, double t) {
    auto ctrl = matrix_to_points2d(ctrl_m, "geo_bezier_subdivide");
    if (!ctrl) {
        return std::unexpected(ctrl.error());
    }
    if (ctrl->size() < 2) {
        return std::unexpected(
            DomainError{"geo_bezier_subdivide", "expected at least 2 control points"});
    }
    const auto [left, right] = geo::bezier_subdivide(*ctrl, t);
    Matrix<double> out(left.size() + right.size(), 2);
    for (size_t i = 0; i < left.size(); ++i) {
        out(i, 0) = left[i].x;
        out(i, 1) = left[i].y;
    }
    for (size_t i = 0; i < right.size(); ++i) {
        out(left.size() + i, 0) = right[i].x;
        out(left.size() + i, 1) = right[i].y;
    }
    return out;
}

Result<Matrix<double>> eval_geo_kdtree_3d_knn(const Matrix<double>& P_m, double qx, double qy,
                                               double qz, double k_d) {
    auto pts = matrix_to_points3d(P_m, "geo_kdtree_3d_knn");
    if (!pts) {
        return std::unexpected(pts.error());
    }
    if (pts->empty()) {
        return std::unexpected(
            DomainError{"geo_kdtree_3d_knn", "expected non-empty Nx3 point matrix"});
    }
    const int k = static_cast<int>(k_d);
    if (k < 1 || k_d != k) {
        return std::unexpected(DomainError{"geo_kdtree_3d_knn", "expected positive integer k"});
    }
    const geo::KDTree3D kd(*pts);
    return int_vector_to_column(kd.knn({qx, qy, qz}, k));
}

Result<Matrix<double>> eval_geo_kdtree_3d_range(const Matrix<double>& P_m, double qx, double qy,
                                                double qz, double r) {
    auto pts = matrix_to_points3d(P_m, "geo_kdtree_3d_range");
    if (!pts) {
        return std::unexpected(pts.error());
    }
    if (pts->empty()) {
        return std::unexpected(
            DomainError{"geo_kdtree_3d_range", "expected non-empty Nx3 point matrix"});
    }
    if (!(r >= 0.0)) {
        return std::unexpected(
            DomainError{"geo_kdtree_3d_range", "expected non-negative radius r"});
    }
    const geo::KDTree3D kd(*pts);
    return int_vector_to_column(kd.range({qx, qy, qz}, r));
}

Result<Matrix<double>> eval_geo_triangulate_polygon(const Matrix<double>& P_m) {
    auto pts = matrix_to_points2d(P_m, "geo_triangulate_polygon");
    if (!pts) {
        return std::unexpected(pts.error());
    }
    if (pts->size() < 3) {
        return std::unexpected(
            DomainError{"geo_triangulate_polygon", "expected at least 3 points"});
    }
    const auto tris = geo::triangulate_polygon(*pts);
    Matrix<double> out(tris.size(), 3);
    for (size_t i = 0; i < tris.size(); ++i) {
        out(i, 0) = static_cast<double>(tris[i].a);
        out(i, 1) = static_cast<double>(tris[i].b);
        out(i, 2) = static_cast<double>(tris[i].c);
    }
    return out;
}

Result<Matrix<double>> eval_geo_convex_hull_3d(const Matrix<double>& P_m) {
    auto pts = matrix_to_points3d(P_m, "geo_convex_hull_3d");
    if (!pts) {
        return std::unexpected(pts.error());
    }
    if (pts->size() < 4) {
        return std::unexpected(
            DomainError{"geo_convex_hull_3d", "expected at least 4 points"});
    }
    const auto faces = geo::convex_hull_3d(*pts);
    Matrix<double> out(faces.size(), 3);
    for (size_t i = 0; i < faces.size(); ++i) {
        out(i, 0) = static_cast<double>(faces[i].a);
        out(i, 1) = static_cast<double>(faces[i].b);
        out(i, 2) = static_cast<double>(faces[i].c);
    }
    return out;
}

Result<Matrix<double>> eval_geo_poly_boolean(const char* fn, const Matrix<double>& a_m,
                                             const Matrix<double>& b_m) {
    auto a = matrix_to_points2d(a_m, fn);
    if (!a) {
        return std::unexpected(a.error());
    }
    auto b = matrix_to_points2d(b_m, fn);
    if (!b) {
        return std::unexpected(b.error());
    }
    const std::string_view name{fn};
    geo::Polygon2D out;
    if (name == "geo_poly_union") {
        out = geo::poly_union(*a, *b);
    } else if (name == "geo_poly_intersect") {
        out = geo::poly_intersect(*a, *b);
    } else if (name == "geo_poly_diff") {
        out = geo::poly_diff(*a, *b);
    } else {
        return std::unexpected(DomainError{fn, "unknown geo polygon boolean"});
    }
    return points2d_to_matrix(out);
}

Result<Matrix<double>> eval_geo_minkowski_sum(const Matrix<double>& a_m,
                                              const Matrix<double>& b_m) {
    auto a = matrix_to_points2d(a_m, "geo_minkowski_sum");
    if (!a) {
        return std::unexpected(a.error());
    }
    auto b = matrix_to_points2d(b_m, "geo_minkowski_sum");
    if (!b) {
        return std::unexpected(b.error());
    }
    return points2d_to_matrix(geo::minkowski_sum_convex(*a, *b));
}

Result<Matrix<double>> eval_geo_clip_polygon(const Matrix<double>& subject_m,
                                             const Matrix<double>& window_m) {
    auto subject = matrix_to_points2d(subject_m, "geo_clip_polygon");
    if (!subject) {
        return std::unexpected(subject.error());
    }
    auto window = matrix_to_points2d(window_m, "geo_clip_polygon");
    if (!window) {
        return std::unexpected(window.error());
    }
    return points2d_to_matrix(geo::clip_polygon(*subject, *window));
}

// Returns 5x1 [cx; cy; width; height; angle_rad] for the minimum-area oriented bounding rect.
Result<Matrix<double>> eval_geo_min_bounding_rect(const Matrix<double>& P_m) {
    auto pts = matrix_to_points2d(P_m, "geo_min_bounding_rect");
    if (!pts) {
        return std::unexpected(pts.error());
    }
    if (pts->empty()) {
        return std::unexpected(
            DomainError{"geo_min_bounding_rect", "expected non-empty Nx2 point matrix"});
    }
    const auto r = geo::min_bounding_rect(*pts);
    Matrix<double> out(5, 1);
    out(0, 0) = r.center.x;
    out(1, 0) = r.center.y;
    out(2, 0) = r.width;
    out(3, 0) = r.height;
    out(4, 0) = r.angle;
    return out;
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

Result<std::vector<int>> matrix_to_int_coeff_vector(const Matrix<double>& m, const char* fn) {
    auto coeffs = matrix_to_coeff_vector(m, fn);
    if (!coeffs) {
        return std::unexpected(coeffs.error());
    }
    std::vector<int> out;
    out.reserve(coeffs->size());
    for (const double entry : *coeffs) {
        if (entry < 0.0 || std::floor(entry) != entry) {
            return std::unexpected(DomainError{fn, "expected non-negative integer entries"});
        }
        out.push_back(static_cast<int>(entry));
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

Result<Matrix<double>> eval_combo_prev_comb(const Matrix<double>& v_m, int n) {
    auto v_vec = matrix_to_coeff_vector(v_m, "combo_prev_comb");
    if (!v_vec) {
        return std::unexpected(v_vec.error());
    }
    if (v_vec->empty()) {
        return std::unexpected(
            DomainError{"combo_prev_comb", "expected non-empty combination vector"});
    }
    if (n < 0) {
        return std::unexpected(
            DomainError{"combo_prev_comb", "expected non-negative integer n"});
    }
    std::vector<int> v;
    v.reserve(v_vec->size());
    for (const double entry : *v_vec) {
        if (entry < 0.0 || std::floor(entry) != entry) {
            return std::unexpected(
                DomainError{"combo_prev_comb", "expected non-negative integer entries"});
        }
        v.push_back(static_cast<int>(entry));
    }
    combo::prev_comb(v, n);
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

Result<Matrix<double>> eval_graph_connected_components(const Matrix<double>& adj_m) {
    auto G = graph_from_adjacency_undirected(adj_m, "graph_connected_components");
    if (!G) {
        return std::unexpected(G.error());
    }
    const auto comps = graph::connected_components(*G);
    if (comps.empty()) {
        return Matrix<double>(0, 1);
    }
    size_t max_sz = 0;
    for (const auto& comp : comps) {
        max_sz = std::max(max_sz, comp.size());
    }
    Matrix<double> out(comps.size(), max_sz);
    for (size_t r = 0; r < comps.size(); ++r) {
        for (size_t c = 0; c < max_sz; ++c) {
            out(r, c) = c < comps[r].size() ? static_cast<double>(comps[r][c]) : -1.0;
        }
    }
    return out;
}

Result<Matrix<double>> eval_graph_louvain(const Matrix<double>& adj_m) {
    auto G = graph_from_adjacency_undirected(adj_m, "graph_louvain");
    if (!G) {
        return std::unexpected(G.error());
    }
    const auto comms = graph::louvain(*G);
    if (comms.empty()) {
        return Matrix<double>(0, 1);
    }
    size_t max_sz = 0;
    for (const auto& comp : comms) {
        max_sz = std::max(max_sz, comp.size());
    }
    Matrix<double> out(comms.size(), max_sz);
    for (size_t r = 0; r < comms.size(); ++r) {
        for (size_t c = 0; c < max_sz; ++c) {
            out(r, c) = c < comms[r].size() ? static_cast<double>(comms[r][c]) : -1.0;
        }
    }
    return out;
}

Matrix<double> graph_edge_components_to_matrix(
    const std::vector<std::vector<graph::Edge>>& components) {
    if (components.empty()) {
        return Matrix<double>(0, 1);
    }
    size_t max_edges = 0;
    for (const auto& comp : components) {
        max_edges = std::max(max_edges, comp.size());
    }
    const size_t cols = max_edges * 3;
    Matrix<double> out(components.size(), cols);
    for (size_t r = 0; r < components.size(); ++r) {
        for (size_t e = 0; e < max_edges; ++e) {
            const size_t base = e * 3;
            if (e < components[r].size()) {
                out(r, base + 0) = static_cast<double>(components[r][e].from);
                out(r, base + 1) = static_cast<double>(components[r][e].to);
                out(r, base + 2) = components[r][e].weight;
            } else {
                out(r, base + 0) = -1.0;
                out(r, base + 1) = -1.0;
                out(r, base + 2) = -1.0;
            }
        }
    }
    return out;
}

Result<double> eval_graph_bipartite_match(const Matrix<double>& adj_m, int left_size) {
    auto G = graph_from_adjacency(adj_m, "graph_bipartite_match");
    if (!G) {
        return std::unexpected(G.error());
    }
    auto matched = graph::bipartite_match(*G, left_size);
    if (!matched) {
        return std::unexpected(matched.error());
    }
    return static_cast<double>(*matched);
}

Result<Matrix<double>> eval_graph_biconnected_components(const Matrix<double>& adj_m) {
    auto G = graph_from_adjacency_undirected(adj_m, "graph_biconnected_components");
    if (!G) {
        return std::unexpected(G.error());
    }
    return graph_edge_components_to_matrix(graph::biconnected_components(*G));
}

Result<Matrix<double>> eval_graph_eulerian_path(const Matrix<double>& adj_m) {
    auto G = graph_from_adjacency_undirected(adj_m, "graph_eulerian_path");
    if (!G) {
        return std::unexpected(G.error());
    }
    const auto res = graph::eulerian_path(*G);
    if (!res.has_path) {
        return std::unexpected(DomainError{"graph_eulerian_path", "no Eulerian path exists"});
    }
    return int_vector_to_column(res.path);
}

Result<double> eval_graph_is_isomorphic(const Matrix<double>& adj_a_m,
                                        const Matrix<double>& adj_b_m) {
    auto Ga = graph_from_adjacency_undirected(adj_a_m, "graph_is_isomorphic");
    if (!Ga) {
        return std::unexpected(Ga.error());
    }
    auto Gb = graph_from_adjacency_undirected(adj_b_m, "graph_is_isomorphic");
    if (!Gb) {
        return std::unexpected(Gb.error());
    }
    return graph::is_isomorphic(*Ga, *Gb) ? 1.0 : 0.0;
}

Result<Matrix<double>> eval_graph_hamiltonian_path(const Matrix<double>& adj_m) {
    auto G = graph_from_adjacency_undirected(adj_m, "graph_hamiltonian_path");
    if (!G) {
        return std::unexpected(G.error());
    }
    auto path = graph::hamiltonian_path(*G);
    if (!path) {
        return std::unexpected(path.error());
    }
    return int_vector_to_column(*path);
}

Result<Matrix<double>> eval_graph_tsp_heuristic(const Matrix<double>& dist_m) {
    auto dist = matrix_to_square_nested(dist_m, "graph_tsp_heuristic");
    if (!dist) {
        return std::unexpected(dist.error());
    }
    const auto res = graph::tsp_heuristic(*dist);
    return int_vector_to_column(res.tour);
}

Result<Matrix<double>> eval_graph_eigenvector_centrality(const Matrix<double>& adj_m) {
    auto G = graph_from_adjacency(adj_m, "graph_eigenvector_centrality");
    if (!G) {
        return std::unexpected(G.error());
    }
    return vector_to_column(graph::eigenvector_centrality(*G));
}

Result<Matrix<double>> eval_graph_katz_centrality(const Matrix<double>& adj_m) {
    auto G = graph_from_adjacency(adj_m, "graph_katz_centrality");
    if (!G) {
        return std::unexpected(G.error());
    }
    return vector_to_column(graph::katz_centrality(*G));
}

Result<Matrix<double>> eval_graph_adjacency_spectrum(const Matrix<double>& adj_m) {
    auto G = graph_from_adjacency(adj_m, "graph_adjacency_spectrum");
    if (!G) {
        return std::unexpected(G.error());
    }
    return vector_to_column(graph::adjacency_spectrum(*G));
}

Result<Matrix<double>> eval_graph_laplacian(const Matrix<double>& adj_m) {
    auto G = graph_from_adjacency(adj_m, "graph_laplacian");
    if (!G) {
        return std::unexpected(G.error());
    }
    return nested_to_matrix(graph::laplacian(*G));
}

Result<Matrix<double>> eval_graph_normalised_laplacian(const Matrix<double>& adj_m) {
    auto G = graph_from_adjacency(adj_m, "graph_normalised_laplacian");
    if (!G) {
        return std::unexpected(G.error());
    }
    return nested_to_matrix(graph::normalised_laplacian(*G));
}

Result<std::vector<std::vector<int>>> community_matrix_to_partition(const Matrix<double>& C,
                                                                     const char* fn) {
    std::vector<std::vector<int>> communities;
    communities.reserve(C.rows());
    for (size_t r = 0; r < C.rows(); ++r) {
        std::vector<int> group;
        group.reserve(C.cols());
        for (size_t c = 0; c < C.cols(); ++c) {
            const double v = C(r, c);
            if (v < 0.0) {
                continue; // louvain/scc style padding sentinel
            }
            const int idx = static_cast<int>(v);
            if (v != static_cast<double>(idx)) {
                return std::unexpected(
                    DomainError{fn, "expected integer vertex indices in community matrix"});
            }
            group.push_back(idx);
        }
        if (!group.empty()) {
            communities.push_back(std::move(group));
        }
    }
    return communities;
}

Result<double> eval_graph_modularity(const Matrix<double>& adj_m, const Matrix<double>& C) {
    auto G = graph_from_adjacency_undirected(adj_m, "graph_modularity");
    if (!G) {
        return std::unexpected(G.error());
    }
    auto communities = community_matrix_to_partition(C, "graph_modularity");
    if (!communities) {
        return std::unexpected(communities.error());
    }
    return graph::modularity(*G, *communities);
}

Result<Matrix<double>> eval_graph_eccentricity(const Matrix<double>& adj_m) {
    auto G = graph_from_adjacency(adj_m, "graph_eccentricity");
    if (!G) {
        return std::unexpected(G.error());
    }
    return int_vector_to_column(graph::eccentricity(*G));
}

Result<double> eval_graph_is_strongly_connected(const Matrix<double>& adj_m) {
    auto G = graph_from_adjacency(adj_m, "graph_is_strongly_connected");
    if (!G) {
        return std::unexpected(G.error());
    }
    return graph::is_strongly_connected(*G) ? 1.0 : 0.0;
}

Result<double> eval_graph_algebraic_connectivity(const Matrix<double>& adj_m) {
    auto G = graph_from_adjacency_undirected(adj_m, "graph_algebraic_connectivity");
    if (!G) {
        return std::unexpected(G.error());
    }
    return graph::algebraic_connectivity(*G);
}

Result<Matrix<double>> eval_graph_articulation_points(const Matrix<double>& adj_m) {
    auto G = graph_from_adjacency_undirected(adj_m, "graph_articulation_points");
    if (!G) {
        return std::unexpected(G.error());
    }
    return int_vector_to_column(graph::articulation_points(*G));
}

Result<double> eval_geo_hermite_x(double p0x, double p0y, double m0x, double m0y, double p1x,
                                  double p1y, double m1x, double m1y, double t) {
    const geo::Point2D p0{p0x, p0y};
    const geo::Vec2D m0{m0x, m0y};
    const geo::Point2D p1{p1x, p1y};
    const geo::Vec2D m1{m1x, m1y};
    return geo::hermite_curve(p0, m0, p1, m1, t).x;
}

Result<Matrix<double>> eval_geo_hermite_curve(double p0x, double p0y, double m0x, double m0y,
                                              double p1x, double p1y, double m1x, double m1y,
                                              double t) {
    const geo::Point2D p0{p0x, p0y};
    const geo::Vec2D m0{m0x, m0y};
    const geo::Point2D p1{p1x, p1y};
    const geo::Vec2D m1{m1x, m1y};
    const geo::Point2D p = geo::hermite_curve(p0, m0, p1, m1, t);
    Matrix<double> out(1, 2);
    out(0, 0) = p.x;
    out(0, 1) = p.y;
    return out;
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

Result<double> eval_stats_max_value(const Matrix<double>& x_m) {
    auto x = matrix_to_coeff_vector(x_m, "stats_max_value");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (x->empty()) {
        return std::unexpected(
            DomainError{"stats_max_value", "expected non-empty vector"});
    }
    return max_value(*x);
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

Result<Matrix<double>> eval_scharr(const Matrix<double>& m) {
    auto gray = matrix_to_gray_image(m);
    if (!gray) {
        return std::unexpected(gray.error());
    }
    return gray_image_to_matrix(image::scharr(*gray));
}

Result<Matrix<double>> eval_roberts(const Matrix<double>& m) {
    auto gray = matrix_to_gray_image(m);
    if (!gray) {
        return std::unexpected(gray.error());
    }
    return gray_image_to_matrix(image::roberts(*gray));
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

Result<Matrix<double>> eval_ifftshift(const Matrix<double>& S_m) {
    auto spec = matrix_to_complex_spectrum(S_m, "ifftshift");
    if (!spec) {
        return std::unexpected(spec.error());
    }
    const auto shifted = ifftshift(*spec);
    Matrix<double> out(shifted.size(), 2);
    for (size_t i = 0; i < shifted.size(); ++i) {
        out(i, 0) = shifted[i].real();
        out(i, 1) = shifted[i].imag();
    }
    return out;
}

Result<Matrix<double>> eval_fftfreq(size_t n, double d) {
    return vector_to_column(fftfreq(n, d));
}

Result<Matrix<double>> eval_rfftfreq(size_t n, double d) {
    return vector_to_column(rfftfreq(n, d));
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

Result<double> eval_poly_resultant(const Matrix<double>& p_m, const Matrix<double>& q_m) {
    auto p = matrix_to_coeff_vector(p_m, "poly_resultant");
    if (!p) {
        return std::unexpected(p.error());
    }
    auto q = matrix_to_coeff_vector(q_m, "poly_resultant");
    if (!q) {
        return std::unexpected(q.error());
    }
    if (p->empty() || q->empty()) {
        return std::unexpected(
            DomainError{"poly_resultant", "expected non-empty coefficient vectors"});
    }
    return poly::poly_resultant(*p, *q);
}

Result<double> eval_poly_discriminant(const Matrix<double>& coeffs_m) {
    auto coeffs = matrix_to_coeff_vector(coeffs_m, "poly_discriminant");
    if (!coeffs) {
        return std::unexpected(coeffs.error());
    }
    if (coeffs->empty()) {
        return std::unexpected(
            DomainError{"poly_discriminant", "expected non-empty coefficient vector"});
    }
    return poly::poly_discriminant(*coeffs);
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

Result<Matrix<double>> eval_signal_conv2(const Matrix<double>& A, const Matrix<double>& K) {
    return conv2(A, K);
}

Result<Matrix<double>> eval_signal_deconv(const Matrix<double>& y_m, const Matrix<double>& b_m) {
    auto y = matrix_to_coeff_vector(y_m, "signal_deconv");
    if (!y) {
        return std::unexpected(y.error());
    }
    auto b = matrix_to_coeff_vector(b_m, "signal_deconv");
    if (!b) {
        return std::unexpected(b.error());
    }
    if (y->empty() || b->empty()) {
        return std::unexpected(
            DomainError{"signal_deconv", "expected non-empty vectors"});
    }
    return vector_to_column(deconv(*y, *b));
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

Result<Matrix<double>> eval_pde_heat_1d_cn(const Matrix<double>& x0_m, double alpha, double dx,
                                            double dt, std::size_t steps) {
    auto x0 = matrix_to_coeff_vector(x0_m, "pde_heat_1d_cn");
    if (!x0) {
        return std::unexpected(x0.error());
    }
    const auto value = pde_heat_1d_cn(*x0, alpha, dx, dt, steps);
    if (value.u.empty() || value.t.empty()) {
        return std::unexpected(DomainError{"pde_heat_1d_cn", "invalid input"});
    }
    return vector_to_column(value.u.back());
}

Result<Matrix<double>> eval_pde_heat_2d_cn_adi(const Matrix<double>& u0_m, double alpha,
                                                double dx, double dy, double dt,
                                                std::size_t steps) {
    auto u0 = matrix_to_grid(u0_m, "pde_heat_2d_cn_adi");
    if (!u0) {
        return std::unexpected(u0.error());
    }
    const auto value = pde_heat_2d_cn_adi(*u0, alpha, dx, dy, dt, steps);
    if (value.u.empty() || value.t.empty()) {
        return std::unexpected(DomainError{"pde_heat_2d_cn_adi", "invalid input"});
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

Result<Matrix<double>> eval_pde_poisson_1d(const Matrix<double>& f_m, double dx, double ua,
                                          double ub) {
    auto f = matrix_to_coeff_vector(f_m, "pde_poisson_1d");
    if (!f) {
        return std::unexpected(f.error());
    }
    const auto value = pde_poisson_1d(*f, dx, ua, ub);
    if (value.u.empty()) {
        return std::unexpected(DomainError{
            "pde_poisson_1d", "invalid grid or input rejected by solver"});
    }
    return vector_to_column(value.u);
}

Result<Matrix<double>> eval_pde_laplace_2d(int nx, int ny, const Matrix<double>& boundary_m) {
    auto boundary = matrix_to_grid(boundary_m, "pde_laplace_2d");
    if (!boundary) {
        return std::unexpected(boundary.error());
    }
    const auto value = pde_laplace_2d(nx, ny, *boundary);
    if (value.u.empty()) {
        return std::unexpected(DomainError{
            "pde_laplace_2d", "invalid grid or input rejected by solver"});
    }
    return grid_to_matrix(value.u);
}

Result<Matrix<double>> eval_pde_helmholtz_2d(const Matrix<double>& f_m, double k, double dx,
                                            double dy,
                                            const Matrix<double>* g_m = nullptr) {
    auto f = matrix_to_grid(f_m, "pde_helmholtz_2d");
    if (!f) {
        return std::unexpected(f.error());
    }
    std::vector<std::vector<double>> g;
    if (g_m != nullptr) {
        auto g_grid = matrix_to_grid(*g_m, "pde_helmholtz_2d");
        if (!g_grid) {
            return std::unexpected(g_grid.error());
        }
        g = std::move(*g_grid);
    }
    const auto value = pde_helmholtz_2d(*f, k, dx, dy, g);
    if (value.u.empty()) {
        return std::unexpected(DomainError{
            "pde_helmholtz_2d", "invalid grid or input rejected by solver"});
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

Result<Matrix<double>> eval_pde_wave_2d(const Matrix<double>& u0_m, const Matrix<double>& v0_m,
                                        double c, double dx, double dy, double dt,
                                        std::size_t steps) {
    auto u0 = matrix_to_grid(u0_m, "pde_wave_2d");
    if (!u0) {
        return std::unexpected(u0.error());
    }
    auto v0 = matrix_to_grid(v0_m, "pde_wave_2d");
    if (!v0) {
        return std::unexpected(v0.error());
    }
    if (u0->size() != v0->size() ||
        (!u0->empty() && u0->front().size() != v0->front().size())) {
        return std::unexpected(
            DomainError{"pde_wave_2d", "u0 and v0 grid dimension mismatch"});
    }
    const auto value = pde_wave_2d(*u0, *v0, c, dx, dy, dt, steps);
    if (value.u.empty() || value.t.empty()) {
        return std::unexpected(DomainError{
            "pde_wave_2d", "CFL stability condition violated or invalid input"});
    }
    return grid_to_matrix(value.u.back());
}

Result<Matrix<double>> eval_pde_advection_1d_lax_wendroff(const Matrix<double>& u0_m, double v,
                                                          double dx, double dt,
                                                          std::size_t steps) {
    auto u0 = matrix_to_coeff_vector(u0_m, "pde_advection_1d_lax_wendroff");
    if (!u0) {
        return std::unexpected(u0.error());
    }
    const auto value = pde_advection_1d_lax_wendroff(*u0, v, dx, dt, steps);
    if (value.u.empty() || value.t.empty()) {
        return std::unexpected(DomainError{
            "pde_advection_1d_lax_wendroff", "CFL stability condition violated or invalid input"});
    }
    return vector_to_column(value.u.back());
}

Result<Matrix<double>> eval_pde_reaction_diffusion_1d(const Matrix<double>& u0_m, double D,
                                                    double r, double dx, double dt,
                                                    std::size_t steps) {
    auto u0 = matrix_to_coeff_vector(u0_m, "pde_reaction_diffusion_1d");
    if (!u0) {
        return std::unexpected(u0.error());
    }
    const auto value = pde_reaction_diffusion_1d(*u0, D, r, dx, dt, steps);
    if (value.u.empty() || value.t.empty()) {
        return std::unexpected(DomainError{
            "pde_reaction_diffusion_1d", "stability condition violated or invalid input"});
    }
    return vector_to_column(value.u.back());
}

int hex_nibble(char c) {
    if (c >= '0' && c <= '9') {
        return c - '0';
    }
    if (c >= 'a' && c <= 'f') {
        return c - 'a' + 10;
    }
    if (c >= 'A' && c <= 'F') {
        return c - 'A' + 10;
    }
    return -1;
}

Result<std::vector<uint8_t>> parse_hex_arg(const std::string& text, const char* fn,
                                           const char* arg_name) {
    std::string hex;
    if (!parse_quoted_string(text, hex)) {
        hex = trim_copy(text);
    }
    // Empty hex is valid (e.g. empty AAD for AES-GCM); odd length is not.
    if (hex.size() % 2 != 0) {
        return std::unexpected(
            DomainError{fn, std::string("invalid hex for ") + arg_name});
    }
    std::vector<uint8_t> out;
    out.reserve(hex.size() / 2);
    for (std::size_t i = 0; i + 1 < hex.size(); i += 2) {
        const int hi = hex_nibble(hex[i]);
        const int lo = hex_nibble(hex[i + 1]);
        if (hi < 0 || lo < 0) {
            return std::unexpected(
                DomainError{fn, std::string("invalid hex for ") + arg_name});
        }
        out.push_back(static_cast<uint8_t>((hi << 4) | lo));
    }
    return out;
}

Result<std::string> eval_crypto_aes128_encrypt_block(const std::string& key_arg,
                                                     const std::string& block_arg) {
    constexpr const char* fn = "crypto_aes128_encrypt_block";
    auto key = parse_hex_arg(key_arg, fn, "key");
    if (!key) {
        return std::unexpected(key.error());
    }
    auto block = parse_hex_arg(block_arg, fn, "block");
    if (!block) {
        return std::unexpected(block.error());
    }
    if (key->size() != crypto::aes128_key_size) {
        return std::unexpected(
            DomainError{fn, "expected 16-byte (32 hex char) AES-128 key"});
    }
    if (block->size() != crypto::aes_block_size) {
        return std::unexpected(
            DomainError{fn, "expected 16-byte (32 hex char) block"});
    }
    return crypto::to_hex(crypto::aes128_encrypt_block(*key, *block)) + "\n";
}

Result<std::string> eval_crypto_aes128_decrypt_block(const std::string& key_arg,
                                                     const std::string& block_arg) {
    constexpr const char* fn = "crypto_aes128_decrypt_block";
    auto key = parse_hex_arg(key_arg, fn, "key");
    if (!key) {
        return std::unexpected(key.error());
    }
    auto block = parse_hex_arg(block_arg, fn, "block");
    if (!block) {
        return std::unexpected(block.error());
    }
    if (key->size() != crypto::aes128_key_size) {
        return std::unexpected(
            DomainError{fn, "expected 16-byte (32 hex char) AES-128 key"});
    }
    if (block->size() != crypto::aes_block_size) {
        return std::unexpected(
            DomainError{fn, "expected 16-byte (32 hex char) block"});
    }
    return crypto::to_hex(crypto::aes128_decrypt_block(*key, *block)) + "\n";
}

Result<std::string> eval_crypto_aes256_encrypt_block(const std::string& key_arg,
                                                     const std::string& block_arg) {
    constexpr const char* fn = "crypto_aes256_encrypt_block";
    auto key = parse_hex_arg(key_arg, fn, "key");
    if (!key) {
        return std::unexpected(key.error());
    }
    auto block = parse_hex_arg(block_arg, fn, "block");
    if (!block) {
        return std::unexpected(block.error());
    }
    if (key->size() != crypto::aes256_key_size) {
        return std::unexpected(
            DomainError{fn, "expected 32-byte (64 hex char) AES-256 key"});
    }
    if (block->size() != crypto::aes_block_size) {
        return std::unexpected(
            DomainError{fn, "expected 16-byte (32 hex char) block"});
    }
    return crypto::to_hex(crypto::aes256_encrypt_block(*key, *block)) + "\n";
}

Result<std::string> eval_crypto_aes256_decrypt_block(const std::string& key_arg,
                                                     const std::string& block_arg) {
    constexpr const char* fn = "crypto_aes256_decrypt_block";
    auto key = parse_hex_arg(key_arg, fn, "key");
    if (!key) {
        return std::unexpected(key.error());
    }
    auto block = parse_hex_arg(block_arg, fn, "block");
    if (!block) {
        return std::unexpected(block.error());
    }
    if (key->size() != crypto::aes256_key_size) {
        return std::unexpected(
            DomainError{fn, "expected 32-byte (64 hex char) AES-256 key"});
    }
    if (block->size() != crypto::aes_block_size) {
        return std::unexpected(
            DomainError{fn, "expected 16-byte (32 hex char) block"});
    }
    return crypto::to_hex(crypto::aes256_decrypt_block(*key, *block)) + "\n";
}

Result<std::string> eval_crypto_aes128_cbc_encrypt(const std::string& key_arg,
                                                   const std::string& iv_arg,
                                                   const std::string& plain_arg) {
    constexpr const char* fn = "crypto_aes128_cbc_encrypt";
    auto key = parse_hex_arg(key_arg, fn, "key");
    if (!key) {
        return std::unexpected(key.error());
    }
    auto iv = parse_hex_arg(iv_arg, fn, "iv");
    if (!iv) {
        return std::unexpected(iv.error());
    }
    auto plain = parse_hex_arg(plain_arg, fn, "plaintext");
    if (!plain) {
        return std::unexpected(plain.error());
    }
    if (key->size() != crypto::aes128_key_size) {
        return std::unexpected(
            DomainError{fn, "expected 16-byte (32 hex char) AES-128 key"});
    }
    if (iv->size() != crypto::aes_block_size) {
        return std::unexpected(DomainError{fn, "expected 16-byte (32 hex char) iv"});
    }
    if (plain->empty() || plain->size() % crypto::aes_block_size != 0) {
        return std::unexpected(
            DomainError{fn, "expected non-empty plaintext with length multiple of 16 bytes"});
    }
    return crypto::to_hex(crypto::aes128_cbc_encrypt(*key, *iv, *plain)) + "\n";
}

Result<std::string> eval_crypto_aes128_cbc_decrypt(const std::string& key_arg,
                                                   const std::string& iv_arg,
                                                   const std::string& cipher_arg) {
    constexpr const char* fn = "crypto_aes128_cbc_decrypt";
    auto key = parse_hex_arg(key_arg, fn, "key");
    if (!key) {
        return std::unexpected(key.error());
    }
    auto iv = parse_hex_arg(iv_arg, fn, "iv");
    if (!iv) {
        return std::unexpected(iv.error());
    }
    auto cipher = parse_hex_arg(cipher_arg, fn, "ciphertext");
    if (!cipher) {
        return std::unexpected(cipher.error());
    }
    if (key->size() != crypto::aes128_key_size) {
        return std::unexpected(
            DomainError{fn, "expected 16-byte (32 hex char) AES-128 key"});
    }
    if (iv->size() != crypto::aes_block_size) {
        return std::unexpected(DomainError{fn, "expected 16-byte (32 hex char) iv"});
    }
    if (cipher->empty() || cipher->size() % crypto::aes_block_size != 0) {
        return std::unexpected(
            DomainError{fn, "expected non-empty ciphertext with length multiple of 16 bytes"});
    }
    return crypto::to_hex(crypto::aes128_cbc_decrypt(*key, *iv, *cipher)) + "\n";
}

Result<std::string> eval_crypto_aes256_cbc_encrypt(const std::string& key_arg,
                                                   const std::string& iv_arg,
                                                   const std::string& plain_arg) {
    constexpr const char* fn = "crypto_aes256_cbc_encrypt";
    auto key = parse_hex_arg(key_arg, fn, "key");
    if (!key) {
        return std::unexpected(key.error());
    }
    auto iv = parse_hex_arg(iv_arg, fn, "iv");
    if (!iv) {
        return std::unexpected(iv.error());
    }
    auto plain = parse_hex_arg(plain_arg, fn, "plaintext");
    if (!plain) {
        return std::unexpected(plain.error());
    }
    if (key->size() != crypto::aes256_key_size) {
        return std::unexpected(
            DomainError{fn, "expected 32-byte (64 hex char) AES-256 key"});
    }
    if (iv->size() != crypto::aes_block_size) {
        return std::unexpected(DomainError{fn, "expected 16-byte (32 hex char) iv"});
    }
    if (plain->empty() || plain->size() % crypto::aes_block_size != 0) {
        return std::unexpected(
            DomainError{fn, "expected non-empty plaintext with length multiple of 16 bytes"});
    }
    return crypto::to_hex(crypto::aes256_cbc_encrypt(*key, *iv, *plain)) + "\n";
}

Result<std::string> eval_crypto_aes256_cbc_decrypt(const std::string& key_arg,
                                                   const std::string& iv_arg,
                                                   const std::string& cipher_arg) {
    constexpr const char* fn = "crypto_aes256_cbc_decrypt";
    auto key = parse_hex_arg(key_arg, fn, "key");
    if (!key) {
        return std::unexpected(key.error());
    }
    auto iv = parse_hex_arg(iv_arg, fn, "iv");
    if (!iv) {
        return std::unexpected(iv.error());
    }
    auto cipher = parse_hex_arg(cipher_arg, fn, "ciphertext");
    if (!cipher) {
        return std::unexpected(cipher.error());
    }
    if (key->size() != crypto::aes256_key_size) {
        return std::unexpected(
            DomainError{fn, "expected 32-byte (64 hex char) AES-256 key"});
    }
    if (iv->size() != crypto::aes_block_size) {
        return std::unexpected(DomainError{fn, "expected 16-byte (32 hex char) iv"});
    }
    if (cipher->empty() || cipher->size() % crypto::aes_block_size != 0) {
        return std::unexpected(
            DomainError{fn, "expected non-empty ciphertext with length multiple of 16 bytes"});
    }
    return crypto::to_hex(crypto::aes256_cbc_decrypt(*key, *iv, *cipher)) + "\n";
}

Result<std::string> eval_crypto_aes128_gcm_encrypt(const std::string& key_arg,
                                                   const std::string& iv_arg,
                                                   const std::string& aad_arg,
                                                   const std::string& plain_arg) {
    constexpr const char* fn = "crypto_aes128_gcm_encrypt";
    auto key = parse_hex_arg(key_arg, fn, "key");
    if (!key) {
        return std::unexpected(key.error());
    }
    auto iv = parse_hex_arg(iv_arg, fn, "iv");
    if (!iv) {
        return std::unexpected(iv.error());
    }
    auto aad = parse_hex_arg(aad_arg, fn, "aad");
    if (!aad) {
        return std::unexpected(aad.error());
    }
    auto plain = parse_hex_arg(plain_arg, fn, "plaintext");
    if (!plain) {
        return std::unexpected(plain.error());
    }
    if (key->size() != crypto::aes128_key_size) {
        return std::unexpected(
            DomainError{fn, "expected 16-byte (32 hex char) AES-128 key"});
    }
    if (iv->empty() || iv->size() > crypto::aes_gcm_iv_size) {
        return std::unexpected(
            DomainError{fn, "expected 1-12 byte iv (2-24 hex chars, zero-padded on right)"});
    }
    const auto seal = crypto::aes128_gcm_encrypt(*key, *iv, *aad, *plain);
    return crypto::to_hex(seal.ciphertext) + " " +
           crypto::to_hex(std::span<const uint8_t>(seal.tag.data(), seal.tag.size())) + "\n";
}

Result<std::string> eval_crypto_aes128_gcm_decrypt(const std::string& key_arg,
                                                   const std::string& iv_arg,
                                                   const std::string& aad_arg,
                                                   const std::string& cipher_arg,
                                                   const std::string& tag_arg) {
    constexpr const char* fn = "crypto_aes128_gcm_decrypt";
    auto key = parse_hex_arg(key_arg, fn, "key");
    if (!key) {
        return std::unexpected(key.error());
    }
    auto iv = parse_hex_arg(iv_arg, fn, "iv");
    if (!iv) {
        return std::unexpected(iv.error());
    }
    auto aad = parse_hex_arg(aad_arg, fn, "aad");
    if (!aad) {
        return std::unexpected(aad.error());
    }
    auto cipher = parse_hex_arg(cipher_arg, fn, "ciphertext");
    if (!cipher) {
        return std::unexpected(cipher.error());
    }
    auto tag = parse_hex_arg(tag_arg, fn, "tag");
    if (!tag) {
        return std::unexpected(tag.error());
    }
    if (key->size() != crypto::aes128_key_size) {
        return std::unexpected(
            DomainError{fn, "expected 16-byte (32 hex char) AES-128 key"});
    }
    if (iv->empty() || iv->size() > crypto::aes_gcm_iv_size) {
        return std::unexpected(
            DomainError{fn, "expected 1-12 byte iv (2-24 hex chars, zero-padded on right)"});
    }
    if (tag->size() != crypto::aes_gcm_tag_size) {
        return std::unexpected(
            DomainError{fn, "expected 16-byte (32 hex char) authentication tag"});
    }
    const auto plain = crypto::aes128_gcm_decrypt(*key, *iv, *aad, *cipher, *tag);
    if (plain.empty() && !cipher->empty()) {
        return std::unexpected(DomainError{fn, "authentication failed or invalid inputs"});
    }
    return crypto::to_hex(plain) + "\n";
}

Result<std::string> eval_crypto_aes256_gcm_encrypt(const std::string& key_arg,
                                                   const std::string& iv_arg,
                                                   const std::string& aad_arg,
                                                   const std::string& plain_arg) {
    constexpr const char* fn = "crypto_aes256_gcm_encrypt";
    auto key = parse_hex_arg(key_arg, fn, "key");
    if (!key) {
        return std::unexpected(key.error());
    }
    auto iv = parse_hex_arg(iv_arg, fn, "iv");
    if (!iv) {
        return std::unexpected(iv.error());
    }
    auto aad = parse_hex_arg(aad_arg, fn, "aad");
    if (!aad) {
        return std::unexpected(aad.error());
    }
    auto plain = parse_hex_arg(plain_arg, fn, "plaintext");
    if (!plain) {
        return std::unexpected(plain.error());
    }
    if (key->size() != crypto::aes256_key_size) {
        return std::unexpected(
            DomainError{fn, "expected 32-byte (64 hex char) AES-256 key"});
    }
    if (iv->empty() || iv->size() > crypto::aes_gcm_iv_size) {
        return std::unexpected(
            DomainError{fn, "expected 1-12 byte iv (2-24 hex chars, zero-padded on right)"});
    }
    const auto seal = crypto::aes256_gcm_encrypt(*key, *iv, *aad, *plain);
    return crypto::to_hex(seal.ciphertext) + " " +
           crypto::to_hex(std::span<const uint8_t>(seal.tag.data(), seal.tag.size())) + "\n";
}

Result<std::string> eval_crypto_aes256_gcm_decrypt(const std::string& key_arg,
                                                   const std::string& iv_arg,
                                                   const std::string& aad_arg,
                                                   const std::string& cipher_arg,
                                                   const std::string& tag_arg) {
    constexpr const char* fn = "crypto_aes256_gcm_decrypt";
    auto key = parse_hex_arg(key_arg, fn, "key");
    if (!key) {
        return std::unexpected(key.error());
    }
    auto iv = parse_hex_arg(iv_arg, fn, "iv");
    if (!iv) {
        return std::unexpected(iv.error());
    }
    auto aad = parse_hex_arg(aad_arg, fn, "aad");
    if (!aad) {
        return std::unexpected(aad.error());
    }
    auto cipher = parse_hex_arg(cipher_arg, fn, "ciphertext");
    if (!cipher) {
        return std::unexpected(cipher.error());
    }
    auto tag = parse_hex_arg(tag_arg, fn, "tag");
    if (!tag) {
        return std::unexpected(tag.error());
    }
    if (key->size() != crypto::aes256_key_size) {
        return std::unexpected(
            DomainError{fn, "expected 32-byte (64 hex char) AES-256 key"});
    }
    if (iv->empty() || iv->size() > crypto::aes_gcm_iv_size) {
        return std::unexpected(
            DomainError{fn, "expected 1-12 byte iv (2-24 hex chars, zero-padded on right)"});
    }
    if (tag->size() != crypto::aes_gcm_tag_size) {
        return std::unexpected(
            DomainError{fn, "expected 16-byte (32 hex char) authentication tag"});
    }
    const auto plain = crypto::aes256_gcm_decrypt(*key, *iv, *aad, *cipher, *tag);
    if (plain.empty() && !cipher->empty()) {
        return std::unexpected(DomainError{fn, "authentication failed or invalid inputs"});
    }
    return crypto::to_hex(plain) + "\n";
}

Result<std::string> eval_crypto_chacha20(const std::string& key_arg, const std::string& nonce_arg,
                                         const std::string& counter_arg,
                                         const std::string& data_arg) {
    constexpr const char* fn = "crypto_chacha20";
    auto key_bytes = parse_hex_arg(key_arg, fn, "key");
    if (!key_bytes) {
        return std::unexpected(key_bytes.error());
    }
    auto nonce_bytes = parse_hex_arg(nonce_arg, fn, "nonce");
    if (!nonce_bytes) {
        return std::unexpected(nonce_bytes.error());
    }
    auto data = parse_hex_arg(data_arg, fn, "data");
    if (!data) {
        return std::unexpected(data.error());
    }
    if (key_bytes->size() != 32) {
        return std::unexpected(DomainError{fn, "expected 32-byte (64 hex char) key"});
    }
    if (nonce_bytes->size() != 12) {
        if (nonce_bytes->size() == 10) {
            nonce_bytes->insert(nonce_bytes->end(), 2, 0x00);
        } else {
            return std::unexpected(DomainError{fn, "expected 12-byte (24 hex char) nonce"});
        }
    }
    double counter_d = 0.0;
    if (!parse_number(trim_copy(counter_arg), counter_d)) {
        return std::unexpected(DomainError{fn, "expected numeric counter"});
    }
    const auto counter_i = static_cast<std::uint32_t>(counter_d);
    if (counter_d < 0.0 || counter_d != static_cast<double>(counter_i)) {
        return std::unexpected(DomainError{fn, "expected non-negative integer counter"});
    }
    std::array<uint8_t, 32> key{};
    std::array<uint8_t, 12> nonce{};
    std::copy(key_bytes->begin(), key_bytes->end(), key.begin());
    std::copy(nonce_bytes->begin(), nonce_bytes->end(), nonce.begin());
    return crypto::to_hex(crypto::chacha20_encrypt(key, nonce, counter_i, *data)) + "\n";
}

Result<std::string> eval_crypto_chacha20_poly1305_encrypt(const std::string& key_arg,
                                                          const std::string& nonce_arg,
                                                          const std::string& aad_arg,
                                                          const std::string& plain_arg) {
    constexpr const char* fn = "crypto_chacha20_poly1305_encrypt";
    auto key_bytes = parse_hex_arg(key_arg, fn, "key");
    if (!key_bytes) {
        return std::unexpected(key_bytes.error());
    }
    auto nonce_bytes = parse_hex_arg(nonce_arg, fn, "nonce");
    if (!nonce_bytes) {
        return std::unexpected(nonce_bytes.error());
    }
    auto aad = parse_hex_arg(aad_arg, fn, "aad");
    if (!aad) {
        return std::unexpected(aad.error());
    }
    auto plain = parse_hex_arg(plain_arg, fn, "plaintext");
    if (!plain) {
        return std::unexpected(plain.error());
    }
    if (key_bytes->size() != 32) {
        return std::unexpected(DomainError{fn, "expected 32-byte (64 hex char) key"});
    }
    if (nonce_bytes->size() != crypto::chacha20_poly1305_nonce_size) {
        return std::unexpected(DomainError{fn, "expected 12-byte (24 hex char) nonce"});
    }
    std::array<uint8_t, 32> key{};
    std::array<uint8_t, 12> nonce{};
    std::copy(key_bytes->begin(), key_bytes->end(), key.begin());
    std::copy(nonce_bytes->begin(), nonce_bytes->end(), nonce.begin());
    const auto seal = crypto::chacha20_poly1305_encrypt(key, nonce, *aad, *plain);
    return crypto::to_hex(seal.ciphertext) + " " +
           crypto::to_hex(std::span<const uint8_t>(seal.tag.data(), seal.tag.size())) + "\n";
}

Result<std::string> eval_crypto_chacha20_poly1305_decrypt(const std::string& key_arg,
                                                          const std::string& nonce_arg,
                                                          const std::string& aad_arg,
                                                          const std::string& cipher_arg,
                                                          const std::string& tag_arg) {
    constexpr const char* fn = "crypto_chacha20_poly1305_decrypt";
    auto key_bytes = parse_hex_arg(key_arg, fn, "key");
    if (!key_bytes) {
        return std::unexpected(key_bytes.error());
    }
    auto nonce_bytes = parse_hex_arg(nonce_arg, fn, "nonce");
    if (!nonce_bytes) {
        return std::unexpected(nonce_bytes.error());
    }
    auto aad = parse_hex_arg(aad_arg, fn, "aad");
    if (!aad) {
        return std::unexpected(aad.error());
    }
    auto cipher = parse_hex_arg(cipher_arg, fn, "ciphertext");
    if (!cipher) {
        return std::unexpected(cipher.error());
    }
    auto tag = parse_hex_arg(tag_arg, fn, "tag");
    if (!tag) {
        return std::unexpected(tag.error());
    }
    if (key_bytes->size() != 32) {
        return std::unexpected(DomainError{fn, "expected 32-byte (64 hex char) key"});
    }
    if (nonce_bytes->size() != crypto::chacha20_poly1305_nonce_size) {
        return std::unexpected(DomainError{fn, "expected 12-byte (24 hex char) nonce"});
    }
    if (tag->size() != crypto::chacha20_poly1305_tag_size) {
        return std::unexpected(
            DomainError{fn, "expected 16-byte (32 hex char) authentication tag"});
    }
    std::array<uint8_t, 32> key{};
    std::array<uint8_t, 12> nonce{};
    std::copy(key_bytes->begin(), key_bytes->end(), key.begin());
    std::copy(nonce_bytes->begin(), nonce_bytes->end(), nonce.begin());
    const auto plain = crypto::chacha20_poly1305_decrypt(key, nonce, *aad, *cipher, *tag);
    if (plain.empty() && !cipher->empty()) {
        return std::unexpected(DomainError{fn, "authentication failed or invalid inputs"});
    }
    return crypto::to_hex(plain) + "\n";
}

Result<std::vector<uint8_t>> parse_hex_arg_allow_empty(const std::string& text, const char* fn,
                                                      const char* arg_name) {
    std::string hex;
    if (!parse_quoted_string(text, hex)) {
        hex = trim_copy(text);
    }
    if (hex.empty()) {
        return std::vector<uint8_t>{};
    }
    return parse_hex_arg(text, fn, arg_name);
}

Result<std::string> eval_crypto_sha256(const std::string& data_arg) {
    constexpr const char* fn = "crypto_sha256";
    auto data = parse_hex_arg(data_arg, fn, "data");
    if (!data) {
        return std::unexpected(data.error());
    }
    return crypto::sha256_hex(std::span<const uint8_t>(*data)) + "\n";
}

Result<std::string> eval_crypto_to_hex(const std::string& data_arg) {
    constexpr const char* fn = "crypto_to_hex";
    auto data = parse_hex_arg(data_arg, fn, "data");
    if (!data) {
        return std::unexpected(data.error());
    }
    return crypto::to_hex(std::span<const uint8_t>(*data)) + "\n";
}

Result<Matrix<double>> eval_crypto_from_hex(const std::string& hex_arg) {
    constexpr const char* fn = "crypto_from_hex";
    auto bytes = parse_hex_arg(hex_arg, fn, "hex");
    if (!bytes) {
        return std::unexpected(bytes.error());
    }
    return bytes_to_matrix_col(compress::Bytes(std::move(*bytes)));
}

Result<std::string> eval_crypto_bytes_to_hex(const Matrix<double>& bytes_m) {
    constexpr const char* fn = "crypto_bytes_to_hex";
    auto bytes = matrix_col_to_bytes(bytes_m, fn);
    if (!bytes) {
        return std::unexpected(bytes.error());
    }
    return crypto::to_hex(std::span<const uint8_t>(*bytes)) + "\n";
}

Result<Matrix<double>> eval_crypto_bytes_to_hex_vec(const Matrix<double>& bytes_m) {
    auto hex = eval_crypto_bytes_to_hex(bytes_m);
    if (!hex) {
        return std::unexpected(hex.error());
    }
    std::string trimmed = trim_copy(*hex);
    compress::Bytes ascii(trimmed.begin(), trimmed.end());
    return bytes_to_matrix_col(ascii);
}

Result<std::string> eval_crypto_sha512(const std::string& data_arg) {
    constexpr const char* fn = "crypto_sha512";
    auto data = parse_hex_arg(data_arg, fn, "data");
    if (!data) {
        return std::unexpected(data.error());
    }
    return crypto::sha512_hex(std::span<const uint8_t>(*data)) + "\n";
}

Result<std::string> eval_crypto_hmac_sha256(const std::string& key_arg,
                                            const std::string& data_arg) {
    constexpr const char* fn = "crypto_hmac_sha256";
    auto key = parse_hex_arg(key_arg, fn, "key");
    if (!key) {
        return std::unexpected(key.error());
    }
    auto data = parse_hex_arg(data_arg, fn, "data");
    if (!data) {
        return std::unexpected(data.error());
    }
    if (key->empty()) {
        return std::unexpected(DomainError{fn, "key must not be empty"});
    }
    return crypto::hmac_sha256_hex(std::span<const uint8_t>(*key),
                                   std::span<const uint8_t>(*data)) +
           "\n";
}

Result<std::string> eval_crypto_hmac_sha512(const std::string& key_arg,
                                            const std::string& data_arg) {
    constexpr const char* fn = "crypto_hmac_sha512";
    auto key = parse_hex_arg(key_arg, fn, "key");
    if (!key) {
        return std::unexpected(key.error());
    }
    auto data = parse_hex_arg(data_arg, fn, "data");
    if (!data) {
        return std::unexpected(data.error());
    }
    if (key->empty()) {
        return std::unexpected(DomainError{fn, "key must not be empty"});
    }
    return crypto::hmac_sha512_hex(std::span<const uint8_t>(*key),
                                   std::span<const uint8_t>(*data)) +
           "\n";
}

Result<std::string> eval_crypto_hkdf_sha256(const std::string& ikm_arg,
                                            const std::string& salt_arg,
                                            const std::string& info_arg,
                                            const std::string& len_arg) {
    constexpr const char* fn = "crypto_hkdf_sha256";
    auto ikm = parse_hex_arg(ikm_arg, fn, "ikm");
    if (!ikm) {
        return std::unexpected(ikm.error());
    }
    auto salt = parse_hex_arg_allow_empty(salt_arg, fn, "salt");
    if (!salt) {
        return std::unexpected(salt.error());
    }
    auto info = parse_hex_arg_allow_empty(info_arg, fn, "info");
    if (!info) {
        return std::unexpected(info.error());
    }
    double len_d = 0.0;
    if (!parse_number(trim_copy(len_arg), len_d)) {
        return std::unexpected(DomainError{fn, "expected numeric output length"});
    }
    const auto len_i = static_cast<std::size_t>(len_d);
    if (len_d < 0.0 || len_d != static_cast<double>(len_i)) {
        return std::unexpected(DomainError{fn, "expected non-negative integer output length"});
    }
    if (len_i > 255 * crypto::sha256_digest_size) {
        return std::unexpected(
            DomainError{fn, "output length exceeds RFC 5869 maximum (255 * 32 bytes)"});
    }
    if (ikm->empty()) {
        return std::unexpected(DomainError{fn, "ikm must not be empty"});
    }
    const auto okm = crypto::hkdf_sha256(*ikm, *salt, *info, len_i);
    if (okm.size() != len_i) {
        return std::unexpected(DomainError{fn, "HKDF expansion failed"});
    }
    return crypto::to_hex(okm) + "\n";
}

Result<std::string> eval_crypto_hkdf_sha512(const std::string& ikm_arg,
                                            const std::string& salt_arg,
                                            const std::string& info_arg,
                                            const std::string& len_arg) {
    constexpr const char* fn = "crypto_hkdf_sha512";
    auto ikm = parse_hex_arg(ikm_arg, fn, "ikm");
    if (!ikm) {
        return std::unexpected(ikm.error());
    }
    auto salt = parse_hex_arg_allow_empty(salt_arg, fn, "salt");
    if (!salt) {
        return std::unexpected(salt.error());
    }
    auto info = parse_hex_arg_allow_empty(info_arg, fn, "info");
    if (!info) {
        return std::unexpected(info.error());
    }
    double len_d = 0.0;
    if (!parse_number(trim_copy(len_arg), len_d)) {
        return std::unexpected(DomainError{fn, "expected numeric output length"});
    }
    const auto len_i = static_cast<std::size_t>(len_d);
    if (len_d < 0.0 || len_d != static_cast<double>(len_i)) {
        return std::unexpected(DomainError{fn, "expected non-negative integer output length"});
    }
    if (len_i > 255 * crypto::sha512_digest_size) {
        return std::unexpected(
            DomainError{fn, "output length exceeds RFC 5869 maximum (255 * 64 bytes)"});
    }
    if (ikm->empty()) {
        return std::unexpected(DomainError{fn, "ikm must not be empty"});
    }
    const auto okm = crypto::hkdf_sha512(*ikm, *salt, *info, len_i);
    if (okm.size() != len_i) {
        return std::unexpected(DomainError{fn, "HKDF expansion failed"});
    }
    return crypto::to_hex(okm) + "\n";
}

Result<std::string> eval_crypto_pbkdf2_sha256(const std::string& pass_arg,
                                              const std::string& salt_arg,
                                              const std::string& iter_arg,
                                              const std::string& dklen_arg) {
    constexpr const char* fn = "crypto_pbkdf2_sha256";
    auto password = parse_hex_arg(pass_arg, fn, "password");
    if (!password) {
        return std::unexpected(password.error());
    }
    auto salt = parse_hex_arg(salt_arg, fn, "salt");
    if (!salt) {
        return std::unexpected(salt.error());
    }
    double iter_d = 0.0;
    if (!parse_number(trim_copy(iter_arg), iter_d)) {
        return std::unexpected(DomainError{fn, "expected numeric iteration count"});
    }
    const auto iter_i = static_cast<std::uint32_t>(iter_d);
    if (iter_d < 0.0 || iter_d != static_cast<double>(iter_i) || iter_i == 0) {
        return std::unexpected(DomainError{fn, "expected positive integer iteration count"});
    }
    double dklen_d = 0.0;
    if (!parse_number(trim_copy(dklen_arg), dklen_d)) {
        return std::unexpected(DomainError{fn, "expected numeric derived key length"});
    }
    const auto dklen_i = static_cast<std::size_t>(dklen_d);
    if (dklen_d < 0.0 || dklen_d != static_cast<double>(dklen_i)) {
        return std::unexpected(DomainError{fn, "expected non-negative integer derived key length"});
    }
    if (password->empty()) {
        return std::unexpected(DomainError{fn, "password must not be empty"});
    }
    if (salt->empty()) {
        return std::unexpected(DomainError{fn, "salt must not be empty"});
    }
    const auto dk = crypto::pbkdf2_hmac_sha256(*password, *salt, iter_i, dklen_i);
    if (dk.size() != dklen_i) {
        return std::unexpected(DomainError{fn, "PBKDF2 derivation failed"});
    }
    return crypto::to_hex(dk) + "\n";
}

Result<std::string> eval_crypto_pbkdf2_hmac_sha512(const std::string& pass_arg,
                                                     const std::string& salt_arg,
                                                     const std::string& iter_arg,
                                                     const std::string& dklen_arg) {
    constexpr const char* fn = "crypto_pbkdf2_hmac_sha512";
    auto password = parse_hex_arg(pass_arg, fn, "password");
    if (!password) {
        return std::unexpected(password.error());
    }
    auto salt = parse_hex_arg(salt_arg, fn, "salt");
    if (!salt) {
        return std::unexpected(salt.error());
    }
    double iter_d = 0.0;
    if (!parse_number(trim_copy(iter_arg), iter_d)) {
        return std::unexpected(DomainError{fn, "expected numeric iteration count"});
    }
    const auto iter_i = static_cast<std::uint32_t>(iter_d);
    if (iter_d < 0.0 || iter_d != static_cast<double>(iter_i) || iter_i == 0) {
        return std::unexpected(DomainError{fn, "expected positive integer iteration count"});
    }
    double dklen_d = 0.0;
    if (!parse_number(trim_copy(dklen_arg), dklen_d)) {
        return std::unexpected(DomainError{fn, "expected numeric derived key length"});
    }
    const auto dklen_i = static_cast<std::size_t>(dklen_d);
    if (dklen_d < 0.0 || dklen_d != static_cast<double>(dklen_i)) {
        return std::unexpected(DomainError{fn, "expected non-negative integer derived key length"});
    }
    if (password->empty()) {
        return std::unexpected(DomainError{fn, "password must not be empty"});
    }
    if (salt->empty()) {
        return std::unexpected(DomainError{fn, "salt must not be empty"});
    }
    const auto dk = crypto::pbkdf2_hmac_sha512(*password, *salt, iter_i, dklen_i);
    if (dk.size() != dklen_i) {
        return std::unexpected(DomainError{fn, "PBKDF2 derivation failed"});
    }
    return crypto::to_hex(dk) + "\n";
}

Result<std::string> eval_crypto_x25519_keypair(const std::string& priv_arg) {
    constexpr const char* fn = "crypto_x25519_keypair";
    auto priv = parse_hex_arg(priv_arg, fn, "private_key");
    if (!priv) {
        return std::unexpected(priv.error());
    }
    if (priv->size() != crypto::x25519_key_size) {
        return std::unexpected(DomainError{fn, "expected 32-byte (64 hex char) private key"});
    }
    const auto kp = crypto::x25519_keypair(*priv);
    if (kp.public_key == std::array<uint8_t, crypto::x25519_key_size>{}) {
        return std::unexpected(DomainError{fn, "key generation failed"});
    }
    return crypto::to_hex(std::span<const uint8_t>(kp.public_key.data(), kp.public_key.size())) +
           "\n";
}

Result<std::string> eval_crypto_x25519_shared(const std::string& priv_arg,
                                              const std::string& pub_arg) {
    constexpr const char* fn = "crypto_x25519_shared";
    auto priv = parse_hex_arg(priv_arg, fn, "private_key");
    if (!priv) {
        return std::unexpected(priv.error());
    }
    auto pub = parse_hex_arg(pub_arg, fn, "public_key");
    if (!pub) {
        return std::unexpected(pub.error());
    }
    if (priv->size() != crypto::x25519_key_size) {
        return std::unexpected(DomainError{fn, "expected 32-byte (64 hex char) private key"});
    }
    if (pub->size() != crypto::x25519_key_size) {
        return std::unexpected(DomainError{fn, "expected 32-byte (64 hex char) public key"});
    }
    const auto shared = crypto::x25519_shared_secret(*priv, *pub);
    return crypto::to_hex(std::span<const uint8_t>(shared.data(), shared.size())) + "\n";
}

Result<std::string> eval_crypto_ed25519_keypair(const std::string& seed_arg) {
    constexpr const char* fn = "crypto_ed25519_keypair";
    auto seed = parse_hex_arg(seed_arg, fn, "seed");
    if (!seed) {
        return std::unexpected(seed.error());
    }
    if (seed->size() != crypto::ed25519_seed_size) {
        return std::unexpected(DomainError{fn, "expected 32-byte (64 hex char) seed"});
    }
    const auto kp = crypto::ed25519_keypair(*seed);
    if (kp.public_key == std::array<uint8_t, crypto::ed25519_public_key_size>{}) {
        return std::unexpected(DomainError{fn, "key generation failed"});
    }
    return crypto::to_hex(std::span<const uint8_t>(kp.public_key.data(), kp.public_key.size())) +
           "\n";
}

Result<std::string> eval_crypto_ed25519_sign(const std::string& secret_arg,
                                              const std::string& msg_arg) {
    constexpr const char* fn = "crypto_ed25519_sign";
    auto secret = parse_hex_arg(secret_arg, fn, "secret");
    if (!secret) {
        return std::unexpected(secret.error());
    }
    auto msg = parse_hex_arg(msg_arg, fn, "message");
    if (!msg) {
        return std::unexpected(msg.error());
    }
    if (secret->size() != crypto::ed25519_seed_size &&
        secret->size() != crypto::ed25519_secret_key_size) {
        return std::unexpected(
            DomainError{fn, "expected 32-byte seed or 64-byte expanded secret key (hex)"});
    }
    const auto sig = crypto::ed25519_sign(*secret, *msg);
    if (sig == std::array<uint8_t, crypto::ed25519_signature_size>{}) {
        return std::unexpected(DomainError{fn, "signing failed"});
    }
    return crypto::to_hex(std::span<const uint8_t>(sig.data(), sig.size())) + "\n";
}

Result<std::string> eval_crypto_ed25519_verify(const std::string& pub_arg,
                                               const std::string& msg_arg,
                                               const std::string& sig_arg) {
    constexpr const char* fn = "crypto_ed25519_verify";
    auto pub = parse_hex_arg(pub_arg, fn, "public_key");
    if (!pub) {
        return std::unexpected(pub.error());
    }
    auto msg = parse_hex_arg(msg_arg, fn, "message");
    if (!msg) {
        return std::unexpected(msg.error());
    }
    auto sig = parse_hex_arg(sig_arg, fn, "signature");
    if (!sig) {
        return std::unexpected(sig.error());
    }
    if (pub->size() != crypto::ed25519_public_key_size) {
        return std::unexpected(DomainError{fn, "expected 32-byte (64 hex char) public key"});
    }
    if (sig->size() != crypto::ed25519_signature_size) {
        return std::unexpected(DomainError{fn, "expected 64-byte (128 hex char) signature"});
    }
    const bool ok = crypto::ed25519_verify(*pub, *msg, *sig);
    return std::string(ok ? "1" : "0") + "\n";
}

Result<std::string> eval_crypto_constant_time_eq(const std::string& hex_a,
                                                 const std::string& hex_b) {
    constexpr const char* fn = "crypto_constant_time_eq";
    auto a = parse_hex_arg(hex_a, fn, "hex_a");
    if (!a) {
        return std::unexpected(a.error());
    }
    auto b = parse_hex_arg(hex_b, fn, "hex_b");
    if (!b) {
        return std::unexpected(b.error());
    }
    const bool ok = crypto::constant_time_eq(*a, *b);
    return std::string(ok ? "1" : "0") + "\n";
}

Result<std::string> eval_crypto_random_bytes(const std::string& n_arg) {
    constexpr const char* fn = "crypto_random_bytes";
    double n_d = 0.0;
    if (!parse_number(trim_copy(n_arg), n_d)) {
        return std::unexpected(DomainError{fn, "expected numeric byte count"});
    }
    const auto n = static_cast<std::size_t>(n_d);
    if (n_d < 0.0 || n_d != static_cast<double>(n)) {
        return std::unexpected(DomainError{fn, "expected non-negative integer byte count"});
    }
    return crypto::to_hex(crypto::random_bytes(n)) + "\n";
}

std::vector<std::size_t> fem_rectangular_boundary_nodes(std::size_t nx, std::size_t ny) {
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

std::vector<std::size_t> fem_box_boundary_nodes(
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

Result<Matrix<double>> eval_fem_poisson1d(std::size_t n) {
    constexpr const char* fn = "fem_poisson1d";
    if (n == 0) {
        return std::unexpected(DomainError{fn, "expected positive n"});
    }
    auto mesh = fem::mesh1d(0.0, 1.0, n);
    if (!mesh) return std::unexpected(mesh.error());
    auto K = fem::assemble_stiffness_1d(*mesh);
    if (!K) return std::unexpected(K.error());
    auto f = fem::assemble_load_1d(*mesh, [](double) { return 1.0; });
    if (!f) return std::unexpected(f.error());
    if (auto r = fem::apply_dirichlet(*K, *f, {0, mesh->nodes.size() - 1}, {0.0, 0.0}); !r) {
        return std::unexpected(r.error());
    }
    const auto u = fem::solve_fem(*K, *f);
    if (!u) {
        return std::unexpected(DomainError{fn, "linear solve failed"});
    }
    Matrix<double> out(u->rows(), 1);
    for (std::size_t i = 0; i < u->rows(); ++i) {
        out(i, 0) = (*u)(i, 0);
    }
    return out;
}

Result<Matrix<double>> eval_fem_poisson2d(std::size_t nx, std::size_t ny) {
    constexpr const char* fn = "fem_poisson2d";
    if (nx == 0 || ny == 0) {
        return std::unexpected(DomainError{fn, "expected positive nx and ny"});
    }
    auto mesh = fem::mesh2d_rectangular(0.0, 0.0, 1.0, 1.0, nx, ny);
    if (!mesh) return std::unexpected(mesh.error());
    auto K = fem::assemble_stiffness_2d(*mesh);
    if (!K) return std::unexpected(K.error());
    auto f = fem::assemble_load_2d(*mesh, [](double, double) { return 1.0; });
    if (!f) return std::unexpected(f.error());
    const auto boundary = fem_rectangular_boundary_nodes(nx, ny);
    std::vector<double> boundary_values(boundary.size(), 0.0);
    if (auto r = fem::apply_dirichlet(*K, *f, boundary, boundary_values); !r) {
        return std::unexpected(r.error());
    }
    const auto u = fem::solve_fem(*K, *f);
    if (!u) {
        return std::unexpected(DomainError{fn, "linear solve failed"});
    }
    Matrix<double> out(u->rows(), 1);
    for (std::size_t i = 0; i < u->rows(); ++i) {
        out(i, 0) = (*u)(i, 0);
    }
    return out;
}

Result<Matrix<double>> eval_fem_poisson3d(
    std::size_t nx, std::size_t ny, std::size_t nz) {
    constexpr const char* fn = "fem_poisson3d";
    if (nx == 0 || ny == 0 || nz == 0) {
        return std::unexpected(DomainError{fn, "expected positive nx, ny, and nz"});
    }
    auto mesh = fem::mesh3d_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, nx, ny, nz);
    if (!mesh) return std::unexpected(mesh.error());
    auto K = fem::assemble_stiffness_3d(*mesh);
    if (!K) return std::unexpected(K.error());
    auto f = fem::assemble_load_3d(*mesh, [](double, double, double) { return 1.0; });
    if (!f) return std::unexpected(f.error());
    const auto boundary = fem_box_boundary_nodes(nx, ny, nz);
    std::vector<double> boundary_values(boundary.size(), 0.0);
    if (auto r = fem::apply_dirichlet(*K, *f, boundary, boundary_values); !r) {
        return std::unexpected(r.error());
    }
    const auto u = fem::solve_fem_3d(*K, *f);
    if (!u) {
        return std::unexpected(DomainError{fn, "linear solve failed"});
    }
    Matrix<double> out(u->rows(), 1);
    for (std::size_t i = 0; i < u->rows(); ++i) {
        out(i, 0) = (*u)(i, 0);
    }
    return out;
}

constexpr double kFemMesh1dTag = 269.0;

Matrix<double> pack_fem_mesh1d(const fem::Mesh1D& mesh) {
    const std::size_t n_nodes = mesh.nodes.size();
    const std::size_t n_elem = mesh.connectivity.size();
    Matrix<double> out(1 + n_nodes + n_elem, 3, 0.0);
    out(0, 0) = kFemMesh1dTag;
    out(0, 1) = static_cast<double>(n_nodes);
    out(0, 2) = static_cast<double>(n_elem);
    for (std::size_t i = 0; i < n_nodes; ++i) {
        out(1 + i, 0) = mesh.nodes[i];
    }
    for (std::size_t e = 0; e < n_elem; ++e) {
        const std::size_t row = 1 + n_nodes + e;
        out(row, 0) = static_cast<double>(mesh.connectivity[e][0]);
        out(row, 1) = static_cast<double>(mesh.connectivity[e][1]);
    }
    return out;
}

Result<fem::Mesh1D> matrix_to_fem_mesh1d(const Matrix<double>& m, const char* fn) {
    if (m.rows() < 1 || m.cols() < 3 || m(0, 0) != kFemMesh1dTag) {
        return std::unexpected(DomainError{fn, "expected packed fem_mesh1d matrix"});
    }
    const double n_nodes_d = m(0, 1);
    const double n_elem_d = m(0, 2);
    if (n_nodes_d < 2.0 || n_elem_d < 1.0 || std::floor(n_nodes_d) != n_nodes_d ||
        std::floor(n_elem_d) != n_elem_d) {
        return std::unexpected(DomainError{fn, "invalid fem_mesh1d header"});
    }
    const std::size_t n_nodes = static_cast<std::size_t>(n_nodes_d);
    const std::size_t n_elem = static_cast<std::size_t>(n_elem_d);
    if (m.rows() != 1 + n_nodes + n_elem) {
        return std::unexpected(DomainError{fn, "fem_mesh1d row count mismatch"});
    }
    fem::Mesh1D mesh;
    mesh.nodes.resize(n_nodes);
    mesh.connectivity.resize(n_elem);
    for (std::size_t i = 0; i < n_nodes; ++i) {
        mesh.nodes[i] = m(1 + i, 0);
    }
    for (std::size_t e = 0; e < n_elem; ++e) {
        const std::size_t row = 1 + n_nodes + e;
        for (int k = 0; k < 2; ++k) {
            const double idx_d = m(row, static_cast<std::size_t>(k));
            if (idx_d < 0.0 || std::floor(idx_d) != idx_d) {
                return std::unexpected(DomainError{fn, "expected non-negative integer element index"});
            }
            mesh.connectivity[e][static_cast<std::size_t>(k)] = static_cast<std::size_t>(idx_d);
        }
    }
    return mesh;
}

Result<Matrix<double>> eval_fem_mesh1d(double a, double b, std::size_t n_elements) {
    constexpr const char* fn = "fem_mesh1d";
    if (n_elements == 0) {
        return std::unexpected(DomainError{fn, "expected positive n_elements"});
    }
    if (b <= a) {
        return std::unexpected(DomainError{fn, "expected b > a"});
    }
    auto mesh = fem::mesh1d(a, b, n_elements);
    if (!mesh) return std::unexpected(mesh.error());
    return pack_fem_mesh1d(*mesh);
}

Result<Matrix<double>> eval_fem_stiffness_1d(const Matrix<double>& mesh_m) {
    constexpr const char* fn = "fem_stiffness_1d";
    auto mesh = matrix_to_fem_mesh1d(mesh_m, fn);
    if (!mesh) {
        return std::unexpected(mesh.error());
    }
    auto K = fem::assemble_stiffness_1d(*mesh);
    if (!K) return std::unexpected(K.error());
    return col_matrix_to_matrix(*K);
}

Result<Matrix<double>> eval_fem_load_1d(const Matrix<double>& mesh_m, double f_const) {
    constexpr const char* fn = "fem_load_1d";
    auto mesh = matrix_to_fem_mesh1d(mesh_m, fn);
    if (!mesh) {
        return std::unexpected(mesh.error());
    }
    auto load = fem::assemble_load_1d(*mesh, [f_const](double) { return f_const; });
    if (!load) return std::unexpected(load.error());
    return col_matrix_to_matrix(*load);
}

Result<Matrix<double>> eval_fem_lagrange_eval(double xi) {
    constexpr const char* fn = "fem_lagrange_eval";
    if (xi < 0.0 || xi > 1.0) {
        return std::unexpected(DomainError{fn, "expected xi in [0, 1]"});
    }
    const auto basis = fem::lagrange_basis();
    const auto values = basis.evaluate(xi);
    if (!values) return std::unexpected(values.error());
    const auto derivs = basis.derivative(xi);
    if (!derivs) return std::unexpected(derivs.error());
    Matrix<double> out(2, 2, 0.0);
    out(0, 0) = (*values)[0];
    out(0, 1) = (*values)[1];
    out(1, 0) = (*derivs)[0];
    out(1, 1) = (*derivs)[1];
    return out;
}

constexpr double kFemMesh2dTag = 270.0;

Matrix<double> pack_fem_mesh2d(const fem::Mesh2D& mesh) {
    const std::size_t n_nodes = mesh.nodes.size();
    const std::size_t n_tri = mesh.triangles.size();
    Matrix<double> out(1 + n_nodes + n_tri, 3, 0.0);
    out(0, 0) = kFemMesh2dTag;
    out(0, 1) = static_cast<double>(n_nodes);
    out(0, 2) = static_cast<double>(n_tri);
    for (std::size_t i = 0; i < n_nodes; ++i) {
        out(1 + i, 0) = mesh.nodes[i][0];
        out(1 + i, 1) = mesh.nodes[i][1];
    }
    for (std::size_t t = 0; t < n_tri; ++t) {
        const std::size_t row = 1 + n_nodes + t;
        out(row, 0) = static_cast<double>(mesh.triangles[t][0]);
        out(row, 1) = static_cast<double>(mesh.triangles[t][1]);
        out(row, 2) = static_cast<double>(mesh.triangles[t][2]);
    }
    return out;
}

Result<fem::Mesh2D> matrix_to_fem_mesh2d(const Matrix<double>& m, const char* fn) {
    if (m.rows() < 1 || m.cols() < 3 || m(0, 0) != kFemMesh2dTag) {
        return std::unexpected(DomainError{fn, "expected packed fem_mesh2d matrix"});
    }
    const double n_nodes_d = m(0, 1);
    const double n_tri_d = m(0, 2);
    if (n_nodes_d < 3.0 || n_tri_d < 1.0 || std::floor(n_nodes_d) != n_nodes_d ||
        std::floor(n_tri_d) != n_tri_d) {
        return std::unexpected(DomainError{fn, "invalid fem_mesh2d header"});
    }
    const std::size_t n_nodes = static_cast<std::size_t>(n_nodes_d);
    const std::size_t n_tri = static_cast<std::size_t>(n_tri_d);
    if (m.rows() != 1 + n_nodes + n_tri) {
        return std::unexpected(DomainError{fn, "fem_mesh2d row count mismatch"});
    }
    fem::Mesh2D mesh;
    mesh.nodes.resize(n_nodes);
    mesh.triangles.resize(n_tri);
    for (std::size_t i = 0; i < n_nodes; ++i) {
        mesh.nodes[i] = {m(1 + i, 0), m(1 + i, 1)};
    }
    for (std::size_t t = 0; t < n_tri; ++t) {
        const std::size_t row = 1 + n_nodes + t;
        for (int k = 0; k < 3; ++k) {
            const double idx_d = m(row, static_cast<std::size_t>(k));
            if (idx_d < 0.0 || std::floor(idx_d) != idx_d) {
                return std::unexpected(DomainError{fn, "expected non-negative integer triangle index"});
            }
            mesh.triangles[t][static_cast<std::size_t>(k)] = static_cast<std::size_t>(idx_d);
        }
    }
    return mesh;
}

Result<std::vector<std::size_t>> matrix_to_fem_node_indices(const Matrix<double>& m,
                                                            const char* fn) {
    auto vec = matrix_to_ml_vec(m, fn);
    if (!vec) {
        return std::unexpected(vec.error());
    }
    std::vector<std::size_t> out;
    out.reserve(vec->size());
    for (const double entry : *vec) {
        if (entry < 0.0 || std::floor(entry) != entry) {
            return std::unexpected(
                DomainError{fn, "expected non-negative integer node indices"});
        }
        out.push_back(static_cast<std::size_t>(entry));
    }
    return out;
}

Result<Matrix<double>> eval_fem_mesh2d_rectangular(
    double x0, double y0, double x1, double y1, std::size_t nx, std::size_t ny) {
    constexpr const char* fn = "fem_mesh2d_rectangular";
    if (nx == 0 || ny == 0) {
        return std::unexpected(DomainError{fn, "expected positive nx and ny"});
    }
    auto mesh = fem::mesh2d_rectangular(x0, y0, x1, y1, nx, ny);
    if (!mesh) return std::unexpected(mesh.error());
    return pack_fem_mesh2d(*mesh);
}

Result<Matrix<double>> eval_fem_stiffness_2d(const Matrix<double>& mesh_m) {
    constexpr const char* fn = "fem_stiffness_2d";
    auto mesh = matrix_to_fem_mesh2d(mesh_m, fn);
    if (!mesh) {
        return std::unexpected(mesh.error());
    }
    auto K = fem::assemble_stiffness_2d(*mesh);
    if (!K) return std::unexpected(K.error());
    return col_matrix_to_matrix(*K);
}

Result<Matrix<double>> eval_fem_load_2d(const Matrix<double>& mesh_m, double f_const) {
    constexpr const char* fn = "fem_load_2d";
    auto mesh = matrix_to_fem_mesh2d(mesh_m, fn);
    if (!mesh) {
        return std::unexpected(mesh.error());
    }
    auto load = fem::assemble_load_2d(*mesh, [f_const](double, double) { return f_const; });
    if (!load) return std::unexpected(load.error());
    return col_matrix_to_matrix(*load);
}

Result<Matrix<double>> eval_fem_apply_dirichlet(const Matrix<double>& K_m,
                                               const Matrix<double>& f_m,
                                               const Matrix<double>& nodes_m,
                                               const Matrix<double>& values_m) {
    constexpr const char* fn = "fem_apply_dirichlet";
    if (K_m.rows() != K_m.cols()) {
        return std::unexpected(DomainError{fn, "expected square stiffness matrix K"});
    }
    auto f_col = sparse_vector_to_col_column(f_m, K_m.rows(), fn);
    if (!f_col) {
        return std::unexpected(f_col.error());
    }
    auto nodes = matrix_to_fem_node_indices(nodes_m, fn);
    if (!nodes) {
        return std::unexpected(nodes.error());
    }
    auto values = matrix_to_ml_vec(values_m, fn);
    if (!values) {
        return std::unexpected(values.error());
    }
    if (nodes->size() != values->size()) {
        return std::unexpected(
            DomainError{fn, "expected node_indices and values with the same length"});
    }
    ColMatrix<double> K = matrix_to_col_matrix(K_m);
    ColMatrix<double> f = *f_col;
    std::vector<double> bc_values(values->begin(), values->end());
    if (auto r = fem::apply_dirichlet(K, f, *nodes, bc_values); !r) {
        return std::unexpected(r.error());
    }
    Matrix<double> out(K.rows(), K.cols() + 1);
    for (std::size_t i = 0; i < K.rows(); ++i) {
        for (std::size_t j = 0; j < K.cols(); ++j) {
            out(i, j) = K(i, j);
        }
        out(i, K.cols()) = f(i, 0);
    }
    return out;
}

Result<Matrix<double>> eval_fem_solve(const Matrix<double>& K_m, const Matrix<double>& f_m) {
    constexpr const char* fn = "fem_solve";
    ColMatrix<double> K = matrix_to_col_matrix(K_m);
    auto f_col = sparse_vector_to_col_column(f_m, K.rows(), fn);
    if (!f_col) {
        return std::unexpected(f_col.error());
    }
    const auto u = fem::solve_fem(K, *f_col);
    if (!u) {
        return std::unexpected(DomainError{fn, "linear solve failed"});
    }
    return col_matrix_to_matrix(*u);
}

Result<Matrix<double>> eval_fem_solve_packed(const Matrix<double>& sys_m) {
    constexpr const char* fn = "fem_solve";
    if (sys_m.rows() < 2 || sys_m.cols() != sys_m.rows() + 1) {
        return std::unexpected(
            DomainError{fn, "expected fem_solve(K,f) or n×(n+1) packed system from fem_apply_dirichlet"});
    }
    Matrix<double> K_m(sys_m.rows(), sys_m.rows());
    Matrix<double> f_m(sys_m.rows(), 1);
    for (std::size_t i = 0; i < sys_m.rows(); ++i) {
        for (std::size_t j = 0; j < sys_m.rows(); ++j) {
            K_m(i, j) = sys_m(i, j);
        }
        f_m(i, 0) = sys_m(i, sys_m.rows());
    }
    return eval_fem_solve(K_m, f_m);
}

Result<Matrix<double>> eval_fem_solve_3d(const Matrix<double>& K_m, const Matrix<double>& f_m) {
    constexpr const char* fn = "fem_solve_3d";
    ColMatrix<double> K = matrix_to_col_matrix(K_m);
    auto f_col = sparse_vector_to_col_column(f_m, K.rows(), fn);
    if (!f_col) {
        return std::unexpected(f_col.error());
    }
    const auto u = fem::solve_fem_3d(K, *f_col);
    if (!u) {
        return std::unexpected(DomainError{fn, "linear solve failed"});
    }
    return col_matrix_to_matrix(*u);
}

Result<Matrix<double>> eval_fem_lagrange_deriv(double xi) {
    constexpr const char* fn = "fem_lagrange_deriv";
    if (xi < 0.0 || xi > 1.0) {
        return std::unexpected(DomainError{fn, "expected xi in [0, 1]"});
    }
    const auto basis = fem::lagrange_basis();
    const auto derivs = basis.derivative(xi);
    if (!derivs) return std::unexpected(derivs.error());
    Matrix<double> out(1, 2, 0.0);
    out(0, 0) = (*derivs)[0];
    out(0, 1) = (*derivs)[1];
    return out;
}

Result<double> eval_cplx_green_function_disk(double zre, double zim, double z0re, double z0im,
                                              double radius) {
    return cplx::green_function_disk(cplx::C(zre, zim), cplx::C(z0re, z0im), radius);
}

Result<std::string> eval_cplx_cauchy_principal_value_call(const std::string& formula_arg,
                                                          const std::string& a_arg,
                                                          const std::string& c_arg,
                                                          const std::string& b_arg,
                                                          const std::string& n_pts_arg) {
    constexpr const char* fn = "cplx_cauchy_principal_value";
    auto expr = parse_sym_quoted_expr(formula_arg, fn);
    if (!expr) {
        return std::unexpected(expr.error());
    }
    double a = 0.0;
    double c = 0.0;
    double b = 0.0;
    double n_pts_d = 200.0;
    if (!parse_number(trim_copy(a_arg), a) || !parse_number(trim_copy(c_arg), c) ||
        !parse_number(trim_copy(b_arg), b)) {
        return std::unexpected(DomainError{
            fn, "expected cplx_cauchy_principal_value(\"formula\", a, c, b[, n_pts])"});
    }
    if (!n_pts_arg.empty() &&
        !parse_number(trim_copy(n_pts_arg), n_pts_d)) {
        return std::unexpected(DomainError{
            fn, "expected cplx_cauchy_principal_value(\"formula\", a, c, b[, n_pts])"});
    }
    const int n_pts_i = static_cast<int>(n_pts_d);
    if (n_pts_i < 0 || n_pts_d != n_pts_i) {
        return std::unexpected(DomainError{fn, "expected non-negative integer n_pts"});
    }
    SymExpr parsed = std::move(*expr);
    auto expr_ptr = std::make_shared<SymExpr>(std::move(parsed));
    cplx::RealFunc f = [expr_ptr](double x) {
        return sym_eval(*expr_ptr, {{"x", x}});
    };
    return std::to_string(
               cplx::cauchy_principal_value(f, a, c, b, n_pts_i)) +
           "\n";
}

Matrix<double> sparse_to_packed_matrix(const Sparse<double>& S) {
    const auto dense = S.to_dense();
    size_t nnz = 0;
    for (size_t i = 0; i < S.rows(); ++i) {
        for (size_t j = 0; j < S.cols(); ++j) {
            if (dense(i, j) != 0.0) {
                ++nnz;
            }
        }
    }
    Matrix<double> out(nnz + 1, 3);
    out(0, 0) = static_cast<double>(S.rows());
    out(0, 1) = static_cast<double>(S.cols());
    out(0, 2) = static_cast<double>(nnz);
    size_t k = 0;
    for (size_t i = 0; i < S.rows(); ++i) {
        for (size_t j = 0; j < S.cols(); ++j) {
            const double value = dense(i, j);
            if (value != 0.0) {
                out(k + 1, 0) = static_cast<double>(i);
                out(k + 1, 1) = static_cast<double>(j);
                out(k + 1, 2) = value;
                ++k;
            }
        }
    }
    return out;
}

Result<Sparse<double>> sparse_from_packed_matrix(const Matrix<double>& m, const char* fn) {
    if (m.rows() < 1 || m.cols() < 3) {
        return std::unexpected(DomainError{
            fn, "expected packed sparse matrix (nnz+1)x3 with header row [rows,cols,nnz]"});
    }
    const double rows_d = m(0, 0);
    const double cols_d = m(0, 1);
    const double nnz_d = m(0, 2);
    const int rows_i = static_cast<int>(rows_d);
    const int cols_i = static_cast<int>(cols_d);
    const int nnz_i = static_cast<int>(nnz_d);
    if (rows_i < 0 || cols_i < 0 || nnz_i < 0 || rows_d != rows_i || cols_d != cols_i ||
        nnz_d != nnz_i) {
        return std::unexpected(DomainError{fn, "invalid sparse header row [rows,cols,nnz]"});
    }
    if (m.rows() != static_cast<size_t>(nnz_i + 1)) {
        return std::unexpected(DomainError{fn, "packed sparse row count must be nnz+1"});
    }
    std::vector<size_t> rowidx(static_cast<size_t>(nnz_i));
    std::vector<size_t> colidx(static_cast<size_t>(nnz_i));
    std::vector<double> values(static_cast<size_t>(nnz_i));
    for (int k = 0; k < nnz_i; ++k) {
        const double row = m(static_cast<size_t>(k + 1), 0);
        const double col = m(static_cast<size_t>(k + 1), 1);
        const int row_i = static_cast<int>(row);
        const int col_i = static_cast<int>(col);
        if (row_i < 0 || col_i < 0 || row != row_i || col != col_i ||
            static_cast<size_t>(row_i) >= static_cast<size_t>(rows_i) ||
            static_cast<size_t>(col_i) >= static_cast<size_t>(cols_i)) {
            return std::unexpected(DomainError{fn, "invalid COO index in packed sparse matrix"});
        }
        rowidx[static_cast<size_t>(k)] = static_cast<size_t>(row_i);
        colidx[static_cast<size_t>(k)] = static_cast<size_t>(col_i);
        values[static_cast<size_t>(k)] = m(static_cast<size_t>(k + 1), 2);
    }
    return Sparse<double>(static_cast<size_t>(rows_i), static_cast<size_t>(cols_i),
                          std::move(rowidx), std::move(colidx), std::move(values));
}

Result<ColMatrix<double>> sparse_vector_to_col_column(const Matrix<double>& m, size_t expected_len,
                                                      const char* fn) {
    if (m.rows() == expected_len && m.cols() == 1) {
        ColMatrix<double> column(expected_len, 1);
        for (size_t i = 0; i < expected_len; ++i) {
            column(i, 0) = m(i, 0);
        }
        return column;
    }
    if (m.rows() == 1 && m.cols() == expected_len) {
        ColMatrix<double> column(expected_len, 1);
        for (size_t j = 0; j < expected_len; ++j) {
            column(j, 0) = m(0, j);
        }
        return column;
    }
    return std::unexpected(
        DomainError{fn, "expected vector length matching sparse column count"});
}

Matrix<double> col_matrix_to_matrix(const ColMatrix<double>& m) {
    Matrix<double> out(m.rows(), m.cols());
    for (size_t i = 0; i < m.rows(); ++i) {
        for (size_t j = 0; j < m.cols(); ++j) {
            out(i, j) = m(i, j);
        }
    }
    return out;
}

Result<Matrix<double>> eval_sparse_from_coo(std::size_t rows, std::size_t cols,
                                            const Matrix<double>& row_idx_m,
                                            const Matrix<double>& col_idx_m,
                                            const Matrix<double>& values_m) {
    constexpr const char* fn = "sparse_from_coo";
    auto row_idx = matrix_to_ml_vec(row_idx_m, fn);
    if (!row_idx) {
        return std::unexpected(row_idx.error());
    }
    auto col_idx = matrix_to_ml_vec(col_idx_m, fn);
    if (!col_idx) {
        return std::unexpected(col_idx.error());
    }
    auto values = matrix_to_ml_vec(values_m, fn);
    if (!values) {
        return std::unexpected(values.error());
    }
    if (row_idx->size() != col_idx->size() || row_idx->size() != values->size()) {
        return std::unexpected(
            DomainError{fn, "row_idx, col_idx, and values must have equal length"});
    }
    std::vector<size_t> rows_vec;
    std::vector<size_t> cols_vec;
    std::vector<double> vals;
    rows_vec.reserve(row_idx->size());
    cols_vec.reserve(col_idx->size());
    vals.reserve(values->size());
    for (size_t k = 0; k < row_idx->size(); ++k) {
        const double row = (*row_idx)[k];
        const double col = (*col_idx)[k];
        const int row_i = static_cast<int>(row);
        const int col_i = static_cast<int>(col);
        if (row_i < 0 || col_i < 0 || row != row_i || col != col_i) {
            return std::unexpected(
                DomainError{fn, "expected non-negative integer COO indices"});
        }
        if (static_cast<size_t>(row_i) >= rows || static_cast<size_t>(col_i) >= cols) {
            return std::unexpected(
                DomainError{fn, "COO index out of bounds for given rows/cols"});
        }
        rows_vec.push_back(static_cast<size_t>(row_i));
        cols_vec.push_back(static_cast<size_t>(col_i));
        vals.push_back((*values)[k]);
    }
    const Sparse<double> sparse(rows, cols, std::move(rows_vec), std::move(cols_vec),
                                std::move(vals));
    return sparse_to_packed_matrix(sparse);
}

Result<Matrix<double>> eval_sparse_spmv(const Matrix<double>& packed_m,
                                       const Matrix<double>& x_m) {
    constexpr const char* fn = "sparse_spmv";
    auto sparse = sparse_from_packed_matrix(packed_m, fn);
    if (!sparse) {
        return std::unexpected(sparse.error());
    }
    auto x = sparse_vector_to_col_column(x_m, sparse->cols(), fn);
    if (!x) {
        return std::unexpected(x.error());
    }
    return col_matrix_to_matrix(sparse->spmv(*x));
}

Result<Matrix<double>> eval_sparse_to_dense(const Matrix<double>& packed_m) {
    constexpr const char* fn = "sparse_to_dense";
    auto sparse = sparse_from_packed_matrix(packed_m, fn);
    if (!sparse) {
        return std::unexpected(sparse.error());
    }
    return col_matrix_to_matrix(sparse->to_dense());
}

Result<Matrix<double>> eval_sparse_add(const Matrix<double>& a_m, const Matrix<double>& b_m) {
    constexpr const char* fn = "sparse_add";
    auto a = sparse_from_packed_matrix(a_m, fn);
    if (!a) {
        return std::unexpected(a.error());
    }
    auto b = sparse_from_packed_matrix(b_m, fn);
    if (!b) {
        return std::unexpected(b.error());
    }
    return sparse_to_packed_matrix(sparse_add(*a, *b));
}

Result<Matrix<double>> eval_cfd_advection1d(std::size_t nx, double vx, double t_end, double dt) {
    constexpr const char* fn = "cfd_advection1d";
    if (nx < 2) {
        return std::unexpected(DomainError{fn, "expected nx >= 2"});
    }
    if (t_end <= 0.0 || dt <= 0.0) {
        return std::unexpected(DomainError{fn, "expected positive t_end and dt"});
    }
    const cfd::Grid1D grid = cfd::grid1d(0.0, 1.0, nx);
    if (grid.n == 0) {
        return std::unexpected(DomainError{fn, "invalid grid dimensions"});
    }
    const auto u0 = cfd::square_pulse(grid, 0.35, 0.1, 1.0);
    const auto v_field = cfd::constant_velocity(grid.n, vx);
    const auto result = cfd::run_advection(
        u0, v_field, t_end, dt, grid.dx, cfd::BoundaryCondition::Periodic);
    if (result.u.empty()) {
        return std::unexpected(
            DomainError{fn, "CFL stability condition violated or invalid input"});
    }
    return vector_to_column(result.u.back());
}

Result<Matrix<double>> eval_cfd_advection2d(std::size_t nx, std::size_t ny, double vx, double vy,
                                            double t_end, double dt) {
    constexpr const char* fn = "cfd_advection2d";
    if (nx < 2 || ny < 2) {
        return std::unexpected(DomainError{fn, "expected nx, ny >= 2"});
    }
    if (t_end <= 0.0 || dt <= 0.0) {
        return std::unexpected(DomainError{fn, "expected positive t_end and dt"});
    }
    const cfd::Grid2D grid = cfd::grid2d(0.0, 1.0, 0.0, 1.0, nx, ny);
    if (grid.nx == 0 || grid.ny == 0) {
        return std::unexpected(DomainError{fn, "invalid grid dimensions"});
    }
    const auto u0 = cfd::square_pulse_2d(grid, 0.35, 0.35, 0.1, 0.1, 1.0);
    const auto vx_field = cfd::constant_velocity(grid.nx * grid.ny, vx);
    const auto vy_field = cfd::constant_velocity(grid.nx * grid.ny, vy);
    const auto result = cfd::run_advection_2d(
        u0, vx_field, vy_field, t_end, dt, grid.dx, grid.dy, cfd::BoundaryCondition::Periodic,
        cfd::BoundaryCondition::Periodic);
    if (result.u.empty()) {
        return std::unexpected(
            DomainError{fn, "CFL stability condition violated or invalid input"});
    }
    return grid_to_matrix(result.u.back());
}

Result<Matrix<double>> eval_cfd_advection3d(std::size_t nx, std::size_t ny, std::size_t nz,
                                            double vx, double vy, double vz, double t_end,
                                            double dt) {
    constexpr const char* fn = "cfd_advection3d";
    if (nx < 2 || ny < 2 || nz < 2) {
        return std::unexpected(DomainError{fn, "expected nx, ny, nz >= 2"});
    }
    if (t_end <= 0.0 || dt <= 0.0) {
        return std::unexpected(DomainError{fn, "expected positive t_end and dt"});
    }
    const cfd::Grid3D grid = cfd::grid3d(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, nx, ny, nz);
    if (grid.nx == 0 || grid.ny == 0 || grid.nz == 0) {
        return std::unexpected(DomainError{fn, "invalid grid dimensions"});
    }
    const auto u0 = cfd::square_pulse_3d(grid, 0.35, 0.35, 0.35, 0.1, 0.1, 0.1, 1.0);
    const std::size_t n_cells = grid.nx * grid.ny * grid.nz;
    const auto vx_field = cfd::constant_velocity(n_cells, vx);
    const auto vy_field = cfd::constant_velocity(n_cells, vy);
    const auto vz_field = cfd::constant_velocity(n_cells, vz);
    const auto result = cfd::run_advection_3d(
        u0, vx_field, vy_field, vz_field, t_end, dt, grid.dx, grid.dy, grid.dz,
        cfd::BoundaryCondition::Periodic, cfd::BoundaryCondition::Periodic,
        cfd::BoundaryCondition::Periodic);
    if (result.u.empty()) {
        return std::unexpected(
            DomainError{fn, "CFL stability condition violated or invalid input"});
    }
    return grid3d_to_matrix(result.u.back());
}

Matrix<double> pack_cfd_grid1d(const cfd::Grid1D& grid) {
    const size_t ncols = std::max(grid.n, size_t{4});
    Matrix<double> out(2, ncols, 0.0);
    out(0, 0) = grid.x0;
    out(0, 1) = grid.x1;
    out(0, 2) = grid.dx;
    out(0, 3) = static_cast<double>(grid.n);
    for (size_t j = 0; j < grid.n; ++j) {
        out(1, j) = grid.x[j];
    }
    return out;
}

Result<cfd::Grid1D> cfd_grid1d_from_packed_matrix(const Matrix<double>& m, const char* fn) {
    if (m.rows() != 2 || m.cols() < 4) {
        return std::unexpected(DomainError{
            fn, "expected packed CFD grid1d matrix (2 rows, header [x0,x1,dx,n])"});
    }
    const double n_d = m(0, 3);
    const int n_i = static_cast<int>(n_d);
    if (n_i < 2 || n_d != n_i) {
        return std::unexpected(DomainError{fn, "invalid packed grid n (expected integer >= 2)"});
    }
    const size_t n = static_cast<size_t>(n_i);
    if (m.cols() < n) {
        return std::unexpected(DomainError{fn, "packed grid coordinate row shorter than n"});
    }
    cfd::Grid1D grid;
    grid.x0 = m(0, 0);
    grid.x1 = m(0, 1);
    grid.dx = m(0, 2);
    grid.n = n;
    grid.x.assign(n, 0.0);
    for (size_t j = 0; j < n; ++j) {
        grid.x[j] = m(1, j);
    }
    return grid;
}

Result<Matrix<double>> eval_cfd_grid1d(double x0, double x1, std::size_t n) {
    constexpr const char* fn = "cfd_grid1d";
    if (x1 <= x0) {
        return std::unexpected(DomainError{fn, "expected x1 > x0"});
    }
    if (n < 2) {
        return std::unexpected(DomainError{fn, "expected n >= 2"});
    }
    const cfd::Grid1D grid = cfd::grid1d(x0, x1, n);
    if (grid.n == 0) {
        return std::unexpected(DomainError{fn, "invalid grid dimensions"});
    }
    return pack_cfd_grid1d(grid);
}

Result<Matrix<double>> eval_cfd_square_pulse(
    const Matrix<double>& grid_m, double xc, double width, double amplitude) {
    constexpr const char* fn = "cfd_square_pulse";
    auto grid = cfd_grid1d_from_packed_matrix(grid_m, fn);
    if (!grid) {
        return std::unexpected(grid.error());
    }
    const auto pulse = cfd::square_pulse(*grid, xc, width, amplitude);
    if (pulse.empty()) {
        return std::unexpected(DomainError{fn, "failed to build square pulse"});
    }
    return vector_to_column(pulse);
}

Result<Matrix<double>> eval_cfd_run_advection(const Matrix<double>& grid_m,
                                              const Matrix<double>& u_m, double v,
                                              double t_end, double dt) {
    constexpr const char* fn = "cfd_run_advection";
    auto grid = cfd_grid1d_from_packed_matrix(grid_m, fn);
    if (!grid) {
        return std::unexpected(grid.error());
    }
    auto u0 = matrix_to_coeff_vector(u_m, fn);
    if (!u0) {
        return std::unexpected(u0.error());
    }
    if (u0->size() != grid->n) {
        return std::unexpected(DomainError{fn, "field length must match grid n"});
    }
    if (t_end <= 0.0 || dt <= 0.0) {
        return std::unexpected(DomainError{fn, "expected positive t_end and dt"});
    }
    const auto vx = cfd::constant_velocity(u0->size(), v);
    const auto result = cfd::run_advection(*u0, vx, t_end, dt, grid->dx);
    if (result.u.empty()) {
        return std::unexpected(
            DomainError{fn, "CFL stability condition violated or invalid input"});
    }
    return vector_to_column(result.u.back());
}

Result<Matrix<double>> eval_cfd_upwind_step_1d(const Matrix<double>& grid_m,
                                               const Matrix<double>& u_m, double v, double dt,
                                               cfd::BoundaryCondition bc) {
    constexpr const char* fn = "cfd_upwind_step_1d";
    auto grid = cfd_grid1d_from_packed_matrix(grid_m, fn);
    if (!grid) {
        return std::unexpected(grid.error());
    }
    auto u0 = matrix_to_coeff_vector(u_m, fn);
    if (!u0) {
        return std::unexpected(u0.error());
    }
    if (u0->size() != grid->n) {
        return std::unexpected(DomainError{fn, "field length must match grid n"});
    }
    if (dt <= 0.0 || grid->dx <= 0.0) {
        return std::unexpected(DomainError{fn, "expected positive dt and dx"});
    }
    const std::vector<double> vx = {v};
    const auto u1 = cfd::upwind_fvm_advection(*u0, vx, dt, grid->dx, bc);
    if (u1.empty()) {
        return std::unexpected(
            DomainError{fn, "CFL stability condition violated or invalid input"});
    }
    return vector_to_column(u1);
}

Matrix<double> pack_cfd_grid2d(const cfd::Grid2D& grid) {
    const size_t ncols = std::max({grid.nx, grid.ny, size_t{8}});
    Matrix<double> out(3, ncols, 0.0);
    out(0, 0) = grid.x0;
    out(0, 1) = grid.x1;
    out(0, 2) = grid.y0;
    out(0, 3) = grid.y1;
    out(0, 4) = grid.dx;
    out(0, 5) = grid.dy;
    out(0, 6) = static_cast<double>(grid.nx);
    out(0, 7) = static_cast<double>(grid.ny);
    for (size_t j = 0; j < grid.nx; ++j) {
        out(1, j) = grid.x[j];
    }
    for (size_t j = 0; j < grid.ny; ++j) {
        out(2, j) = grid.y[j];
    }
    return out;
}

Result<cfd::Grid2D> cfd_grid2d_from_packed_matrix(const Matrix<double>& m, const char* fn) {
    if (m.rows() != 3 || m.cols() < 8) {
        return std::unexpected(DomainError{
            fn, "expected packed CFD grid2d matrix (3 rows, header [x0,x1,y0,y1,dx,dy,nx,ny])"});
    }
    const double nx_d = m(0, 6);
    const double ny_d = m(0, 7);
    const int nx_i = static_cast<int>(nx_d);
    const int ny_i = static_cast<int>(ny_d);
    if (nx_i < 2 || ny_i < 2 || nx_d != nx_i || ny_d != ny_i) {
        return std::unexpected(DomainError{fn, "invalid packed grid nx/ny (expected integers >= 2)"});
    }
    const size_t nx = static_cast<size_t>(nx_i);
    const size_t ny = static_cast<size_t>(ny_i);
    if (m.cols() < std::max(nx, ny)) {
        return std::unexpected(DomainError{fn, "packed grid coordinate rows shorter than nx/ny"});
    }
    cfd::Grid2D grid;
    grid.x0 = m(0, 0);
    grid.x1 = m(0, 1);
    grid.y0 = m(0, 2);
    grid.y1 = m(0, 3);
    grid.dx = m(0, 4);
    grid.dy = m(0, 5);
    grid.nx = nx;
    grid.ny = ny;
    grid.x.assign(nx, 0.0);
    grid.y.assign(ny, 0.0);
    for (size_t j = 0; j < nx; ++j) {
        grid.x[j] = m(1, j);
    }
    for (size_t j = 0; j < ny; ++j) {
        grid.y[j] = m(2, j);
    }
    return grid;
}

Result<cfd::BoundaryCondition> parse_cfd_bc(double value, const char* fn) {
    if (value == 0.0) {
        return cfd::BoundaryCondition::Periodic;
    }
    if (value == 1.0) {
        return cfd::BoundaryCondition::ZeroFlux;
    }
    return std::unexpected(DomainError{fn, "expected bc 0 (periodic) or 1 (zero_flux)"});
}

Result<Matrix<double>> eval_cfd_grid2d(
    double x0, double x1, double y0, double y1, std::size_t nx, std::size_t ny) {
    constexpr const char* fn = "cfd_grid2d";
    if (x1 <= x0 || y1 <= y0) {
        return std::unexpected(DomainError{fn, "expected x1 > x0 and y1 > y0"});
    }
    if (nx < 2 || ny < 2) {
        return std::unexpected(DomainError{fn, "expected nx, ny >= 2"});
    }
    const cfd::Grid2D grid = cfd::grid2d(x0, x1, y0, y1, nx, ny);
    if (grid.nx == 0 || grid.ny == 0) {
        return std::unexpected(DomainError{fn, "invalid grid dimensions"});
    }
    return pack_cfd_grid2d(grid);
}

Result<Matrix<double>> eval_cfd_square_pulse_2d(
    const Matrix<double>& grid_m,
    double xc,
    double yc,
    double width_x,
    double width_y,
    double amplitude) {
    constexpr const char* fn = "cfd_square_pulse_2d";
    auto grid = cfd_grid2d_from_packed_matrix(grid_m, fn);
    if (!grid) {
        return std::unexpected(grid.error());
    }
    const auto pulse = cfd::square_pulse_2d(*grid, xc, yc, width_x, width_y, amplitude);
    if (pulse.empty() || pulse.front().empty()) {
        return std::unexpected(DomainError{fn, "failed to build square pulse"});
    }
    return grid_to_matrix(pulse);
}

Result<Matrix<double>> eval_cfd_upwind_step_2d(
    const Matrix<double>& u_m,
    double vx,
    double vy,
    double dt,
    double dx,
    double dy,
    cfd::BoundaryCondition bc_x,
    cfd::BoundaryCondition bc_y) {
    constexpr const char* fn = "cfd_upwind_step_2d";
    auto u = matrix_to_grid(u_m, fn);
    if (!u) {
        return std::unexpected(u.error());
    }
    if (u->empty() || u->front().empty()) {
        return std::unexpected(DomainError{fn, "expected non-empty 2D field matrix"});
    }
    if (dt <= 0.0 || dx <= 0.0 || dy <= 0.0) {
        return std::unexpected(DomainError{fn, "expected positive dt, dx, and dy"});
    }
    const std::vector<double> vx_field = {vx};
    const std::vector<double> vy_field = {vy};
    const auto u1 = cfd::upwind_fvm_advection_2d(
        *u, vx_field, vy_field, dt, dx, dy, bc_x, bc_y);
    if (u1.empty()) {
        return std::unexpected(
            DomainError{fn, "CFL stability condition violated or invalid input"});
    }
    return grid_to_matrix(u1);
}

Result<double> eval_cfd_integrated_mass_2d(
    const Matrix<double>& u_m, double dx, double dy) {
    constexpr const char* fn = "cfd_integrated_mass_2d";
    auto u = matrix_to_grid(u_m, fn);
    if (!u) {
        return std::unexpected(u.error());
    }
    if (dx <= 0.0 || dy <= 0.0) {
        return std::unexpected(DomainError{fn, "expected positive dx and dy"});
    }
    return cfd::integrated_mass_2d(*u, dx, dy);
}

Result<double> eval_cfd_integrated_mass_1d(const Matrix<double>& grid_m,
                                           const Matrix<double>& u_m) {
    constexpr const char* fn = "cfd_integrated_mass_1d";
    auto grid = cfd_grid1d_from_packed_matrix(grid_m, fn);
    if (!grid) {
        return std::unexpected(grid.error());
    }
    auto u = matrix_to_coeff_vector(u_m, fn);
    if (!u) {
        return std::unexpected(u.error());
    }
    if (u->size() != grid->n) {
        return std::unexpected(DomainError{fn, "field length must match grid n"});
    }
    return cfd::integrated_mass(*u, grid->dx);
}

Result<double> eval_cfd_integrated_mass_2d_from_grid(const Matrix<double>& grid_m,
                                                     const Matrix<double>& u_m) {
    constexpr const char* fn = "cfd_integrated_mass_2d";
    auto grid = cfd_grid2d_from_packed_matrix(grid_m, fn);
    if (!grid) {
        return std::unexpected(grid.error());
    }
    auto u = matrix_to_grid(u_m, fn);
    if (!u) {
        return std::unexpected(u.error());
    }
    if (u->size() != grid->ny || (!u->empty() && u->front().size() != grid->nx)) {
        return std::unexpected(DomainError{fn, "field shape must match packed grid nx/ny"});
    }
    return cfd::integrated_mass_2d(*u, grid->dx, grid->dy);
}

Result<Matrix<double>> eval_cfd_constant_velocity(std::size_t n, double v) {
    constexpr const char* fn = "cfd_constant_velocity";
    if (n < 1) {
        return std::unexpected(DomainError{fn, "expected n >= 1"});
    }
    return vector_to_column(cfd::constant_velocity(n, v));
}

Result<Matrix<double>> eval_cfd_upwind_step_2d_from_grid(
    const Matrix<double>& grid_m,
    const Matrix<double>& u_m,
    double vx,
    double vy,
    double dt,
    cfd::BoundaryCondition bc_x,
    cfd::BoundaryCondition bc_y) {
    constexpr const char* fn = "cfd_upwind_step_2d";
    auto grid = cfd_grid2d_from_packed_matrix(grid_m, fn);
    if (!grid) {
        return std::unexpected(grid.error());
    }
    auto u = matrix_to_grid(u_m, fn);
    if (!u) {
        return std::unexpected(u.error());
    }
    if (u->empty() || u->front().empty()) {
        return std::unexpected(DomainError{fn, "expected non-empty 2D field matrix"});
    }
    if (dt <= 0.0 || grid->dx <= 0.0 || grid->dy <= 0.0) {
        return std::unexpected(DomainError{fn, "expected positive dt, dx, and dy"});
    }
    const std::vector<double> vx_field = {vx};
    const std::vector<double> vy_field = {vy};
    const auto u1 = cfd::upwind_fvm_advection_2d(
        *u, vx_field, vy_field, dt, grid->dx, grid->dy, bc_x, bc_y);
    if (u1.empty()) {
        return std::unexpected(
            DomainError{fn, "CFL stability condition violated or invalid input"});
    }
    return grid_to_matrix(u1);
}

Result<Matrix<double>> eval_cfd_run_advection_2d(const Matrix<double>& grid_m,
                                                 const Matrix<double>& u_m, double vx, double vy,
                                                 double t_end, double dt) {
    constexpr const char* fn = "cfd_run_advection_2d";
    auto grid = cfd_grid2d_from_packed_matrix(grid_m, fn);
    if (!grid) {
        return std::unexpected(grid.error());
    }
    auto u0 = matrix_to_grid(u_m, fn);
    if (!u0) {
        return std::unexpected(u0.error());
    }
    if (t_end <= 0.0 || dt <= 0.0) {
        return std::unexpected(DomainError{fn, "expected positive t_end and dt"});
    }
    const std::size_t n_cells = grid->nx * grid->ny;
    const auto vx_field = cfd::constant_velocity(n_cells, vx);
    const auto vy_field = cfd::constant_velocity(n_cells, vy);
    const auto result = cfd::run_advection_2d(
        *u0, vx_field, vy_field, t_end, dt, grid->dx, grid->dy);
    if (result.u.empty()) {
        return std::unexpected(
            DomainError{fn, "CFL stability condition violated or invalid input"});
    }
    return grid_to_matrix(result.u.back());
}

constexpr double kFemMesh3dTag = 271.0;

Matrix<double> pack_fem_mesh3d(const fem::Mesh3D& mesh) {
    const std::size_t n_nodes = mesh.nodes.size();
    const std::size_t n_tet = mesh.tetrahedra.size();
    Matrix<double> out(1 + n_nodes + n_tet, 4, 0.0);
    out(0, 0) = kFemMesh3dTag;
    out(0, 1) = static_cast<double>(n_nodes);
    out(0, 2) = static_cast<double>(n_tet);
    for (std::size_t i = 0; i < n_nodes; ++i) {
        out(1 + i, 0) = mesh.nodes[i][0];
        out(1 + i, 1) = mesh.nodes[i][1];
        out(1 + i, 2) = mesh.nodes[i][2];
    }
    for (std::size_t t = 0; t < n_tet; ++t) {
        const std::size_t row = 1 + n_nodes + t;
        for (int k = 0; k < 4; ++k) {
            out(row, static_cast<std::size_t>(k)) =
                static_cast<double>(mesh.tetrahedra[t][static_cast<std::size_t>(k)]);
        }
    }
    return out;
}

Result<fem::Mesh3D> matrix_to_fem_mesh3d(const Matrix<double>& m, const char* fn) {
    if (m.rows() < 1 || m.cols() < 4 || m(0, 0) != kFemMesh3dTag) {
        return std::unexpected(DomainError{fn, "expected packed fem_mesh3d matrix"});
    }
    const double n_nodes_d = m(0, 1);
    const double n_tet_d = m(0, 2);
    if (n_nodes_d < 4.0 || n_tet_d < 1.0 || std::floor(n_nodes_d) != n_nodes_d ||
        std::floor(n_tet_d) != n_tet_d) {
        return std::unexpected(DomainError{fn, "invalid fem_mesh3d header"});
    }
    const std::size_t n_nodes = static_cast<std::size_t>(n_nodes_d);
    const std::size_t n_tet = static_cast<std::size_t>(n_tet_d);
    if (m.rows() != 1 + n_nodes + n_tet) {
        return std::unexpected(DomainError{fn, "fem_mesh3d row count mismatch"});
    }
    fem::Mesh3D mesh;
    mesh.nodes.resize(n_nodes);
    mesh.tetrahedra.resize(n_tet);
    for (std::size_t i = 0; i < n_nodes; ++i) {
        mesh.nodes[i] = {m(1 + i, 0), m(1 + i, 1), m(1 + i, 2)};
    }
    for (std::size_t t = 0; t < n_tet; ++t) {
        const std::size_t row = 1 + n_nodes + t;
        for (int k = 0; k < 4; ++k) {
            const double idx_d = m(row, static_cast<std::size_t>(k));
            if (idx_d < 0.0 || std::floor(idx_d) != idx_d) {
                return std::unexpected(
                    DomainError{fn, "expected non-negative integer tetrahedron index"});
            }
            mesh.tetrahedra[t][static_cast<std::size_t>(k)] = static_cast<std::size_t>(idx_d);
        }
    }
    return mesh;
}

Result<Matrix<double>> eval_fem_mesh3d_box(
    double x0, double y0, double z0, double x1, double y1, double z1, std::size_t nx,
    std::size_t ny, std::size_t nz) {
    constexpr const char* fn = "fem_mesh3d_box";
    if (nx == 0 || ny == 0 || nz == 0) {
        return std::unexpected(DomainError{fn, "expected positive nx, ny, and nz"});
    }
    auto mesh = fem::mesh3d_box(x0, y0, z0, x1, y1, z1, nx, ny, nz);
    if (!mesh) return std::unexpected(mesh.error());
    return pack_fem_mesh3d(*mesh);
}

Result<Matrix<double>> eval_fem_stiffness_3d(const Matrix<double>& mesh_m) {
    constexpr const char* fn = "fem_stiffness_3d";
    auto mesh = matrix_to_fem_mesh3d(mesh_m, fn);
    if (!mesh) {
        return std::unexpected(mesh.error());
    }
    auto K = fem::assemble_stiffness_3d(*mesh);
    if (!K) return std::unexpected(K.error());
    return col_matrix_to_matrix(*K);
}

Result<Matrix<double>> eval_fem_load_3d(const Matrix<double>& mesh_m, double f_const) {
    constexpr const char* fn = "fem_load_3d";
    auto mesh = matrix_to_fem_mesh3d(mesh_m, fn);
    if (!mesh) {
        return std::unexpected(mesh.error());
    }
    auto load = fem::assemble_load_3d(*mesh, [f_const](double, double, double) { return f_const; });
    if (!load) return std::unexpected(load.error());
    return col_matrix_to_matrix(*load);
}

Matrix<double> pack_cfd_grid3d(const cfd::Grid3D& grid) {
    const size_t ncols = std::max({grid.nx, grid.ny, grid.nz, size_t{12}});
    Matrix<double> out(4, ncols, 0.0);
    out(0, 0) = grid.x0;
    out(0, 1) = grid.x1;
    out(0, 2) = grid.y0;
    out(0, 3) = grid.y1;
    out(0, 4) = grid.z0;
    out(0, 5) = grid.z1;
    out(0, 6) = grid.dx;
    out(0, 7) = grid.dy;
    out(0, 8) = grid.dz;
    out(0, 9) = static_cast<double>(grid.nx);
    out(0, 10) = static_cast<double>(grid.ny);
    out(0, 11) = static_cast<double>(grid.nz);
    for (size_t j = 0; j < grid.nx; ++j) {
        out(1, j) = grid.x[j];
    }
    for (size_t j = 0; j < grid.ny; ++j) {
        out(2, j) = grid.y[j];
    }
    for (size_t j = 0; j < grid.nz; ++j) {
        out(3, j) = grid.z[j];
    }
    return out;
}

Result<cfd::Grid3D> cfd_grid3d_from_packed_matrix(const Matrix<double>& m, const char* fn) {
    if (m.rows() != 4 || m.cols() < 12) {
        return std::unexpected(DomainError{
            fn,
            "expected packed CFD grid3d matrix (4 rows, header [x0..z1,dx,dy,dz,nx,ny,nz])"});
    }
    const double nx_d = m(0, 9);
    const double ny_d = m(0, 10);
    const double nz_d = m(0, 11);
    const int nx_i = static_cast<int>(nx_d);
    const int ny_i = static_cast<int>(ny_d);
    const int nz_i = static_cast<int>(nz_d);
    if (nx_i < 2 || ny_i < 2 || nz_i < 2 || nx_d != nx_i || ny_d != ny_i || nz_d != nz_i) {
        return std::unexpected(
            DomainError{fn, "invalid packed grid nx/ny/nz (expected integers >= 2)"});
    }
    const size_t nx = static_cast<size_t>(nx_i);
    const size_t ny = static_cast<size_t>(ny_i);
    const size_t nz = static_cast<size_t>(nz_i);
    if (m.cols() < std::max({nx, ny, nz})) {
        return std::unexpected(DomainError{fn, "packed grid coordinate rows shorter than nx/ny/nz"});
    }
    cfd::Grid3D grid;
    grid.x0 = m(0, 0);
    grid.x1 = m(0, 1);
    grid.y0 = m(0, 2);
    grid.y1 = m(0, 3);
    grid.z0 = m(0, 4);
    grid.z1 = m(0, 5);
    grid.dx = m(0, 6);
    grid.dy = m(0, 7);
    grid.dz = m(0, 8);
    grid.nx = nx;
    grid.ny = ny;
    grid.nz = nz;
    grid.x.assign(nx, 0.0);
    grid.y.assign(ny, 0.0);
    grid.z.assign(nz, 0.0);
    for (size_t j = 0; j < nx; ++j) {
        grid.x[j] = m(1, j);
    }
    for (size_t j = 0; j < ny; ++j) {
        grid.y[j] = m(2, j);
    }
    for (size_t j = 0; j < nz; ++j) {
        grid.z[j] = m(3, j);
    }
    return grid;
}

Result<std::vector<std::vector<std::vector<double>>>> matrix_to_grid3d(const Matrix<double>& m,
                                                                       std::size_t nx,
                                                                       std::size_t ny,
                                                                       std::size_t nz,
                                                                       const char* fn) {
    if (nx < 1 || ny < 1 || nz < 1) {
        return std::unexpected(DomainError{fn, "expected positive nx, ny, and nz"});
    }
    if (m.rows() != nz * ny || m.cols() != nx) {
        return std::unexpected(DomainError{
            fn, "expected 3D field matrix with rows nz*ny and cols nx"});
    }
    std::vector<std::vector<std::vector<double>>> grid(
        nz, std::vector<std::vector<double>>(ny, std::vector<double>(nx, 0.0)));
    for (std::size_t k = 0; k < nz; ++k) {
        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t i = 0; i < nx; ++i) {
                grid[k][j][i] = m(k * ny + j, i);
            }
        }
    }
    return grid;
}

Result<Matrix<double>> eval_cfd_grid3d(double x0, double x1, double y0, double y1, double z0,
                                      double z1, std::size_t nx, std::size_t ny, std::size_t nz) {
    constexpr const char* fn = "cfd_grid3d";
    if (x1 <= x0 || y1 <= y0 || z1 <= z0) {
        return std::unexpected(DomainError{fn, "expected x1>x0, y1>y0, and z1>z0"});
    }
    if (nx < 2 || ny < 2 || nz < 2) {
        return std::unexpected(DomainError{fn, "expected nx, ny, nz >= 2"});
    }
    const cfd::Grid3D grid = cfd::grid3d(x0, x1, y0, y1, z0, z1, nx, ny, nz);
    if (grid.nx == 0 || grid.ny == 0 || grid.nz == 0) {
        return std::unexpected(DomainError{fn, "invalid grid dimensions"});
    }
    return pack_cfd_grid3d(grid);
}

Result<Matrix<double>> eval_cfd_square_pulse_3d(const Matrix<double>& grid_m, double xc, double yc,
                                                double zc, double width_x, double width_y,
                                                double width_z, double amplitude) {
    constexpr const char* fn = "cfd_square_pulse_3d";
    auto grid = cfd_grid3d_from_packed_matrix(grid_m, fn);
    if (!grid) {
        return std::unexpected(grid.error());
    }
    const auto pulse =
        cfd::square_pulse_3d(*grid, xc, yc, zc, width_x, width_y, width_z, amplitude);
    if (pulse.empty() || pulse.front().empty() || pulse.front().front().empty()) {
        return std::unexpected(DomainError{fn, "failed to build square pulse"});
    }
    return grid3d_to_matrix(pulse);
}

Result<Matrix<double>> eval_cfd_upwind_step_3d(const Matrix<double>& grid_m,
                                               const Matrix<double>& u_m, double vx, double vy,
                                               double vz, double dt,
                                               cfd::BoundaryCondition bc_x,
                                               cfd::BoundaryCondition bc_y,
                                               cfd::BoundaryCondition bc_z) {
    constexpr const char* fn = "cfd_upwind_step_3d";
    auto grid = cfd_grid3d_from_packed_matrix(grid_m, fn);
    if (!grid) {
        return std::unexpected(grid.error());
    }
    auto u = matrix_to_grid3d(u_m, grid->nx, grid->ny, grid->nz, fn);
    if (!u) {
        return std::unexpected(u.error());
    }
    if (dt <= 0.0 || grid->dx <= 0.0 || grid->dy <= 0.0 || grid->dz <= 0.0) {
        return std::unexpected(DomainError{fn, "expected positive dt, dx, dy, and dz"});
    }
    const std::vector<double> vx_field = {vx};
    const std::vector<double> vy_field = {vy};
    const std::vector<double> vz_field = {vz};
    const auto u1 = cfd::upwind_fvm_advection_3d(
        *u, vx_field, vy_field, vz_field, dt, grid->dx, grid->dy, grid->dz, bc_x, bc_y, bc_z);
    if (u1.empty()) {
        return std::unexpected(
            DomainError{fn, "CFL stability condition violated or invalid input"});
    }
    return grid3d_to_matrix(u1);
}

Result<double> eval_cfd_integrated_mass_3d(const Matrix<double>& grid_m,
                                           const Matrix<double>& u_m) {
    constexpr const char* fn = "cfd_integrated_mass_3d";
    auto grid = cfd_grid3d_from_packed_matrix(grid_m, fn);
    if (!grid) {
        return std::unexpected(grid.error());
    }
    auto u = matrix_to_grid3d(u_m, grid->nx, grid->ny, grid->nz, fn);
    if (!u) {
        return std::unexpected(u.error());
    }
    return cfd::integrated_mass_3d(*u, grid->dx, grid->dy, grid->dz);
}

Result<Matrix<double>> eval_gria_gf2n_generate_field(int n) {
    constexpr const char* fn = "gria_gf2n_generate_field";
    const auto field = gria::gf2n::generate_field(n);
    if (field.empty()) {
        return std::unexpected(DomainError{fn, "expected 1 <= n <= 16"});
    }
    std::vector<double> values;
    values.reserve(field.size());
    for (uint64_t v : field) {
        values.push_back(static_cast<double>(v));
    }
    return vector_to_column(values);
}

Result<Matrix<double>> eval_quantum_eigenspectrum(const Matrix<double>& H_m) {
    constexpr const char* fn = "quantum_eigenspectrum";
    auto H = matrix_to_density_matrix(H_m, fn);
    if (!H) {
        return std::unexpected(H.error());
    }
    return vector_to_column(quantum::eigenspectrum(*H));
}

Result<Matrix<double>> eval_quantum_ground_state(const Matrix<double>& H_m) {
    constexpr const char* fn = "quantum_ground_state";
    auto H = matrix_to_density_matrix(H_m, fn);
    if (!H) {
        return std::unexpected(H.error());
    }
    return ket_to_column_matrix(quantum::ground_state(*H));
}

Result<Matrix<double>> eval_quantum_time_evolve_psi(const Matrix<double>& H_m,
                                                      const Matrix<double>& psi_m, double t) {
    constexpr const char* fn = "quantum_time_evolve_psi";
    auto H = matrix_to_density_matrix(H_m, fn);
    if (!H) {
        return std::unexpected(H.error());
    }
    auto psi = matrix_to_ket(psi_m, fn);
    if (!psi) {
        return std::unexpected(psi.error());
    }
    if (H->size() != psi->size()) {
        return std::unexpected(
            DomainError{fn, "expected H and psi with matching dimension"});
    }
    const auto U = quantum::time_evolution_operator(*H, t);
    return ket_to_column_matrix(quantum::op_apply(U, *psi));
}

Result<Matrix<double>> eval_quantum_anticommutator(const Matrix<double>& A_m,
                                                   const Matrix<double>& B_m) {
    auto A = matrix_to_density_matrix(A_m, "quantum_anticommutator");
    if (!A) {
        return std::unexpected(A.error());
    }
    auto B = matrix_to_density_matrix(B_m, "quantum_anticommutator");
    if (!B) {
        return std::unexpected(B.error());
    }
    if (A->size() != B->size()) {
        return std::unexpected(
            DomainError{"quantum_anticommutator", "operators must have same dimension"});
    }
    return density_matrix_to_matrix(quantum::anticommutator(*A, *B));
}

Result<Matrix<double>> eval_quantum_schmidt_decomposition(const Matrix<double>& psi_m, int dim_a,
                                                          int dim_b) {
    constexpr const char* fn = "quantum_schmidt_decomposition";
    auto psi = matrix_to_ket(psi_m, fn);
    if (!psi) {
        return std::unexpected(psi.error());
    }
    return vector_to_column(quantum::schmidt_decomposition(*psi, dim_a, dim_b).coefficients);
}

Result<double> eval_izaac_exponential_mechanism(const Matrix<double>& scores_m, double epsilon,
                                                double sensitivity) {
    constexpr const char* fn = "izaac_exponential_mechanism";
    auto scores = matrix_to_coeff_vector(scores_m, fn);
    if (!scores) {
        return std::unexpected(scores.error());
    }
    if (scores->empty()) {
        return std::unexpected(DomainError{fn, "expected non-empty score vector"});
    }
    auto rng_check = require_session_rng(fn);
    if (!rng_check) {
        return std::unexpected(rng_check.error());
    }
    return static_cast<double>(izaac::diffpriv::exponential_mechanism(
        *scores, epsilon, sensitivity, *izaac::g_session_rng));
}

Result<Matrix<double>> eval_mpc_split(uint64_t secret, int n, int k) {
    constexpr const char* fn = "mpc_split";
    if (n < 2 || k < 2 || k > n) {
        return std::unexpected(DomainError{fn, "expected n>=2 and 2<=k<=n"});
    }
    if (secret >= izaac::mpc::PRIME) {
        return std::unexpected(DomainError{fn, "secret must be < mpc PRIME"});
    }
    auto rng_check = require_session_rng(fn);
    if (!rng_check) {
        return std::unexpected(rng_check.error());
    }
    const auto shares = izaac::mpc::split_secret(secret, n, k, *izaac::g_session_rng);
    Matrix<double> out(shares.size(), 3, 0.0);
    for (size_t i = 0; i < shares.size(); ++i) {
        out(i, 0) = static_cast<double>(shares[i].x);
        out(i, 1) = static_cast<double>(shares[i].y & 0xFFFFFFFFu);
        out(i, 2) = static_cast<double>(shares[i].y >> 32);
    }
    return out;
}

Result<double> eval_mpc_reconstruct(const Matrix<double>& shares_m) {
    constexpr const char* fn = "mpc_reconstruct";
    if (shares_m.cols() < 3 || shares_m.rows() < 2) {
        return std::unexpected(DomainError{fn, "expected n×3 share matrix [x,y_lo,y_hi]"});
    }
    std::vector<izaac::mpc::Share> shares;
    shares.reserve(shares_m.rows());
    for (size_t i = 0; i < shares_m.rows(); ++i) {
        const double x_d = shares_m(i, 0);
        const double y_lo = shares_m(i, 1);
        const double y_hi = shares_m(i, 2);
        if (x_d != std::floor(x_d) || y_lo != std::floor(y_lo) || y_hi != std::floor(y_hi)) {
            return std::unexpected(DomainError{fn, "share components must be integers"});
        }
        const uint64_t y =
            (static_cast<uint64_t>(y_hi) << 32) | static_cast<uint64_t>(y_lo);
        shares.push_back({static_cast<int>(x_d), y});
    }
    auto secret = izaac::mpc::reconstruct_secret(shares);
    if (!secret) {
        return std::unexpected(secret.error());
    }
    return static_cast<double>(*secret);
}

Result<Matrix<double>> eval_simulate_gbm_path(double s0, double mu, double sigma, double dt,
                                              size_t steps) {
    constexpr const char* fn = "simulate_gbm_path";
    if (s0 <= 0.0 || sigma < 0.0 || dt <= 0.0 || steps < 1) {
        return std::unexpected(DomainError{fn, "expected s0>0, sigma>=0, dt>0, steps>=1"});
    }
    auto rng_check = require_session_rng(fn);
    if (!rng_check) {
        return std::unexpected(rng_check.error());
    }
    return vector_to_column(izaac::backtest::simulate_gbm_path(s0, mu, sigma, dt, steps,
                                                              *izaac::g_session_rng));
}

Result<Matrix<double>> eval_run_backtest(const Matrix<double>& prices_m,
                                         const Matrix<double>& positions_m,
                                         double initial_capital) {
    constexpr const char* fn = "run_backtest";
    auto prices = matrix_to_coeff_vector(prices_m, fn);
    if (!prices) {
        return std::unexpected(prices.error());
    }
    if (prices->size() < 2) {
        return std::unexpected(DomainError{fn, "expected price vector length >= 2"});
    }
    auto pos_vec = matrix_to_coeff_vector(positions_m, fn);
    if (!pos_vec) {
        return std::unexpected(pos_vec.error());
    }
    if (pos_vec->size() != prices->size()) {
        return std::unexpected(DomainError{fn, "positions length must match prices"});
    }
    std::vector<int> positions;
    positions.reserve(pos_vec->size());
    for (double v : *pos_vec) {
        if (v != std::floor(v)) {
            return std::unexpected(DomainError{fn, "positions must be integers"});
        }
        positions.push_back(static_cast<int>(v));
    }
    const auto bt = izaac::backtest::run_backtest(*prices, positions, initial_capital);
    Matrix<double> out(1, 4, 0.0);
    out(0, 0) = bt.total_return;
    out(0, 1) = bt.max_drawdown;
    out(0, 2) = bt.sharpe_ratio;
    out(0, 3) = static_cast<double>(bt.equity_curve.size());
    return out;
}

Result<Matrix<double>> eval_run_backtest_equity(const Matrix<double>& prices_m,
                                                const Matrix<double>& positions_m,
                                                double initial_capital) {
    constexpr const char* fn = "run_backtest_equity";
    auto prices = matrix_to_coeff_vector(prices_m, fn);
    if (!prices) {
        return std::unexpected(prices.error());
    }
    if (prices->size() < 2) {
        return std::unexpected(DomainError{fn, "expected price vector length >= 2"});
    }
    auto pos_vec = matrix_to_coeff_vector(positions_m, fn);
    if (!pos_vec) {
        return std::unexpected(pos_vec.error());
    }
    if (pos_vec->size() != prices->size()) {
        return std::unexpected(DomainError{fn, "positions length must match prices"});
    }
    std::vector<int> positions;
    positions.reserve(pos_vec->size());
    for (double v : *pos_vec) {
        if (v != std::floor(v)) {
            return std::unexpected(DomainError{fn, "positions must be integers"});
        }
        positions.push_back(static_cast<int>(v));
    }
    const auto bt = izaac::backtest::run_backtest(*prices, positions, initial_capital);
    return vector_to_column(bt.equity_curve);
}

Result<double> eval_run_backtest_sharpe(const Matrix<double>& prices_m,
                                        const Matrix<double>& positions_m,
                                        double initial_capital) {
    constexpr const char* fn = "run_backtest_sharpe";
    auto prices = matrix_to_coeff_vector(prices_m, fn);
    if (!prices) {
        return std::unexpected(prices.error());
    }
    if (prices->size() < 2) {
        return std::unexpected(DomainError{fn, "expected price vector length >= 2"});
    }
    auto pos_vec = matrix_to_coeff_vector(positions_m, fn);
    if (!pos_vec) {
        return std::unexpected(pos_vec.error());
    }
    if (pos_vec->size() != prices->size()) {
        return std::unexpected(DomainError{fn, "positions length must match prices"});
    }
    std::vector<int> positions;
    positions.reserve(pos_vec->size());
    for (double v : *pos_vec) {
        if (v != std::floor(v)) {
            return std::unexpected(DomainError{fn, "positions must be integers"});
        }
        positions.push_back(static_cast<int>(v));
    }
    const auto bt = izaac::backtest::run_backtest(*prices, positions, initial_capital);
    return bt.sharpe_ratio;
}

Result<double> eval_run_backtest_max_drawdown(const Matrix<double>& prices_m,
                                              const Matrix<double>& positions_m,
                                              double initial_capital) {
    constexpr const char* fn = "run_backtest_max_drawdown";
    auto prices = matrix_to_coeff_vector(prices_m, fn);
    if (!prices) {
        return std::unexpected(prices.error());
    }
    if (prices->size() < 2) {
        return std::unexpected(DomainError{fn, "expected price vector length >= 2"});
    }
    auto pos_vec = matrix_to_coeff_vector(positions_m, fn);
    if (!pos_vec) {
        return std::unexpected(pos_vec.error());
    }
    if (pos_vec->size() != prices->size()) {
        return std::unexpected(DomainError{fn, "positions length must match prices"});
    }
    std::vector<int> positions;
    positions.reserve(pos_vec->size());
    for (double v : *pos_vec) {
        if (v != std::floor(v)) {
            return std::unexpected(DomainError{fn, "positions must be integers"});
        }
        positions.push_back(static_cast<int>(v));
    }
    const auto bt = izaac::backtest::run_backtest(*prices, positions, initial_capital);
    return bt.max_drawdown;
}

Result<double> eval_run_backtest_total_return(const Matrix<double>& prices_m,
                                              const Matrix<double>& positions_m,
                                              double initial_capital) {
    constexpr const char* fn = "run_backtest_total_return";
    auto prices = matrix_to_coeff_vector(prices_m, fn);
    if (!prices) {
        return std::unexpected(prices.error());
    }
    if (prices->size() < 2) {
        return std::unexpected(DomainError{fn, "expected price vector length >= 2"});
    }
    auto pos_vec = matrix_to_coeff_vector(positions_m, fn);
    if (!pos_vec) {
        return std::unexpected(pos_vec.error());
    }
    if (pos_vec->size() != prices->size()) {
        return std::unexpected(DomainError{fn, "positions length must match prices"});
    }
    std::vector<int> positions;
    positions.reserve(pos_vec->size());
    for (double v : *pos_vec) {
        if (v != std::floor(v)) {
            return std::unexpected(DomainError{fn, "positions must be integers"});
        }
        positions.push_back(static_cast<int>(v));
    }
    const auto bt = izaac::backtest::run_backtest(*prices, positions, initial_capital);
    return bt.total_return;
}

Result<Matrix<double>> eval_izaac_vrf_keygen() {
    const auto key = izaac::keygen();
    Matrix<double> out(2, 32, 0.0);
    for (size_t i = 0; i < 32; ++i) {
        out(0, i) = static_cast<double>(key.private_key[i]);
        out(1, i) = static_cast<double>(key.public_key[i]);
    }
    return out;
}

Result<Matrix<double>> eval_izaac_fuzz_mutate(const Matrix<double>& input_m, size_t max_edits) {
    constexpr const char* fn = "izaac_fuzz_mutate";
    auto bytes_vec = matrix_to_coeff_vector(input_m, fn);
    if (!bytes_vec) {
        return std::unexpected(bytes_vec.error());
    }
    if (bytes_vec->empty()) {
        return std::unexpected(DomainError{fn, "expected non-empty byte vector"});
    }
    std::vector<uint8_t> input;
    input.reserve(bytes_vec->size());
    for (double v : *bytes_vec) {
        if (v < 0.0 || v > 255.0 || v != std::floor(v)) {
            return std::unexpected(DomainError{fn, "expected byte values in [0,255]"});
        }
        input.push_back(static_cast<uint8_t>(v));
    }
    auto rng_check = require_session_rng(fn);
    if (!rng_check) {
        return std::unexpected(rng_check.error());
    }
    const auto mutated = izaac::fuzz::mutate(input, *izaac::g_session_rng, max_edits);
    std::vector<double> out_vals(mutated.begin(), mutated.end());
    return vector_to_column(out_vals);
}

Result<izaac::VRFKey> vrf_key_from_matrix(const Matrix<double>& m, const char* fn) {
    if (m.rows() < 2 || m.cols() < 32) {
        return std::unexpected(DomainError{fn, "expected 2x32 VRF key matrix from izaac_vrf_keygen"});
    }
    izaac::VRFKey key{};
    for (size_t i = 0; i < 32; ++i) {
        const double priv = m(0, i);
        const double pub = m(1, i);
        if (priv != std::floor(priv) || pub != std::floor(pub) || priv < 0.0 || priv > 255.0 ||
            pub < 0.0 || pub > 255.0) {
            return std::unexpected(DomainError{fn, "expected byte values in [0,255] for key rows"});
        }
        key.private_key[i] = static_cast<uint8_t>(priv);
        key.public_key[i] = static_cast<uint8_t>(pub);
    }
    return key;
}

Result<std::array<uint8_t, 32>> key32_from_matrix(const Matrix<double>& m, const char* fn) {
    std::vector<double> flat;
    flat.reserve(m.rows() * m.cols());
    for (size_t i = 0; i < m.rows(); ++i) {
        for (size_t j = 0; j < m.cols(); ++j) {
            flat.push_back(m(i, j));
        }
    }
    if (flat.size() != 32) {
        return std::unexpected(DomainError{fn, "expected 32-byte key matrix"});
    }
    std::array<uint8_t, 32> key{};
    for (size_t i = 0; i < 32; ++i) {
        if (flat[i] != std::floor(flat[i]) || flat[i] < 0.0 || flat[i] > 255.0) {
            return std::unexpected(DomainError{fn, "expected byte values in [0,255] for key"});
        }
        key[i] = static_cast<uint8_t>(flat[i]);
    }
    return key;
}

Result<std::vector<uint8_t>> byte_vector_from_matrix(const Matrix<double>& m, const char* fn) {
    auto bytes_vec = matrix_to_coeff_vector(m, fn);
    if (!bytes_vec) {
        return std::unexpected(bytes_vec.error());
    }
    if (bytes_vec->empty()) {
        return std::unexpected(DomainError{fn, "expected non-empty byte vector"});
    }
    std::vector<uint8_t> out;
    out.reserve(bytes_vec->size());
    for (double v : *bytes_vec) {
        if (v != std::floor(v) || v < 0.0 || v > 255.0) {
            return std::unexpected(DomainError{fn, "expected byte values in [0,255]"});
        }
        out.push_back(static_cast<uint8_t>(v));
    }
    return out;
}

Result<Matrix<double>> pack_vrf_proof(const izaac::VRFProof& proof) {
    std::vector<uint8_t> packed;
    packed.reserve(proof.proof.size() + proof.output.size());
    packed.insert(packed.end(), proof.proof.begin(), proof.proof.end());
    packed.insert(packed.end(), proof.output.begin(), proof.output.end());
    return bytes_to_matrix_col(packed);
}

Result<izaac::VRFProof> vrf_proof_from_matrix(const Matrix<double>& m, const char* fn) {
    auto bytes = byte_vector_from_matrix(m, fn);
    if (!bytes) {
        return std::unexpected(bytes.error());
    }
    if (bytes->size() != 144) {
        return std::unexpected(DomainError{fn, "expected 144-byte packed VRF proof matrix"});
    }
    izaac::VRFProof proof{};
    for (size_t i = 0; i < 80; ++i) {
        proof.proof[i] = (*bytes)[i];
    }
    for (size_t i = 0; i < 64; ++i) {
        proof.output[i] = (*bytes)[80 + i];
    }
    return proof;
}

Result<Matrix<double>> pack_izaac_ciphertext(const izaac::crypto::CipherText& ct) {
    std::vector<uint8_t> packed = ct.data;
    packed.insert(packed.end(), ct.tag.begin(), ct.tag.end());
    return bytes_to_matrix_col(packed);
}

Result<izaac::crypto::CipherText> izaac_ciphertext_from_matrix(const Matrix<double>& m,
                                                               const char* fn) {
    auto bytes = byte_vector_from_matrix(m, fn);
    if (!bytes) {
        return std::unexpected(bytes.error());
    }
    if (bytes->size() < 48) {
        return std::unexpected(DomainError{fn, "expected ciphertext with at least 48 bytes"});
    }
    izaac::crypto::CipherText ct{};
    ct.data.assign(bytes->begin(), bytes->end() - 32);
    for (size_t i = 0; i < 32; ++i) {
        ct.tag[i] = (*bytes)[bytes->size() - 32 + i];
    }
    return ct;
}

Result<Matrix<double>> eval_izaac_vrf_prove(const Matrix<double>& key_m,
                                            const Matrix<double>& msg_m) {
    constexpr const char* fn = "izaac_vrf_prove";
    auto key = vrf_key_from_matrix(key_m, fn);
    if (!key) {
        return std::unexpected(key.error());
    }
    auto msg = byte_vector_from_matrix(msg_m, fn);
    if (!msg) {
        return std::unexpected(msg.error());
    }
    return pack_vrf_proof(izaac::prove(*key, *msg));
}

Result<double> eval_izaac_vrf_verify(const Matrix<double>& pub_m, const Matrix<double>& msg_m,
                                     const Matrix<double>& proof_m) {
    constexpr const char* fn = "izaac_vrf_verify";
    auto pub_key = key32_from_matrix(pub_m, fn);
    if (!pub_key) {
        return std::unexpected(pub_key.error());
    }
    auto msg = byte_vector_from_matrix(msg_m, fn);
    if (!msg) {
        return std::unexpected(msg.error());
    }
    auto proof = vrf_proof_from_matrix(proof_m, fn);
    if (!proof) {
        return std::unexpected(proof.error());
    }
    return izaac::verify(*pub_key, *msg, *proof) ? 1.0 : 0.0;
}

Result<Matrix<double>> eval_izaac_encrypt(const Matrix<double>& plaintext_m,
                                          const Matrix<double>& key_m) {
    constexpr const char* fn = "izaac_encrypt";
    auto plaintext = byte_vector_from_matrix(plaintext_m, fn);
    if (!plaintext) {
        return std::unexpected(plaintext.error());
    }
    auto key = key32_from_matrix(key_m, fn);
    if (!key) {
        return std::unexpected(key.error());
    }
    return pack_izaac_ciphertext(izaac::crypto::encrypt(*plaintext, *key));
}

Result<Matrix<double>> eval_izaac_decrypt(const Matrix<double>& ciphertext_m,
                                          const Matrix<double>& key_m) {
    constexpr const char* fn = "izaac_decrypt";
    auto ct = izaac_ciphertext_from_matrix(ciphertext_m, fn);
    if (!ct) {
        return std::unexpected(ct.error());
    }
    auto key = key32_from_matrix(key_m, fn);
    if (!key) {
        return std::unexpected(key.error());
    }
    auto plain = izaac::crypto::decrypt(*ct, *key);
    if (!plain) {
        return std::unexpected(plain.error());
    }
    return bytes_to_matrix_col(*plain);
}

Result<Matrix<double>> eval_izaac_randn_matrix(size_t rows, size_t cols) {
    constexpr const char* fn = "izaac_randn_matrix";
    if (rows < 1 || cols < 1) {
        return std::unexpected(DomainError{fn, "expected positive rows and cols"});
    }
    auto rng_check = require_session_rng(fn);
    if (!rng_check) {
        return std::unexpected(rng_check.error());
    }
    return izaac::randn_matrix(rows, cols);
}

Result<double> eval_quantum_schmidt_number(const Matrix<double>& psi_m, int dim_a, int dim_b) {
    auto psi = matrix_to_ket(psi_m, "quantum_schmidt_number");
    if (!psi) {
        return std::unexpected(psi.error());
    }
    return static_cast<double>(quantum::schmidt_number(*psi, dim_a, dim_b));
}

Result<Matrix<double>> eval_quantum_ket_tensor_product(const Matrix<double>& psi1_m,
                                                       const Matrix<double>& psi2_m) {
    auto psi1 = matrix_to_ket(psi1_m, "quantum_ket_tensor_product");
    if (!psi1) {
        return std::unexpected(psi1.error());
    }
    auto psi2 = matrix_to_ket(psi2_m, "quantum_ket_tensor_product");
    if (!psi2) {
        return std::unexpected(psi2.error());
    }
    return ket_to_column_matrix(quantum::tensor_product_states(*psi1, *psi2));
}

Result<Matrix<double>> eval_quantum_outer(const Matrix<double>& ket_m,
                                          const Matrix<double>& bra_m) {
    auto ket = matrix_to_ket(ket_m, "quantum_outer");
    if (!ket) {
        return std::unexpected(ket.error());
    }
    auto bra = matrix_to_ket(bra_m, "quantum_outer");
    if (!bra) {
        return std::unexpected(bra.error());
    }
    return density_matrix_to_matrix(quantum::outer(*ket, *bra));
}

Result<Matrix<double>> eval_cfd_run_advection_3d(const Matrix<double>& grid_m,
                                                 const Matrix<double>& u_m, double vx, double vy,
                                                 double vz, double t_end, double dt) {
    constexpr const char* fn = "cfd_run_advection_3d";
    auto grid = cfd_grid3d_from_packed_matrix(grid_m, fn);
    if (!grid) {
        return std::unexpected(grid.error());
    }
    auto u0 = matrix_to_grid3d(u_m, grid->nx, grid->ny, grid->nz, fn);
    if (!u0) {
        return std::unexpected(u0.error());
    }
    if (t_end <= 0.0 || dt <= 0.0) {
        return std::unexpected(DomainError{fn, "expected positive t_end and dt"});
    }
    const std::size_t n_cells = grid->nx * grid->ny * grid->nz;
    const auto vx_field = cfd::constant_velocity(n_cells, vx);
    const auto vy_field = cfd::constant_velocity(n_cells, vy);
    const auto vz_field = cfd::constant_velocity(n_cells, vz);
    const auto result = cfd::run_advection_3d(
        *u0, vx_field, vy_field, vz_field, t_end, dt, grid->dx, grid->dy, grid->dz);
    if (result.u.empty()) {
        return std::unexpected(
            DomainError{fn, "CFL stability condition violated or invalid input"});
    }
    return grid3d_to_matrix(result.u.back());
}

Result<Matrix<double>> eval_quantum_dagger(const Matrix<double>& op_m) {
    constexpr const char* fn = "quantum_dagger";
    auto op = matrix_to_density_matrix(op_m, fn);
    if (!op) {
        return std::unexpected(op.error());
    }
    return density_matrix_to_matrix(quantum::dagger(*op));
}

Result<Matrix<double>> eval_quantum_matmul_dm(const Matrix<double>& A_m,
                                              const Matrix<double>& B_m) {
    constexpr const char* fn = "quantum_matmul_dm";
    auto A = matrix_to_density_matrix(A_m, fn);
    if (!A) {
        return std::unexpected(A.error());
    }
    auto B = matrix_to_density_matrix(B_m, fn);
    if (!B) {
        return std::unexpected(B.error());
    }
    if (A->size() != B->size()) {
        return std::unexpected(
            DomainError{fn, "operators must have same dimension"});
    }
    return density_matrix_to_matrix(quantum::matmul_dm(*A, *B));
}

Result<Matrix<double>> eval_izaac_rand_matrix(size_t rows, size_t cols) {
    constexpr const char* fn = "izaac_rand_matrix";
    if (rows < 1 || cols < 1) {
        return std::unexpected(DomainError{fn, "expected positive rows and cols"});
    }
    auto rng_check = require_session_rng(fn);
    if (!rng_check) {
        return std::unexpected(rng_check.error());
    }
    return izaac::rand_matrix(rows, cols);
}

constexpr double kQuantumSchmidtBasesTag = 282.0;

Result<Matrix<double>> eval_quantum_schmidt_bases(const Matrix<double>& psi_m, int dim_a,
                                                  int dim_b) {
    constexpr const char* fn = "quantum_schmidt_bases";
    auto psi = matrix_to_ket(psi_m, fn);
    if (!psi) {
        return std::unexpected(psi.error());
    }
    const auto decomp = quantum::schmidt_decomposition(*psi, dim_a, dim_b);
    const std::size_t rank = decomp.coefficients.size();
    if (rank == 0) {
        return std::unexpected(DomainError{fn, "empty Schmidt decomposition"});
    }
    const std::size_t cols = std::max(rank, size_t{3});
    Matrix<double> out(1 + static_cast<std::size_t>(dim_a) + static_cast<std::size_t>(dim_b),
                       cols, 0.0);
    out(0, 0) = kQuantumSchmidtBasesTag;
    out(0, 1) = static_cast<double>(dim_a);
    out(0, 2) = static_cast<double>(dim_b);
    for (std::size_t k = 0; k < rank; ++k) {
        for (int i = 0; i < dim_a; ++i) {
            out(1 + static_cast<std::size_t>(i), k) = decomp.basis_a[k][static_cast<std::size_t>(i)].real();
        }
        for (int j = 0; j < dim_b; ++j) {
            out(1 + static_cast<std::size_t>(dim_a) + static_cast<std::size_t>(j), k) =
                decomp.basis_b[k][static_cast<std::size_t>(j)].real();
        }
    }
    return out;
}

constexpr double kQuantumBellStatesTag = 283.0;

Result<Matrix<double>> eval_quantum_bell_states() {
    constexpr const char* fn = "quantum_bell_states";
    const auto states = quantum::bell_states();
    if (states.size() != 4) {
        return std::unexpected(DomainError{fn, "expected four Bell states"});
    }
    constexpr std::size_t dim = 4;
    Matrix<double> out(1 + dim, 4, 0.0);
    out(0, 0) = kQuantumBellStatesTag;
    for (std::size_t k = 0; k < states.size(); ++k) {
        if (states[k].size() != dim) {
            return std::unexpected(DomainError{fn, "unexpected Bell state dimension"});
        }
        for (std::size_t i = 0; i < dim; ++i) {
            out(1 + i, k) = states[k][i].real();
        }
    }
    return out;
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

Result<Matrix<double>> eval_poly_lagrange(const Matrix<double>& xs_m, const Matrix<double>& ys_m) {
    auto xs = matrix_to_coeff_vector(xs_m, "poly_lagrange");
    if (!xs) {
        return std::unexpected(xs.error());
    }
    auto ys = matrix_to_coeff_vector(ys_m, "poly_lagrange");
    if (!ys) {
        return std::unexpected(ys.error());
    }
    if (xs->empty() || ys->empty()) {
        return std::unexpected(
            DomainError{"poly_lagrange", "expected non-empty node/value vectors"});
    }
    if (xs->size() != ys->size()) {
        return std::unexpected(
            DomainError{"poly_lagrange", "node and value vector length mismatch"});
    }
    return vector_to_column(poly::poly_lagrange(*xs, *ys));
}

Result<Matrix<double>> eval_poly_interp_newton(const Matrix<double>& xs_m,
                                               const Matrix<double>& ys_m) {
    auto xs = matrix_to_coeff_vector(xs_m, "poly_interp_newton");
    if (!xs) {
        return std::unexpected(xs.error());
    }
    auto ys = matrix_to_coeff_vector(ys_m, "poly_interp_newton");
    if (!ys) {
        return std::unexpected(ys.error());
    }
    if (xs->empty() || ys->empty()) {
        return std::unexpected(
            DomainError{"poly_interp_newton", "expected non-empty node/value vectors"});
    }
    if (xs->size() != ys->size()) {
        return std::unexpected(
            DomainError{"poly_interp_newton", "node and value vector length mismatch"});
    }
    const auto coeffs = poly::interp_newton(*xs, *ys);
    if (coeffs.empty()) {
        return std::unexpected(
            DomainError{"poly_interp_newton", "invalid interpolation nodes or values"});
    }
    return vector_to_column(coeffs);
}

Result<Matrix<double>> eval_poly_roots(const Matrix<double>& coeffs_m) {
    auto coeffs = matrix_to_coeff_vector(coeffs_m, "poly_roots");
    if (!coeffs) {
        return std::unexpected(coeffs.error());
    }
    if (coeffs->empty()) {
        return std::unexpected(
            DomainError{"poly_roots", "expected non-empty coefficient vector"});
    }
    const auto roots = poly::poly_roots(*coeffs);
    Matrix<double> out(roots.size(), 2);
    for (size_t i = 0; i < roots.size(); ++i) {
        out(i, 0) = roots[i].real();
        out(i, 1) = roots[i].imag();
    }
    return out;
}

Result<Matrix<double>> eval_poly_fit(const Matrix<double>& xs_m, const Matrix<double>& ys_m,
                                     int degree) {
    auto xs = matrix_to_coeff_vector(xs_m, "poly_fit");
    if (!xs) {
        return std::unexpected(xs.error());
    }
    auto ys = matrix_to_coeff_vector(ys_m, "poly_fit");
    if (!ys) {
        return std::unexpected(ys.error());
    }
    if (xs->empty() || ys->empty()) {
        return std::unexpected(
            DomainError{"poly_fit", "expected non-empty node/value vectors"});
    }
    if (xs->size() != ys->size()) {
        return std::unexpected(
            DomainError{"poly_fit", "node and value vector length mismatch"});
    }
    if (degree < 0) {
        return std::unexpected(DomainError{"poly_fit", "expected non-negative degree"});
    }
    const auto coeffs = poly::poly_fit(*xs, *ys, degree);
    if (coeffs.empty()) {
        return std::unexpected(DomainError{"poly_fit", "fit failed"});
    }
    return vector_to_column(coeffs);
}

Result<Matrix<double>> eval_poly_interp_hermite(const Matrix<double>& xs_m,
                                                const Matrix<double>& ys_m,
                                                const Matrix<double>& dys_m) {
    auto xs = matrix_to_coeff_vector(xs_m, "poly_interp_hermite");
    if (!xs) {
        return std::unexpected(xs.error());
    }
    auto ys = matrix_to_coeff_vector(ys_m, "poly_interp_hermite");
    if (!ys) {
        return std::unexpected(ys.error());
    }
    auto dys = matrix_to_coeff_vector(dys_m, "poly_interp_hermite");
    if (!dys) {
        return std::unexpected(dys.error());
    }
    if (xs->empty() || ys->empty() || dys->empty()) {
        return std::unexpected(DomainError{
            "poly_interp_hermite", "expected non-empty node/value/derivative vectors"});
    }
    if (xs->size() != ys->size() || xs->size() != dys->size()) {
        return std::unexpected(DomainError{
            "poly_interp_hermite", "node/value/derivative vector length mismatch"});
    }
    const auto coeffs = poly::interp_hermite(*xs, *ys, *dys);
    if (coeffs.empty()) {
        return std::unexpected(DomainError{
            "poly_interp_hermite", "invalid Hermite interpolation nodes or values"});
    }
    return vector_to_column(coeffs);
}

Result<Matrix<double>> eval_poly_gcd(const Matrix<double>& a_m, const Matrix<double>& b_m) {
    auto a = matrix_to_coeff_vector(a_m, "poly_gcd");
    if (!a) {
        return std::unexpected(a.error());
    }
    auto b = matrix_to_coeff_vector(b_m, "poly_gcd");
    if (!b) {
        return std::unexpected(b.error());
    }
    if (a->empty() || b->empty()) {
        return std::unexpected(
            DomainError{"poly_gcd", "expected non-empty coefficient vectors"});
    }
    return vector_to_column(poly::poly_gcd(*a, *b));
}

Result<Matrix<double>> eval_poly_squarefree(const Matrix<double>& coeffs_m) {
    auto coeffs = matrix_to_coeff_vector(coeffs_m, "poly_squarefree");
    if (!coeffs) {
        return std::unexpected(coeffs.error());
    }
    if (coeffs->empty()) {
        return std::unexpected(
            DomainError{"poly_squarefree", "expected non-empty coefficient vector"});
    }
    return vector_to_column(poly::poly_squarefree(*coeffs));
}

Result<Matrix<double>> eval_poly_factor(const Matrix<double>& coeffs_m) {
    constexpr const char* fn = "poly_factor";
    auto coeffs = matrix_to_coeff_vector(coeffs_m, fn);
    if (!coeffs) {
        return std::unexpected(coeffs.error());
    }
    if (coeffs->empty()) {
        return std::unexpected(DomainError{fn, "expected non-empty coefficient vector"});
    }
    const auto factors = poly::poly_factor(*coeffs);
    if (factors.empty()) {
        return Matrix<double>(0, 0);
    }
    size_t max_len = 0;
    for (const auto& f : factors) {
        max_len = std::max(max_len, f.coeffs.size());
    }
    const size_t cols = max_len + 1;
    Matrix<double> out(factors.size(), cols, 0.0);
    for (size_t i = 0; i < factors.size(); ++i) {
        for (size_t j = 0; j < factors[i].coeffs.size(); ++j) {
            out(i, j) = factors[i].coeffs[j];
        }
        out(i, cols - 1) = static_cast<double>(factors[i].multiplicity);
    }
    return out;
}

Result<Matrix<double>> eval_poly_rational_roots(const Matrix<double>& coeffs_m, double tol) {
    constexpr const char* fn = "poly_rational_roots";
    auto coeffs = matrix_to_coeff_vector(coeffs_m, fn);
    if (!coeffs) {
        return std::unexpected(coeffs.error());
    }
    if (coeffs->empty()) {
        return std::unexpected(DomainError{fn, "expected non-empty coefficient vector"});
    }
    const auto roots = poly::poly_rational_roots(*coeffs, tol);
    if (!roots) {
        return std::unexpected(roots.error());
    }
    Matrix<double> out(roots->size(), 2);
    for (size_t i = 0; i < roots->size(); ++i) {
        out(i, 0) = static_cast<double>((*roots)[i].first);
        out(i, 1) = static_cast<double>((*roots)[i].second);
    }
    return out;
}

Result<Matrix<double>> eval_poly_factor_rational(const Matrix<double>& coeffs_m, double tol) {
    constexpr const char* fn = "poly_factor_rational";
    auto coeffs = matrix_to_coeff_vector(coeffs_m, fn);
    if (!coeffs) {
        return std::unexpected(coeffs.error());
    }
    if (coeffs->empty()) {
        return std::unexpected(DomainError{fn, "expected non-empty coefficient vector"});
    }
    const auto fact = poly::poly_factor_rational(*coeffs, tol);
    if (!fact) {
        return std::unexpected(fact.error());
    }
    const size_t m = fact->linear_roots.size();
    const size_t rem_len = fact->remainder.size();
    Matrix<double> out(m + rem_len, 2, 0.0);
    for (size_t i = 0; i < m; ++i) {
        out(i, 0) = static_cast<double>(fact->linear_roots[i].first);
        out(i, 1) = static_cast<double>(fact->linear_roots[i].second);
    }
    for (size_t i = 0; i < rem_len; ++i) {
        out(m + i, 0) = fact->remainder[i];
    }
    return out;
}

Result<Matrix<double>> eval_poly_partial_fractions(const Matrix<double>& num_m,
                                                   const Matrix<double>& den_m) {
    constexpr const char* fn = "poly_partial_fractions";
    auto num = matrix_to_coeff_vector(num_m, fn);
    if (!num) {
        return std::unexpected(num.error());
    }
    auto den = matrix_to_coeff_vector(den_m, fn);
    if (!den) {
        return std::unexpected(den.error());
    }
    if (num->empty() || den->empty()) {
        return std::unexpected(
            DomainError{fn, "expected non-empty numerator and denominator vectors"});
    }
    const auto res = poly::poly_partial_fractions(*num, *den);
    const size_t q_len = res.quotient.size();
    const size_t t_len = res.terms.size();
    const size_t rows = std::max(q_len, t_len);
    if (rows == 0) {
        return Matrix<double>(0, 9);
    }
    Matrix<double> out(rows, 9, 0.0);
    for (size_t i = 0; i < q_len; ++i) {
        out(i, 0) = res.quotient[i];
    }
    for (size_t i = 0; i < t_len; ++i) {
        const auto& term = res.terms[i];
        out(i, 1) = term.A;
        out(i, 2) = term.B;
        out(i, 3) = term.C;
        out(i, 4) = term.r;
        out(i, 5) = term.p;
        out(i, 6) = term.q;
        out(i, 7) = static_cast<double>(term.k);
        out(i, 8) = term.is_quadratic ? 1.0 : 0.0;
    }
    return out;
}

Result<double> eval_poly_root_count(const Matrix<double>& coeffs_m, double a, double b) {
    constexpr const char* fn = "poly_root_count";
    auto coeffs = matrix_to_coeff_vector(coeffs_m, fn);
    if (!coeffs) {
        return std::unexpected(coeffs.error());
    }
    if (coeffs->empty()) {
        return std::unexpected(DomainError{fn, "expected non-empty coefficient vector"});
    }
    return static_cast<double>(poly::poly_root_count(*coeffs, a, b));
}

Result<double> eval_poly_cheb_eval(const Matrix<double>& cheb_m, double x) {
    constexpr const char* fn = "poly_cheb_eval";
    auto cheb = matrix_to_coeff_vector(cheb_m, fn);
    if (!cheb) {
        return std::unexpected(cheb.error());
    }
    if (cheb->empty()) {
        return std::unexpected(DomainError{fn, "expected non-empty Chebyshev coefficient vector"});
    }
    return poly::poly_cheb_eval(*cheb, x);
}

Result<Matrix<double>> eval_poly_cheb_expand(const Matrix<double>& coeffs_m, int n,
                                              double a = -1.0, double b = 1.0) {
    constexpr const char* fn = "poly_cheb_expand";
    auto coeffs = matrix_to_coeff_vector(coeffs_m, fn);
    if (!coeffs) {
        return std::unexpected(coeffs.error());
    }
    if (coeffs->empty()) {
        return std::unexpected(DomainError{fn, "expected non-empty coefficient vector"});
    }
    if (n < 0) {
        return std::unexpected(DomainError{fn, "expected non-negative integer n"});
    }
    const std::vector<double> p = *coeffs;
    auto f = [p](double x) {
        const auto value = poly::poly_eval(p, x);
        return value.empty() ? 0.0 : value[0];
    };
    return vector_to_column(poly::poly_cheb_expand(f, n, a, b));
}

Result<Matrix<double>> eval_poly_monic(const Matrix<double>& coeffs_m) {
    auto coeffs = matrix_to_coeff_vector(coeffs_m, "poly_monic");
    if (!coeffs) {
        return std::unexpected(coeffs.error());
    }
    if (coeffs->empty()) {
        return std::unexpected(
            DomainError{"poly_monic", "expected non-empty coefficient vector"});
    }
    return vector_to_column(poly::poly_monic(*coeffs));
}

Result<Matrix<double>> eval_poly_reverse(const Matrix<double>& coeffs_m) {
    auto coeffs = matrix_to_coeff_vector(coeffs_m, "poly_reverse");
    if (!coeffs) {
        return std::unexpected(coeffs.error());
    }
    if (coeffs->empty()) {
        return std::unexpected(
            DomainError{"poly_reverse", "expected non-empty coefficient vector"});
    }
    return vector_to_column(poly::poly_reverse(*coeffs));
}

Result<Matrix<double>> eval_poly_shift(const Matrix<double>& coeffs_m, double a) {
    auto coeffs = matrix_to_coeff_vector(coeffs_m, "poly_shift");
    if (!coeffs) {
        return std::unexpected(coeffs.error());
    }
    if (coeffs->empty()) {
        return std::unexpected(
            DomainError{"poly_shift", "expected non-empty coefficient vector"});
    }
    return vector_to_column(poly::poly_shift(*coeffs, a));
}

Result<Matrix<double>> eval_poly_scale(const Matrix<double>& coeffs_m, double a) {
    auto coeffs = matrix_to_coeff_vector(coeffs_m, "poly_scale");
    if (!coeffs) {
        return std::unexpected(coeffs.error());
    }
    if (coeffs->empty()) {
        return std::unexpected(
            DomainError{"poly_scale", "expected non-empty coefficient vector"});
    }
    return vector_to_column(poly::poly_scale(*coeffs, a));
}

Result<Matrix<double>> eval_poly_pow(const Matrix<double>& coeffs_m, int n) {
    auto coeffs = matrix_to_coeff_vector(coeffs_m, "poly_pow");
    if (!coeffs) {
        return std::unexpected(coeffs.error());
    }
    if (coeffs->empty()) {
        return std::unexpected(
            DomainError{"poly_pow", "expected non-empty coefficient vector"});
    }
    if (n < 0) {
        return std::unexpected(
            DomainError{"poly_pow", "poly_pow: negative exponent unsupported"});
    }
    auto powered = poly::poly_pow(*coeffs, n);
    if (!powered)
        return std::unexpected(powered.error());
    return vector_to_column(*powered);
}

Result<Matrix<double>> eval_poly_lcm(const Matrix<double>& a_m, const Matrix<double>& b_m) {
    auto a = matrix_to_coeff_vector(a_m, "poly_lcm");
    if (!a) {
        return std::unexpected(a.error());
    }
    auto b = matrix_to_coeff_vector(b_m, "poly_lcm");
    if (!b) {
        return std::unexpected(b.error());
    }
    if (a->empty() || b->empty()) {
        return std::unexpected(
            DomainError{"poly_lcm", "expected non-empty coefficient vectors"});
    }
    return vector_to_column(poly::poly_lcm(*a, *b));
}

Result<Matrix<double>> eval_poly_div_quot(const Matrix<double>& a_m, const Matrix<double>& b_m) {
    auto a = matrix_to_coeff_vector(a_m, "poly_div_quot");
    if (!a) {
        return std::unexpected(a.error());
    }
    auto b = matrix_to_coeff_vector(b_m, "poly_div_quot");
    if (!b) {
        return std::unexpected(b.error());
    }
    if (a->empty() || b->empty()) {
        return std::unexpected(
            DomainError{"poly_div_quot", "expected non-empty coefficient vectors"});
    }
    return vector_to_column(poly::poly_div_quot(*a, *b));
}

Result<Matrix<double>> eval_poly_mod(const Matrix<double>& a_m, const Matrix<double>& b_m) {
    auto a = matrix_to_coeff_vector(a_m, "poly_mod");
    if (!a) {
        return std::unexpected(a.error());
    }
    auto b = matrix_to_coeff_vector(b_m, "poly_mod");
    if (!b) {
        return std::unexpected(b.error());
    }
    if (a->empty() || b->empty()) {
        return std::unexpected(
            DomainError{"poly_mod", "expected non-empty coefficient vectors"});
    }
    return vector_to_column(poly::poly_mod(*a, *b));
}

Result<Matrix<double>> eval_poly_eval_at(const Matrix<double>& coeffs_m,
                                         const Matrix<double>& xs_m) {
    auto coeffs = matrix_to_coeff_vector(coeffs_m, "poly_eval_at");
    if (!coeffs) {
        return std::unexpected(coeffs.error());
    }
    auto xs = matrix_to_coeff_vector(xs_m, "poly_eval_at");
    if (!xs) {
        return std::unexpected(xs.error());
    }
    if (coeffs->empty()) {
        return std::unexpected(
            DomainError{"poly_eval_at", "expected non-empty coefficient vector"});
    }
    return vector_to_column(poly::poly_eval_at(*coeffs, std::span<const double>(*xs)));
}

Result<Matrix<double>> eval_poly_sylvester(const Matrix<double>& p_m, const Matrix<double>& q_m) {
    auto p = matrix_to_coeff_vector(p_m, "poly_sylvester");
    if (!p) {
        return std::unexpected(p.error());
    }
    auto q = matrix_to_coeff_vector(q_m, "poly_sylvester");
    if (!q) {
        return std::unexpected(q.error());
    }
    if (p->empty() || q->empty()) {
        return std::unexpected(
            DomainError{"poly_sylvester", "expected non-empty coefficient vectors"});
    }
    const auto S = poly::poly_sylvester(*p, *q);
    Matrix<double> out(S.rows(), S.cols());
    for (size_t i = 0; i < S.rows(); ++i) {
        for (size_t j = 0; j < S.cols(); ++j) {
            out(i, j) = S(i, j);
        }
    }
    return out;
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

Result<double> eval_stats_partial_correlation(const Matrix<double>& x_m,
                                              const Matrix<double>& y_m,
                                              const Matrix<double>& z_m) {
    auto x = matrix_to_coeff_vector(x_m, "stats_partial_correlation");
    if (!x) {
        return std::unexpected(x.error());
    }
    auto y = matrix_to_coeff_vector(y_m, "stats_partial_correlation");
    if (!y) {
        return std::unexpected(y.error());
    }
    auto z = matrix_to_coeff_vector(z_m, "stats_partial_correlation");
    if (!z) {
        return std::unexpected(z.error());
    }
    if (x->size() != y->size() || x->size() != z->size()) {
        return std::unexpected(
            DomainError{"stats_partial_correlation", "vector length mismatch"});
    }
    if (x->empty()) {
        return std::unexpected(
            DomainError{"stats_partial_correlation", "expected non-empty vectors"});
    }
    return partial_correlation(*x, *y, *z);
}

Result<double> eval_stats_weighted_mean(const Matrix<double>& x_m, const Matrix<double>& w_m) {
    auto x = matrix_to_coeff_vector(x_m, "stats_weighted_mean");
    if (!x) {
        return std::unexpected(x.error());
    }
    auto w = matrix_to_coeff_vector(w_m, "stats_weighted_mean");
    if (!w) {
        return std::unexpected(w.error());
    }
    if (x->size() != w->size()) {
        return std::unexpected(
            DomainError{"stats_weighted_mean", "vector length mismatch"});
    }
    if (x->empty()) {
        return std::unexpected(
            DomainError{"stats_weighted_mean", "expected non-empty vectors"});
    }
    return weighted_mean(*x, *w);
}

Result<double> eval_stats_weighted_variance(const Matrix<double>& x_m, const Matrix<double>& w_m) {
    auto x = matrix_to_coeff_vector(x_m, "stats_weighted_variance");
    if (!x) {
        return std::unexpected(x.error());
    }
    auto w = matrix_to_coeff_vector(w_m, "stats_weighted_variance");
    if (!w) {
        return std::unexpected(w.error());
    }
    if (x->size() != w->size()) {
        return std::unexpected(
            DomainError{"stats_weighted_variance", "vector length mismatch"});
    }
    if (x->empty()) {
        return std::unexpected(
            DomainError{"stats_weighted_variance", "expected non-empty vectors"});
    }
    return weighted_variance(*x, *w);
}

Result<double> eval_stats_weighted_correlation(const Matrix<double>& x_m,
                                               const Matrix<double>& y_m,
                                               const Matrix<double>& w_m) {
    auto x = matrix_to_coeff_vector(x_m, "stats_weighted_correlation");
    if (!x) {
        return std::unexpected(x.error());
    }
    auto y = matrix_to_coeff_vector(y_m, "stats_weighted_correlation");
    if (!y) {
        return std::unexpected(y.error());
    }
    auto w = matrix_to_coeff_vector(w_m, "stats_weighted_correlation");
    if (!w) {
        return std::unexpected(w.error());
    }
    if (x->size() != y->size() || x->size() != w->size()) {
        return std::unexpected(
            DomainError{"stats_weighted_correlation", "vector length mismatch"});
    }
    if (x->empty()) {
        return std::unexpected(
            DomainError{"stats_weighted_correlation", "expected non-empty vectors"});
    }
    return weighted_correlation(*x, *y, *w);
}

Result<double> eval_stats_bootstrap_mean(const Matrix<double>& x_m, int n_boot, unsigned seed) {
    auto x = matrix_to_coeff_vector(x_m, "stats_bootstrap_mean");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (x->empty()) {
        return std::unexpected(
            DomainError{"stats_bootstrap_mean", "expected non-empty vector"});
    }
    if (n_boot < 1) {
        return std::unexpected(
            DomainError{"stats_bootstrap_mean", "expected positive integer n_boot"});
    }
    return bootstrap_mean(*x, n_boot, seed);
}

Result<double> eval_stats_trimmed_mean(const Matrix<double>& x_m, double frac) {
    auto x = matrix_to_coeff_vector(x_m, "stats_trimmed_mean");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (x->empty()) {
        return std::unexpected(
            DomainError{"stats_trimmed_mean", "expected non-empty vector"});
    }
    return trimmed_mean(*x, frac);
}

Result<double> eval_stats_vif(const Matrix<double>& X_m, double j_d, const char* fn) {
    if (X_m.rows() == 0 || X_m.cols() == 0) {
        return std::unexpected(DomainError{fn, "expected non-empty design matrix X"});
    }
    if (j_d < 0.0 || j_d != std::floor(j_d)) {
        return std::unexpected(DomainError{fn, "expected non-negative integer column index j"});
    }
    const auto j = static_cast<size_t>(j_d);
    std::vector<std::vector<double>> X(X_m.rows(), std::vector<double>(X_m.cols()));
    for (size_t i = 0; i < X_m.rows(); ++i) {
        for (size_t c = 0; c < X_m.cols(); ++c) {
            X[i][c] = X_m(i, c);
        }
    }
    return variance_inflation_factor(X, j);
}

Result<Matrix<double>> eval_stats_arfit(const Matrix<double>& x_m, int p) {
    auto x = matrix_to_coeff_vector(x_m, "stats_arfit");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (x->empty()) {
        return std::unexpected(DomainError{"stats_arfit", "expected non-empty vector"});
    }
    if (p < 1) {
        return std::unexpected(DomainError{"stats_arfit", "expected positive integer p"});
    }
    auto phi = arfit(*x, p);
    if (phi.empty()) {
        return std::unexpected(
            DomainError{"stats_arfit", "insufficient samples for AR(p) fit"});
    }
    return vector_to_column(phi);
}

Result<Matrix<double>> eval_stats_multiple_regression(const Matrix<double>& X_m,
                                                      const Matrix<double>& y_m) {
    if (X_m.rows() == 0 || X_m.cols() == 0) {
        return std::unexpected(
            DomainError{"stats_multiple_regression", "expected non-empty design matrix X"});
    }
    auto y = matrix_to_coeff_vector(y_m, "stats_multiple_regression");
    if (!y) {
        return std::unexpected(y.error());
    }
    if (y->size() != X_m.rows()) {
        return std::unexpected(DomainError{
            "stats_multiple_regression",
            "y length must equal number of rows in X"});
    }
    std::vector<std::vector<double>> X(X_m.rows(), std::vector<double>(X_m.cols()));
    for (size_t i = 0; i < X_m.rows(); ++i) {
        for (size_t j = 0; j < X_m.cols(); ++j) {
            X[i][j] = X_m(i, j);
        }
    }
    auto beta = multiple_regression(X, *y);
    if (beta.empty()) {
        return std::unexpected(
            DomainError{"stats_multiple_regression", "regression failed"});
    }
    return vector_to_column(beta);
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

Result<Matrix<double>> eval_signal_xcorr(const Matrix<double>& a_m, const Matrix<double>& b_m,
                                         int max_lag) {
    auto a = matrix_to_coeff_vector(a_m, "signal_xcorr");
    if (!a) {
        return std::unexpected(a.error());
    }
    auto b = matrix_to_coeff_vector(b_m, "signal_xcorr");
    if (!b) {
        return std::unexpected(b.error());
    }
    if (a->empty() || b->empty()) {
        return std::unexpected(DomainError{"signal_xcorr", "expected non-empty vectors"});
    }
    if (max_lag < 0) {
        return std::unexpected(
            DomainError{"signal_xcorr", "expected non-negative integer max_lag"});
    }
    auto out = xcorr(*a, *b, max_lag);
    if (out.empty()) {
        return std::unexpected(DomainError{"signal_xcorr", "xcorr failed"});
    }
    return vector_to_column(out);
}

Result<Matrix<double>> eval_signal_xcov(const Matrix<double>& a_m, const Matrix<double>& b_m,
                                        int max_lag) {
    auto a = matrix_to_coeff_vector(a_m, "signal_xcov");
    if (!a) {
        return std::unexpected(a.error());
    }
    auto b = matrix_to_coeff_vector(b_m, "signal_xcov");
    if (!b) {
        return std::unexpected(b.error());
    }
    if (a->empty() || b->empty()) {
        return std::unexpected(DomainError{"signal_xcov", "expected non-empty vectors"});
    }
    if (max_lag < 0) {
        return std::unexpected(
            DomainError{"signal_xcov", "expected non-negative integer max_lag"});
    }
    auto out = xcov(*a, *b, max_lag);
    if (out.empty()) {
        return std::unexpected(DomainError{"signal_xcov", "xcov failed"});
    }
    return vector_to_column(out);
}

Result<Matrix<double>> eval_signal_autocorr(const Matrix<double>& x_m, int max_lag) {
    auto x = matrix_to_coeff_vector(x_m, "signal_autocorr");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (x->empty()) {
        return std::unexpected(
            DomainError{"signal_autocorr", "expected non-empty signal vector"});
    }
    if (max_lag < 0) {
        return std::unexpected(
            DomainError{"signal_autocorr", "expected non-negative integer max_lag"});
    }
    auto out = autocorr(*x, max_lag);
    if (out.empty()) {
        return std::unexpected(DomainError{"signal_autocorr", "autocorr failed"});
    }
    return vector_to_column(out);
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

Result<Matrix<double>> eval_graph_min_arborescence(const Matrix<double>& adj_m, int root) {
    auto G = graph_from_adjacency(adj_m, "graph_min_arborescence");
    if (!G) {
        return std::unexpected(G.error());
    }
    if (root < 0 || root >= G->n_vertices()) {
        return std::unexpected(DomainError{"graph_min_arborescence", "root out of range"});
    }
    const auto res = graph::min_arborescence(*G, root);
    const auto& edges = res.edges;
    Matrix<double> out(1 + edges.size(), 3);
    out(0, 0) = res.total_weight;
    out(0, 1) = 0.0;
    out(0, 2) = 0.0;
    for (size_t i = 0; i < edges.size(); ++i) {
        out(1 + i, 0) = static_cast<double>(edges[i].from);
        out(1 + i, 1) = static_cast<double>(edges[i].to);
        out(1 + i, 2) = edges[i].weight;
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

Result<double> eval_graph_is_planar(const Matrix<double>& adj_m) {
    auto G = graph_from_adjacency_undirected(adj_m, "graph_is_planar");
    if (!G) {
        return std::unexpected(G.error());
    }
    return graph::is_planar_k5_k33_check(*G) ? 1.0 : 0.0;
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

Result<Matrix<double>> shortest_path_dist_parent_matrix(const std::vector<double>& dist,
                                                        const std::vector<int>& parent) {
    const size_t n = dist.size();
    Matrix<double> out(n, 2);
    for (size_t i = 0; i < n; ++i) {
        out(i, 0) = dist[i];
        out(i, 1) = static_cast<double>(parent[i]);
    }
    return out;
}

Result<Matrix<double>> eval_graph_dijkstra(const Matrix<double>& adj_m, int source) {
    auto G = graph_from_adjacency(adj_m, "graph_dijkstra");
    if (!G) {
        return std::unexpected(G.error());
    }
    if (source < 0 || source >= G->n_vertices()) {
        return std::unexpected(DomainError{"graph_dijkstra", "source out of range"});
    }
    const auto [dist, parent] = graph::dijkstra(*G, source);
    return shortest_path_dist_parent_matrix(dist, parent);
}

Result<Matrix<double>> eval_graph_bellman_ford(const Matrix<double>& adj_m, int source) {
    auto G = graph_from_adjacency(adj_m, "graph_bellman_ford");
    if (!G) {
        return std::unexpected(G.error());
    }
    if (source < 0 || source >= G->n_vertices()) {
        return std::unexpected(DomainError{"graph_bellman_ford", "source out of range"});
    }
    auto result = graph::bellman_ford(*G, source);
    if (!result) {
        return std::unexpected(result.error());
    }
    const auto& [dist, parent] = *result;
    return shortest_path_dist_parent_matrix(dist, parent);
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

Result<Matrix<double>> eval_signal_cheby2(int order, double rs_db, double cutoff, double fs,
                                          FilterType type) {
    if (order < 1) {
        return std::unexpected(DomainError{"signal_cheby2", "expected order >= 1"});
    }
    const auto coeffs = cheby2(order, rs_db, cutoff, fs, type);
    if (coeffs.b.empty()) {
        return std::unexpected(
            DomainError{"signal_cheby2", "invalid filter design parameters"});
    }
    return iir_coeffs_to_matrix(coeffs);
}

Result<Matrix<double>> eval_signal_cheby1(int order, double rp_db, double cutoff, double fs,
                                          FilterType type) {
    if (order < 1) {
        return std::unexpected(DomainError{"signal_cheby1", "expected order >= 1"});
    }
    const auto coeffs = cheby1(order, rp_db, cutoff, fs, type);
    if (coeffs.b.empty()) {
        return std::unexpected(
            DomainError{"signal_cheby1", "invalid filter design parameters"});
    }
    return iir_coeffs_to_matrix(coeffs);
}

Result<Matrix<double>> eval_signal_firwin(int n_taps, double cutoff, FirWindow window) {
    if (n_taps < 1) {
        return std::unexpected(DomainError{"signal_firwin", "expected n_taps >= 1"});
    }
    const auto taps = firwin(n_taps, cutoff, window);
    if (taps.empty()) {
        return std::unexpected(
            DomainError{"signal_firwin", "invalid FIR design parameters"});
    }
    return vector_to_column(taps);
}

Result<Matrix<double>> eval_signal_firwin_highpass(int n_taps, double cutoff, FirWindow window) {
    if (n_taps < 1 || (n_taps % 2) == 0) {
        return std::unexpected(
            DomainError{"signal_firwin_highpass", "expected odd n_taps >= 1"});
    }
    const auto taps = firwin_highpass(n_taps, cutoff, window);
    if (taps.empty()) {
        return std::unexpected(
            DomainError{"signal_firwin_highpass", "invalid FIR design parameters"});
    }
    return vector_to_column(taps);
}

Result<Matrix<double>> eval_signal_filtfilt(const Matrix<double>& b_m, const Matrix<double>& a_m,
                                            const Matrix<double>& x_m) {
    auto b = matrix_to_coeff_vector(b_m, "signal_filtfilt");
    if (!b) {
        return std::unexpected(b.error());
    }
    auto a = matrix_to_coeff_vector(a_m, "signal_filtfilt");
    if (!a) {
        return std::unexpected(a.error());
    }
    auto x = matrix_to_coeff_vector(x_m, "signal_filtfilt");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (b->empty()) {
        return std::unexpected(
            DomainError{"signal_filtfilt", "expected non-empty b coefficients"});
    }
    if (a->empty()) {
        return std::unexpected(
            DomainError{"signal_filtfilt", "expected non-empty a coefficients"});
    }
    if (x->empty()) {
        return std::unexpected(
            DomainError{"signal_filtfilt", "expected non-empty signal vector"});
    }
    return vector_to_column(filtfilt(*b, *a, *x));
}

Result<Matrix<double>> eval_signal_filter(const Matrix<double>& b_m, const Matrix<double>& a_m,
                                          const Matrix<double>& x_m) {
    auto b = matrix_to_coeff_vector(b_m, "signal_filter");
    if (!b) {
        return std::unexpected(b.error());
    }
    auto a = matrix_to_coeff_vector(a_m, "signal_filter");
    if (!a) {
        return std::unexpected(a.error());
    }
    auto x = matrix_to_coeff_vector(x_m, "signal_filter");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (b->empty()) {
        return std::unexpected(
            DomainError{"signal_filter", "expected non-empty b coefficients"});
    }
    if (a->empty()) {
        return std::unexpected(
            DomainError{"signal_filter", "expected non-empty a coefficients"});
    }
    if (x->empty()) {
        return std::unexpected(
            DomainError{"signal_filter", "expected non-empty signal vector"});
    }
    return vector_to_column(filter(*b, *a, *x));
}

Result<Matrix<double>> eval_signal_sosfilt(const Matrix<double>& sos_m,
                                           const Matrix<double>& x_m) {
    if (sos_m.cols() != 6) {
        return std::unexpected(
            DomainError{"signal_sosfilt", "expected Kx6 SOS matrix"});
    }
    auto x = matrix_to_coeff_vector(x_m, "signal_sosfilt");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (x->empty()) {
        return std::unexpected(
            DomainError{"signal_sosfilt", "expected non-empty signal vector"});
    }
    std::vector<std::array<double, 6>> sos(sos_m.rows());
    for (size_t i = 0; i < sos_m.rows(); ++i) {
        for (size_t j = 0; j < 6; ++j) {
            sos[i][j] = sos_m(i, j);
        }
    }
    return vector_to_column(sosfilt(sos, *x));
}

Result<Matrix<double>> eval_signal_periodogram(const Matrix<double>& x_m, double fs) {
    auto x = matrix_to_coeff_vector(x_m, "signal_periodogram");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (x->empty()) {
        return std::unexpected(
            DomainError{"signal_periodogram", "expected non-empty signal vector"});
    }
    const auto result = periodogram(*x, fs);
    if (!result) {
        return std::unexpected(result.error());
    }
    return psd_result_to_matrix(*result);
}

Result<Matrix<double>> eval_signal_welch_psd(const Matrix<double>& x_m, double fs,
                                             size_t nperseg) {
    auto x = matrix_to_coeff_vector(x_m, "signal_welch_psd");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (x->empty()) {
        return std::unexpected(
            DomainError{"signal_welch_psd", "expected non-empty signal vector"});
    }
    const auto result = welch_psd(*x, fs, nperseg, 0.5);
    if (!result) {
        return std::unexpected(result.error());
    }
    return psd_result_to_matrix(*result);
}

Result<Matrix<double>> eval_signal_coherence(const Matrix<double>& x_m, const Matrix<double>& y_m,
                                             double fs, size_t nperseg) {
    auto x = matrix_to_coeff_vector(x_m, "signal_coherence");
    if (!x) {
        return std::unexpected(x.error());
    }
    auto y = matrix_to_coeff_vector(y_m, "signal_coherence");
    if (!y) {
        return std::unexpected(y.error());
    }
    if (x->empty() || y->empty()) {
        return std::unexpected(
            DomainError{"signal_coherence", "expected non-empty signal vectors"});
    }
    const auto result = coherence(*x, *y, fs, nperseg, 0.5);
    if (!result) {
        return std::unexpected(result.error());
    }
    return coherence_result_to_matrix(*result);
}

Result<Matrix<double>> eval_signal_spectrogram(const Matrix<double>& x_m, double fs) {
    auto x = matrix_to_coeff_vector(x_m, "signal_spectrogram");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (x->empty()) {
        return std::unexpected(
            DomainError{"signal_spectrogram", "expected non-empty signal vector"});
    }
    const size_t segment_len = (x->size() >= 256) ? 256 : x->size();
    const auto result = spectrogram(*x, fs, segment_len, 0.5);
    if (!result) {
        return std::unexpected(result.error());
    }
    // Magnitude STFT: rows = time segments, cols = frequency bins.
    return result->magnitude;
}

Result<Matrix<double>> eval_signal_envelope(const Matrix<double>& x_m) {
    auto x = matrix_to_coeff_vector(x_m, "signal_envelope");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (x->empty()) {
        return std::unexpected(
            DomainError{"signal_envelope", "expected non-empty signal vector"});
    }
    return vector_to_column(envelope(*x));
}

Result<Matrix<double>> eval_signal_hilbert(const Matrix<double>& x_m) {
    auto x = matrix_to_coeff_vector(x_m, "signal_hilbert");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (x->empty()) {
        return std::unexpected(
            DomainError{"signal_hilbert", "expected non-empty signal vector"});
    }
    const auto z = hilbert(*x);
    Matrix<double> out(z.size(), 2);
    for (size_t i = 0; i < z.size(); ++i) {
        out(i, 0) = z[i].real();
        out(i, 1) = z[i].imag();
    }
    return out;
}

Result<Matrix<double>> complex_vec_to_re_im_matrix(const std::vector<std::complex<double>>& z,
                                                   const char* fn) {
    if (z.empty()) {
        return std::unexpected(DomainError{fn, "transform failed"});
    }
    Matrix<double> out(z.size(), 2);
    for (size_t i = 0; i < z.size(); ++i) {
        out(i, 0) = z[i].real();
        out(i, 1) = z[i].imag();
    }
    return out;
}

Result<Matrix<double>> eval_signal_czt(const Matrix<double>& x_m, int m, double w_re, double w_im,
                                       double a_re, double a_im) {
    auto x = matrix_to_coeff_vector(x_m, "signal_czt");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (x->empty()) {
        return std::unexpected(DomainError{"signal_czt", "expected non-empty signal vector"});
    }
    if (m < 1) {
        return std::unexpected(DomainError{"signal_czt", "expected positive integer m"});
    }
    const std::complex<double> w(w_re, w_im);
    const std::complex<double> a(a_re, a_im);
    return complex_vec_to_re_im_matrix(czt(*x, m, w, a), "signal_czt");
}

Result<Matrix<double>> eval_signal_czt_zoom(const Matrix<double>& x_m, double f_start,
                                            double f_stop, int m, double fs) {
    auto x = matrix_to_coeff_vector(x_m, "signal_czt_zoom");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (x->empty()) {
        return std::unexpected(
            DomainError{"signal_czt_zoom", "expected non-empty signal vector"});
    }
    if (m < 1) {
        return std::unexpected(DomainError{"signal_czt_zoom", "expected positive integer m"});
    }
    return complex_vec_to_re_im_matrix(czt_zoom_fft(*x, f_start, f_stop, m, fs),
                                       "signal_czt_zoom");
}

Result<Matrix<double>> eval_signal_instantaneous_freq(const Matrix<double>& x_m, double fs) {
    auto x = matrix_to_coeff_vector(x_m, "signal_instantaneous_freq");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (x->empty()) {
        return std::unexpected(
            DomainError{"signal_instantaneous_freq", "expected non-empty signal vector"});
    }
    return vector_to_column(instantaneous_freq(*x, fs));
}

Result<Matrix<double>> eval_signal_instantaneous_phase(const Matrix<double>& x_m) {
    auto x = matrix_to_coeff_vector(x_m, "signal_instantaneous_phase");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (x->empty()) {
        return std::unexpected(
            DomainError{"signal_instantaneous_phase", "expected non-empty signal vector"});
    }
    return vector_to_column(instantaneous_phase(*x));
}

Result<Matrix<double>> eval_signal_unwrap(const Matrix<double>& x_m) {
    auto x = matrix_to_coeff_vector(x_m, "signal_unwrap");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (x->empty()) {
        return std::unexpected(
            DomainError{"signal_unwrap", "expected non-empty phase vector"});
    }
    return vector_to_column(unwrap(*x));
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

Result<double> eval_stats_ks_norm(const Matrix<double>& x_m, double mu, double sigma) {
    auto x = matrix_to_coeff_vector(x_m, "stats_ks_norm");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (x->empty()) {
        return std::unexpected(DomainError{"stats_ks_norm", "expected non-empty vector"});
    }
    const double mu_copy = mu;
    const double sigma_copy = sigma;
    return ks_test(*x, [mu_copy, sigma_copy](double t) {
        return norm_cdf(t, mu_copy, sigma_copy);
    });
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

Result<Matrix<double>> eval_stats_linear_regression(const Matrix<double>& x_m,
                                                    const Matrix<double>& y_m) {
    auto x = matrix_to_coeff_vector(x_m, "stats_linear_regression");
    if (!x) {
        return std::unexpected(x.error());
    }
    auto y = matrix_to_coeff_vector(y_m, "stats_linear_regression");
    if (!y) {
        return std::unexpected(y.error());
    }
    if (x->empty() || y->empty()) {
        return std::unexpected(
            DomainError{"stats_linear_regression", "expected non-empty vectors"});
    }
    if (x->size() != y->size()) {
        return std::unexpected(
            DomainError{"stats_linear_regression", "vector length mismatch"});
    }
    const auto lr = linear_regression(*x, *y);
    Matrix<double> out(1, 3);
    out(0, 0) = lr.slope;
    out(0, 1) = lr.intercept;
    out(0, 2) = lr.r_squared;
    return out;
}

Result<Matrix<double>> eval_stats_pacf(const Matrix<double>& x_m, int max_lag) {
    auto x = matrix_to_coeff_vector(x_m, "stats_pacf");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (x->empty()) {
        return std::unexpected(DomainError{"stats_pacf", "expected non-empty vector"});
    }
    if (max_lag < 0) {
        return std::unexpected(
            DomainError{"stats_pacf", "expected non-negative integer max_lag"});
    }
    return vector_to_column(pacf(*x, max_lag));
}

Result<Matrix<double>> eval_stats_kde(const Matrix<double>& samples_m,
                                      const Matrix<double>& grid_m, double h,
                                      const char* kernel = "gaussian") {
    auto samples = matrix_to_coeff_vector(samples_m, "stats_kde");
    if (!samples) {
        return std::unexpected(samples.error());
    }
    auto grid = matrix_to_coeff_vector(grid_m, "stats_kde");
    if (!grid) {
        return std::unexpected(grid.error());
    }
    if (samples->empty()) {
        return std::unexpected(DomainError{"stats_kde", "expected non-empty samples"});
    }
    if (grid->empty()) {
        return std::unexpected(DomainError{"stats_kde", "expected non-empty grid"});
    }
    if (!(h > 0.0)) {
        return std::unexpected(DomainError{"stats_kde", "expected positive bandwidth h"});
    }
    return vector_to_column(kde(*samples, *grid, h, kernel));
}

Result<Matrix<double>> eval_stats_bootstrap_ci(const Matrix<double>& x_m) {
    auto x = matrix_to_coeff_vector(x_m, "stats_bootstrap_ci");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (x->empty()) {
        return std::unexpected(
            DomainError{"stats_bootstrap_ci", "expected non-empty vector"});
    }
    const BootstrapResult ci = bootstrap_ci(
        *x, [](std::span<const double> sample) { return mean(sample); });
    Matrix<double> out(1, 4);
    out(0, 0) = ci.point_estimate;
    out(0, 1) = ci.lower;
    out(0, 2) = ci.upper;
    out(0, 3) = ci.std_error;
    return out;
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

Result<Matrix<double>> eval_fft_goertzel(const Matrix<double>& x_m, double f, double fs) {
    auto x = matrix_to_coeff_vector(x_m, "fft_goertzel");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (x->empty()) {
        return std::unexpected(DomainError{"fft_goertzel", "expected non-empty signal vector"});
    }
    const auto bin = goertzel(std::span<const double>(*x), f, fs);
    Matrix<double> out(1, 2);
    out(0, 0) = bin.real();
    out(0, 1) = bin.imag();
    return out;
}

Result<Matrix<double>> eval_sph_harm(int l, int m, double theta, double phi) {
    const std::complex<double> y = sph_harm_y(l, m, theta, phi);
    Matrix<double> out(1, 2);
    out(0, 0) = y.real();
    out(0, 1) = y.imag();
    return out;
}

Result<Matrix<double>> eval_graph_greedy_colour(const Matrix<double>& adj_m) {
    auto G = graph_from_adjacency_undirected(adj_m, "graph_greedy_colour");
    if (!G) {
        return std::unexpected(G.error());
    }
    return int_vector_to_column(graph::greedy_colour(*G));
}

Matrix<double> graph_to_adjacency_matrix(const graph::Graph& G) {
    const int n = G.n_vertices();
    Matrix<double> out(static_cast<size_t>(n), static_cast<size_t>(n));
    for (int u = 0; u < n; ++u) {
        for (const auto& [v, w] : G.neighbors(u)) {
            out(static_cast<size_t>(u), static_cast<size_t>(v)) = w;
        }
    }
    return out;
}

Result<Matrix<double>> eval_graph_k_core_decomposition(const Matrix<double>& adj_m) {
    auto G = graph_from_adjacency_undirected(adj_m, "graph_k_core_decomposition");
    if (!G) {
        return std::unexpected(G.error());
    }
    return int_vector_to_column(graph::k_core_decomposition(*G));
}

Result<Matrix<double>> eval_graph_k_core_subgraph(const Matrix<double>& adj_m, int k) {
    auto G = graph_from_adjacency_undirected(adj_m, "graph_k_core_subgraph");
    if (!G) {
        return std::unexpected(G.error());
    }
    return graph_to_adjacency_matrix(graph::k_core_subgraph(*G, k));
}

Result<double> eval_graph_chromatic_number(const Matrix<double>& adj_m) {
    auto G = graph_from_adjacency_undirected(adj_m, "graph_chromatic_number");
    if (!G) {
        return std::unexpected(G.error());
    }
    if (G->n_vertices() == 0) {
        return 0.0;
    }
    return static_cast<double>(graph::chromatic_number_approx(*G));
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

Result<Matrix<double>> eval_fft_ifft2(const Matrix<double>& spectrum_m) {
    auto spec = matrix_to_complex_spectrum(spectrum_m, "ifft2");
    if (!spec) {
        return std::unexpected(spec.error());
    }
    auto out = ifft2(*spec);
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

Result<Matrix<double>> eval_fft_idst2(const Matrix<double>& x_m) {
    auto x = matrix_to_coeff_vector(x_m, "idst2");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (x->empty()) {
        return std::unexpected(DomainError{"idst2", "expected non-empty vector"});
    }
    auto signal = idst2(*x);
    if (!signal) {
        return std::unexpected(signal.error());
    }
    return vector_to_column(*signal);
}

Result<std::vector<std::vector<double>>> matrix_to_groups(const Matrix<double>& m,
                                                           const char* fn) {
    if (m.rows() < 2) {
        return std::unexpected(
            DomainError{fn, "expected at least two group rows in groups matrix"});
    }
    if (m.cols() < 1) {
        return std::unexpected(DomainError{fn, "expected non-empty group rows"});
    }
    std::vector<std::vector<double>> groups;
    groups.reserve(m.rows());
    for (size_t i = 0; i < m.rows(); ++i) {
        std::vector<double> group;
        group.reserve(m.cols());
        for (size_t j = 0; j < m.cols(); ++j) {
            group.push_back(m(i, j));
        }
        groups.push_back(std::move(group));
    }
    return groups;
}

Result<Matrix<double>> eval_kruskal_wallis(const Matrix<double>& groups_m) {
    auto groups = matrix_to_groups(groups_m, "kruskal_wallis");
    if (!groups) {
        return std::unexpected(groups.error());
    }
    const auto result = kruskal_wallis(*groups);
    Matrix<double> out(3, 1);
    out(0, 0) = result.h_stat;
    out(1, 0) = static_cast<double>(result.df);
    out(2, 0) = result.p_value;
    return out;
}

Result<Matrix<double>> eval_stats_shapiro_wilk(const Matrix<double>& x_m) {
    auto x = matrix_to_coeff_vector(x_m, "stats_shapiro_wilk");
    if (!x) {
        return std::unexpected(x.error());
    }
    const auto result = shapiro_wilk(*x);
    Matrix<double> out(1, 2);
    out(0, 0) = result.w_stat;
    out(0, 1) = result.p_value;
    return out;
}

Result<Matrix<double>> eval_stats_mann_whitney_u(const Matrix<double>& a_m,
                                                  const Matrix<double>& b_m) {
    auto a = matrix_to_coeff_vector(a_m, "stats_mann_whitney_u");
    if (!a) {
        return std::unexpected(a.error());
    }
    auto b = matrix_to_coeff_vector(b_m, "stats_mann_whitney_u");
    if (!b) {
        return std::unexpected(b.error());
    }
    const auto result = mann_whitney_u(*a, *b);
    // MannWhitneyResult exposes U and p; recover |z| from the two-tailed normal p-value.
    double z = 0.0;
    if (result.p_value > 0.0 && result.p_value < 1.0) {
        z = norm_ppf(1.0 - 0.5 * result.p_value, 0.0, 1.0);
    }
    Matrix<double> out(1, 3);
    out(0, 0) = result.u_stat;
    out(0, 1) = z;
    out(0, 2) = result.p_value;
    return out;
}

Result<Matrix<double>> eval_stats_one_way_anova(const Matrix<double>& groups_m) {
    // Groups matrix: each ROW is one group (same convention as kruskal_wallis).
    auto groups = matrix_to_groups(groups_m, "stats_one_way_anova");
    if (!groups) {
        return std::unexpected(groups.error());
    }
    const auto result = one_way_anova(*groups);
    Matrix<double> out(1, 4);
    out(0, 0) = result.f_stat;
    out(0, 1) = result.p_value;
    out(0, 2) = static_cast<double>(result.df_between);
    out(0, 3) = static_cast<double>(result.df_within);
    return out;
}

Result<Matrix<double>> eval_stats_levene(const Matrix<double>& groups_m) {
    // Groups matrix: each ROW is one group (same convention as stats_one_way_anova).
    auto groups = matrix_to_groups(groups_m, "stats_levene");
    if (!groups) {
        return std::unexpected(groups.error());
    }
    const auto result = levene_test(*groups);
    Matrix<double> out(1, 4);
    out(0, 0) = result.f_stat;
    out(0, 1) = result.p_value;
    out(0, 2) = static_cast<double>(result.df_between);
    out(0, 3) = static_cast<double>(result.df_within);
    return out;
}

Result<Matrix<double>> eval_stats_bartlett(const Matrix<double>& groups_m) {
    // Groups matrix: each ROW is one group (same convention as stats_one_way_anova).
    auto groups = matrix_to_groups(groups_m, "stats_bartlett");
    if (!groups) {
        return std::unexpected(groups.error());
    }
    const auto result = bartlett_test(*groups);
    Matrix<double> out(1, 3);
    out(0, 0) = result.chi2_stat;
    out(0, 1) = static_cast<double>(result.df);
    out(0, 2) = result.p_value;
    return out;
}

Result<Matrix<double>> eval_stats_fligner(const Matrix<double>& groups_m) {
    // Groups matrix: each ROW is one group (same convention as stats_one_way_anova).
    auto groups = matrix_to_groups(groups_m, "stats_fligner");
    if (!groups) {
        return std::unexpected(groups.error());
    }
    const auto result = fligner_test(*groups);
    Matrix<double> out(1, 3);
    out(0, 0) = result.chi2_stat;
    out(0, 1) = static_cast<double>(result.df);
    out(0, 2) = result.p_value;
    return out;
}

Result<Matrix<double>> eval_stats_wilcoxon_signed_rank(const Matrix<double>& x_m,
                                                       const Matrix<double>& y_m) {
    auto x = matrix_to_coeff_vector(x_m, "stats_wilcoxon_signed_rank");
    if (!x) {
        return std::unexpected(x.error());
    }
    auto y = matrix_to_coeff_vector(y_m, "stats_wilcoxon_signed_rank");
    if (!y) {
        return std::unexpected(y.error());
    }
    const auto result = wilcoxon_signed_rank(*x, *y);
    Matrix<double> out(1, 4);
    out(0, 0) = result.w_stat;
    out(0, 1) = result.z_stat;
    out(0, 2) = result.p_value;
    out(0, 3) = static_cast<double>(result.n_eff);
    return out;
}

Result<Matrix<double>> eval_stats_friedman(const Matrix<double>& data_m) {
    // Blocks Ãƒâ€” treatments: each ROW is one block (same layout as friedman(data)).
    if (data_m.rows() < 2 || data_m.cols() < 2) {
        return std::unexpected(
            DomainError{"stats_friedman", "expected blocksÃƒâ€”treatments matrix (>=2 rows, >=2 cols)"});
    }
    std::vector<std::vector<double>> data;
    data.reserve(data_m.rows());
    for (size_t i = 0; i < data_m.rows(); ++i) {
        std::vector<double> row;
        row.reserve(data_m.cols());
        for (size_t j = 0; j < data_m.cols(); ++j) {
            row.push_back(data_m(i, j));
        }
        data.push_back(std::move(row));
    }
    const auto result = friedman(data);
    Matrix<double> out(1, 3);
    out(0, 0) = result.chi2_stat;
    out(0, 1) = static_cast<double>(result.df);
    out(0, 2) = result.p_value;
    return out;
}

Result<Matrix<double>> eval_stats_ks_2sample(const Matrix<double>& a_m,
                                             const Matrix<double>& b_m) {
    auto a = matrix_to_coeff_vector(a_m, "stats_ks_2sample");
    if (!a) {
        return std::unexpected(a.error());
    }
    auto b = matrix_to_coeff_vector(b_m, "stats_ks_2sample");
    if (!b) {
        return std::unexpected(b.error());
    }
    const auto result = ks_test_2sample(*a, *b);
    Matrix<double> out(1, 2);
    out(0, 0) = result.d_stat;
    out(0, 1) = result.p_value;
    return out;
}

Result<Matrix<double>> eval_stats_jarque_bera(const Matrix<double>& x_m) {
    auto x = matrix_to_coeff_vector(x_m, "stats_jarque_bera");
    if (!x) {
        return std::unexpected(x.error());
    }
    const auto result = jarque_bera(*x);
    Matrix<double> out(1, 3);
    out(0, 0) = result.jb_stat;
    out(0, 1) = static_cast<double>(result.df);
    out(0, 2) = result.p_value;
    return out;
}

Result<Matrix<double>> eval_stats_ljung_box(const Matrix<double>& x_m, int max_lag) {
    auto x = matrix_to_coeff_vector(x_m, "stats_ljung_box");
    if (!x) {
        return std::unexpected(x.error());
    }
    if (max_lag < 1) {
        return std::unexpected(
            DomainError{"stats_ljung_box", "expected positive integer max_lag"});
    }
    const auto result = ljung_box(*x, max_lag);
    Matrix<double> out(1, 3);
    out(0, 0) = result.q_stat;
    out(0, 1) = static_cast<double>(result.df);
    out(0, 2) = result.p_value;
    return out;
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

Result<Matrix<double>> eval_combo_gray_code(int n) {
    if (n < 0) {
        return std::unexpected(
            DomainError{"combo_gray_code", "expected non-negative integer n"});
    }
    return combo_enum_rows_to_matrix(combo::gray_code(n));
}

Result<Matrix<double>> eval_combo_dyck_paths(int n) {
    if (n < 0) {
        return std::unexpected(
            DomainError{"combo_dyck_paths", "expected non-negative integer n"});
    }
    const auto paths = combo::dyck_paths(n);
    Matrix<double> out(paths.size(), static_cast<size_t>(2 * n));
    for (size_t r = 0; r < paths.size(); ++r) {
        for (size_t c = 0; c < paths[r].size(); ++c) {
            out(r, c) = paths[r][c] == '(' ? 1.0 : -1.0;
        }
    }
    return out;
}

Result<Matrix<double>> eval_combo_necklaces(int n, int k) {
    if (n < 0) {
        return std::unexpected(
            DomainError{"combo_necklaces", "expected non-negative integer n"});
    }
    if (k <= 0) {
        return std::unexpected(
            DomainError{"combo_necklaces", "expected positive integer k"});
    }
    return combo_enum_rows_to_matrix(combo::necklaces(n, k));
}

Result<Matrix<double>> eval_combo_bracelets(int n, int k) {
    if (n < 0) {
        return std::unexpected(
            DomainError{"combo_bracelets", "expected non-negative integer n"});
    }
    if (k <= 0) {
        return std::unexpected(
            DomainError{"combo_bracelets", "expected positive integer k"});
    }
    return combo_enum_rows_to_matrix(combo::bracelets(n, k));
}

Result<Matrix<double>> eval_combo_lyndon_words(int n, int k) {
    if (n < 0) {
        return std::unexpected(
            DomainError{"combo_lyndon_words", "expected non-negative integer n"});
    }
    if (k <= 0) {
        return std::unexpected(
            DomainError{"combo_lyndon_words", "expected positive integer k"});
    }
    return combo_enum_rows_to_matrix(combo::lyndon_words(n, k));
}

Result<Matrix<double>> eval_combo_de_bruijn_sequence(int k, int n) {
    if (k <= 0) {
        return std::unexpected(
            DomainError{"combo_de_bruijn_sequence", "expected positive integer k"});
    }
    if (n <= 0) {
        return std::unexpected(
            DomainError{"combo_de_bruijn_sequence", "expected positive integer n"});
    }
    return int_vector_to_column(combo::de_bruijn_sequence(k, n));
}

Result<Matrix<double>> eval_combo_motzkin_paths(int n) {
    if (n < 0) {
        return std::unexpected(
            DomainError{"combo_motzkin_paths", "expected non-negative integer n"});
    }
    const auto paths = combo::motzkin_paths(n);
    Matrix<double> out(paths.size(), static_cast<size_t>(n));
    for (size_t r = 0; r < paths.size(); ++r) {
        for (size_t c = 0; c < paths[r].size(); ++c) {
            const char ch = paths[r][c];
            out(r, c) = ch == 'U' ? 1.0 : (ch == 'D' ? -1.0 : 0.0);
        }
    }
    return out;
}

Result<Matrix<double>> eval_combo_set_partitions(int n) {
    if (n < 0) {
        return std::unexpected(
            DomainError{"combo_set_partitions", "expected non-negative integer n"});
    }
    const auto parts = combo::set_partitions(n);
    Matrix<double> out(parts.size(), static_cast<size_t>(n));
    for (size_t r = 0; r < parts.size(); ++r) {
        std::vector<int> labels(n, 0);
        for (size_t b = 0; b < parts[r].size(); ++b) {
            for (int elem : parts[r][b]) {
                labels[elem] = static_cast<int>(b);
            }
        }
        for (int c = 0; c < n; ++c) {
            out(r, static_cast<size_t>(c)) = static_cast<double>(labels[c]);
        }
    }
    return out;
}

Result<Matrix<double>> eval_combo_restricted_partitions(int n, int k) {
    if (n < 0) {
        return std::unexpected(
            DomainError{"combo_restricted_partitions", "expected non-negative integer n"});
    }
    if (k < 0) {
        return std::unexpected(
            DomainError{"combo_restricted_partitions", "expected non-negative integer k"});
    }
    return combo_enum_rows_to_matrix(combo::restricted_partitions(n, k));
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

diffgeo::CurveFn circular_helix_curve(double a, double b) {
    return [a, b](double t) -> std::array<double, 3> {
        return {a * std::cos(t), a * std::sin(t), b * t};
    };
}

Result<double> eval_diffgeo_helix_torsion(double t, double a, double b) {
    if (a <= 0.0) {
        return std::unexpected(DomainError{"diffgeo_helix_torsion", "expected a > 0"});
    }
    return diffgeo::torsion(circular_helix_curve(a, b), t);
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

Result<double> eval_cplx_cauchy_integral(double z0re, double z0im) {
    constexpr int N = 120;
    std::vector<cplx::C> contour;
    contour.reserve(static_cast<size_t>(N) + 1);
    for (int k = 0; k <= N; ++k) {
        const double theta = 2.0 * M_PI * static_cast<double>(k) / static_cast<double>(N);
        contour.push_back(cplx::C(std::cos(theta), std::sin(theta)));
    }
    const auto f = [](const cplx::C& z) -> cplx::C { return z * z + cplx::C(1.0); };
    const cplx::C val = cplx::cauchy_integral(f, cplx::C(z0re, z0im), contour, 80);
    return val.real();
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

Result<Matrix<double>> eval_run_length_encode_vec(const Matrix<double>& m) {
  return bytes_to_matrix_col(compress::run_length_encode(matrix_to_bytes(m)));
}

Result<Matrix<double>> eval_run_length_decode_vec(const Matrix<double>& m) {
    constexpr const char* fn = "run_length_decode_vec";
    if (m.cols() != 1) {
        return std::unexpected(
            DomainError{fn, "expected Nx1 encoded byte vector"});
    }
    compress::Bytes encoded;
    encoded.reserve(m.rows());
    for (size_t i = 0; i < m.rows(); ++i) {
        const double v = m(i, 0);
        if (v < 0.0 || v > 255.0 || std::floor(v) != v) {
            return std::unexpected(
                DomainError{fn, "encoded values must be uint8 in [0,255]"});
        }
        encoded.push_back(static_cast<uint8_t>(v));
    }
    return bytes_to_matrix_col(compress::run_length_decode(encoded));
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

Matrix<double> simplicial_complex_to_matrix(const topo::SimplicialComplex& sc) {
    const auto& simplices = sc.all_simplices();
    Matrix<double> out(simplices.size(), 3);
    for (size_t r = 0; r < simplices.size(); ++r) {
        for (size_t c = 0; c < 3; ++c) {
            out(r, c) = (c < simplices[r].size()) ? static_cast<double>(simplices[r][c]) : -1.0;
        }
    }
    return out;
}

Result<topo::SimplicialComplex> matrix_to_simplicial_complex(const Matrix<double>& m,
                                                             const char* fn) {
    if (m.rows() == 0 || m.cols() < 1) {
        return std::unexpected(DomainError{fn, "expected non-empty simplex matrix"});
    }
    topo::SimplicialComplex sc;
    for (size_t r = 0; r < m.rows(); ++r) {
        topo::Simplex simplex;
        for (size_t c = 0; c < m.cols(); ++c) {
            const double v = m(r, c);
            if (v < 0.0) {
                break;
            }
            const int i = static_cast<int>(v);
            if (i < 0 || v != static_cast<double>(i)) {
                return std::unexpected(DomainError{fn, "expected non-negative integer vertex indices"});
            }
            simplex.push_back(i);
        }
        if (!simplex.empty()) {
            sc.add_simplex(simplex);
        }
    }
    return sc;
}

Result<Matrix<double>> eval_topo_cech_complex(const Matrix<double>& dist_m, double epsilon,
                                               int max_dim) {
    auto nested = matrix_to_square_nested(dist_m, "topo_cech_complex");
    if (!nested) {
        return std::unexpected(nested.error());
    }
    if (max_dim < 0) {
        return std::unexpected(
            DomainError{"topo_cech_complex", "expected non-negative integer max_dim"});
    }
    return simplicial_complex_to_matrix(topo::cech_complex(*nested, epsilon, max_dim));
}

Result<Matrix<double>> eval_topo_vietoris_rips(const Matrix<double>& dist_m, double r,
                                              int max_dim) {
    auto nested = matrix_to_square_nested(dist_m, "topo_vietoris_rips");
    if (!nested) {
        return std::unexpected(nested.error());
    }
    if (max_dim < 0) {
        return std::unexpected(
            DomainError{"topo_vietoris_rips", "expected non-negative integer max_dim"});
    }
    return simplicial_complex_to_matrix(topo::vietoris_rips(*nested, r, max_dim));
}

Result<Matrix<double>> eval_topo_simplicial_betti(const Matrix<double>& sc_m) {
    constexpr const char* fn = "topo_simplicial_betti";
    auto sc = matrix_to_simplicial_complex(sc_m, fn);
    if (!sc) {
        return std::unexpected(sc.error());
    }
    const auto betti = sc->betti_numbers();
    std::vector<double> values;
    values.reserve(betti.size());
    for (int b : betti) {
        values.push_back(static_cast<double>(b));
    }
    return vector_to_column(values);
}

Result<double> eval_topo_simplicial_euler(const Matrix<double>& sc_m) {
    constexpr const char* fn = "topo_simplicial_euler";
    auto sc = matrix_to_simplicial_complex(sc_m, fn);
    if (!sc) {
        return std::unexpected(sc.error());
    }
    return static_cast<double>(sc->euler_characteristic());
}

Result<Matrix<double>> eval_topo_simplicial_counts(const Matrix<double>& sc_m) {
    constexpr const char* fn = "topo_simplicial_counts";
    auto sc = matrix_to_simplicial_complex(sc_m, fn);
    if (!sc) {
        return std::unexpected(sc.error());
    }
    const auto counts = sc->simplex_counts();
    std::vector<double> values;
    values.reserve(counts.size());
    for (int c : counts) {
        values.push_back(static_cast<double>(c));
    }
    return vector_to_column(values);
}

Result<double> eval_topo_simplicial_dimension(const Matrix<double>& sc_m) {
    constexpr const char* fn = "topo_simplicial_dimension";
    auto sc = matrix_to_simplicial_complex(sc_m, fn);
    if (!sc) {
        return std::unexpected(sc.error());
    }
    return static_cast<double>(sc->dimension());
}

Result<std::vector<std::vector<double>>> matrix_to_topo_points2d(const Matrix<double>& m,
                                                                 const char* fn) {
    auto pts = matrix_to_points2d(m, fn);
    if (!pts) {
        return std::unexpected(pts.error());
    }
    if (pts->empty()) {
        return std::unexpected(DomainError{fn, "expected non-empty Nx2 point matrix"});
    }
    std::vector<std::vector<double>> out;
    out.reserve(pts->size());
    for (const auto& p : *pts) {
        out.push_back({p.x, p.y});
    }
    return out;
}

Result<std::vector<int>> matrix_to_index_column(const Matrix<double>& m, const char* fn) {
    auto vec = matrix_to_coeff_vector(m, fn);
    if (!vec) {
        return std::unexpected(vec.error());
    }
    if (vec->empty()) {
        return std::unexpected(DomainError{fn, "expected non-empty index column vector"});
    }
    std::vector<int> out;
    out.reserve(vec->size());
    for (const double v : *vec) {
        const int i = static_cast<int>(v);
        if (i < 0 || v != i) {
            return std::unexpected(DomainError{fn, "expected non-negative integer indices"});
        }
        out.push_back(i);
    }
    return out;
}

Result<Matrix<double>> eval_topo_alpha_complex(const Matrix<double>& P_m, double alpha,
                                               int max_dim) {
    auto pts = matrix_to_topo_points2d(P_m, "topo_alpha_complex");
    if (!pts) {
        return std::unexpected(pts.error());
    }
    if (max_dim < 0) {
        return std::unexpected(
            DomainError{"topo_alpha_complex", "expected non-negative integer max_dim"});
    }
    return simplicial_complex_to_matrix(topo::alpha_complex(*pts, alpha, max_dim));
}

Result<Matrix<double>> eval_topo_select_landmarks(const Matrix<double>& P_m, int n_landmarks,
                                                  int seed_index) {
    auto pts = matrix_to_nested(P_m, "topo_select_landmarks");
    if (!pts) {
        return std::unexpected(pts.error());
    }
    if (n_landmarks < 1) {
        return std::unexpected(
            DomainError{"topo_select_landmarks", "expected positive integer n"});
    }
    return int_vector_to_column(topo::select_landmarks_maxmin(*pts, n_landmarks, seed_index));
}

Result<Matrix<double>> eval_topo_witness_complex(const Matrix<double>& P_m,
                                                 const Matrix<double>& landmarks_m,
                                                 double max_epsilon, int max_dim) {
    auto pts = matrix_to_nested(P_m, "topo_witness_complex");
    if (!pts) {
        return std::unexpected(pts.error());
    }
    auto landmarks = matrix_to_index_column(landmarks_m, "topo_witness_complex");
    if (!landmarks) {
        return std::unexpected(landmarks.error());
    }
    if (max_dim < 0) {
        return std::unexpected(
            DomainError{"topo_witness_complex", "expected non-negative integer max_dim"});
    }
    return simplicial_complex_to_matrix(
        topo::witness_complex(*pts, *landmarks, max_epsilon, max_dim));
}

Result<Matrix<double>> eval_topo_persistence_landscape(const Matrix<double>& dgm_m, int n_layers,
                                                       int n_samples, double t_min, double t_max) {
    auto diagram = matrix_to_persistence_diagram(dgm_m, "topo_persistence_landscape");
    if (!diagram) {
        return std::unexpected(diagram.error());
    }
    if (n_layers < 1 || n_samples < 2) {
        return std::unexpected(DomainError{
            "topo_persistence_landscape", "expected n_layers >= 1 and n_samples >= 2"});
    }
    const auto layers =
        topo::persistence_landscape(*diagram, n_layers, n_samples, t_min, t_max);
    if (layers.empty()) {
        return Matrix<double>(0, 0);
    }
    Matrix<double> out(layers.size(), layers[0].size());
    for (size_t i = 0; i < layers.size(); ++i) {
        for (size_t j = 0; j < layers[i].size(); ++j) {
            out(i, j) = layers[i][j];
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

Result<double> eval_diffgeo_sphere_gauss_bonnet(int n) {
    if (n < 1) {
        return std::unexpected(
            DomainError{"diffgeo_sphere_gauss_bonnet", "expected positive integer n"});
    }
    const auto sphere = unit_sphere_surface();
    constexpr double u0 = -M_PI / 2.0;
    constexpr double u1 = M_PI / 2.0;
    constexpr double v0 = 0.0;
    constexpr double v1 = 2.0 * M_PI;
    return diffgeo::gauss_bonnet_integral(sphere, u0, u1, v0, v1, n, n);
}

Result<double> eval_diffgeo_sphere_gauss_bonnet_residual(int n) {
    if (n < 1) {
        return std::unexpected(DomainError{
            "diffgeo_sphere_gauss_bonnet_residual", "expected positive integer n"});
    }
    const auto sphere = unit_sphere_surface();
    constexpr double u0 = -M_PI / 2.0;
    constexpr double u1 = M_PI / 2.0;
    constexpr double v0 = 0.0;
    constexpr double v1 = 2.0 * M_PI;
    return diffgeo::gauss_bonnet_residual(sphere, u0, u1, v0, v1, n, n, 2);
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
           callee == "diffgeo_sphere_gauss_bonnet" ||
           callee == "diffgeo_sphere_gauss_bonnet_residual" ||
           callee == "topo_euler_tetrahedron" || callee == "topo_euler_sphere_surface" ||
           callee == "cplx_contour_integral_oneoverz_im" || callee == "cplx_line_integral_one" ||
           callee == "mpi_rank" || callee == "mpi_size" ||
           callee == "cuda_nccl_available" || callee == "cuda_nccl_comm_size" ||
           callee == "cuda_nccl_device_count";
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
    if (fn == "diffgeo_sphere_gauss_bonnet") {
        return eval_diffgeo_sphere_gauss_bonnet(200);
    }
    if (fn == "diffgeo_sphere_gauss_bonnet_residual") {
        return eval_diffgeo_sphere_gauss_bonnet_residual(200);
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
    if (fn == "mpi_rank") {
        return static_cast<double>(ms::distributed::rank(repl_mpi_context()));
    }
    if (fn == "mpi_size") {
        return static_cast<double>(ms::distributed::size(repl_mpi_context()));
    }
    if (fn == "cuda_nccl_available") {
        return ms::cuda::nccl_available() ? 1.0 : 0.0;
    }
    if (fn == "cuda_nccl_comm_size") {
        return static_cast<double>(ms::cuda::nccl_comm_size());
    }
    if (fn == "cuda_nccl_device_count") {
        return static_cast<double>(ms::cuda::nccl_device_count());
    }
    return std::unexpected(DomainError{"eval", "unknown nullary scalar function: " + fn});
}

bool is_nullary_matrix_callee(const std::string& callee) {
    return callee == "quantum_pauli_x" || callee == "quantum_pauli_y" ||
           callee == "quantum_pauli_z" || callee == "quantum_pauli_plus" ||
           callee == "quantum_pauli_minus" || callee == "quantum_cnot_gate" ||
           callee == "quantum_swap_gate" || callee == "quantum_toffoli_gate" ||
           callee == "quantum_identity" || callee == "quantum_hadamard_gate" ||
           callee == "quantum_bell_states" || callee == "izaac_vrf_keygen";
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
    if (fn == "quantum_bell_states") {
        return eval_quantum_bell_states();
    }
    if (fn == "izaac_vrf_keygen") {
        return eval_izaac_vrf_keygen();
    }
    return std::unexpected(DomainError{"eval", "unknown nullary matrix function: " + fn});
}

bool is_unary_scalar_matrix_callee(const std::string& callee) {
    return callee == "quantum_rotation_z" || callee == "quantum_rotation_x" ||
           callee == "quantum_rotation_y" || callee == "quantum_phase_gate" ||
           callee == "quantum_qft_gate" || callee == "quantum_identity_n" ||
           callee == "quantum_ghz_state" || callee == "quantum_w_state" ||
           callee == "quantum_bell_state" ||
           callee == "numthy_divisors_vec" || callee == "numthy_divisors" ||
           callee == "numthy_factor_vec" || callee == "numthy_factor" ||
           callee == "combo_derangements" || callee == "combo_all_permutations" ||
           callee == "combo_all_subsets" || callee == "combo_all_compositions" ||
           callee == "combo_all_partitions" || callee == "combo_gray_code" ||
           callee == "combo_dyck_paths" || callee == "signal_hamming" ||
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
    if (fn == "combo_gray_code") {
        const int n = static_cast<int>(arg);
        if (n < 0 || arg != n) {
            return std::unexpected(
                DomainError{"combo_gray_code", "expected non-negative integer n"});
        }
        return eval_combo_gray_code(n);
    }
    if (fn == "combo_dyck_paths") {
        const int n = static_cast<int>(arg);
        if (n < 0 || arg != n) {
            return std::unexpected(
                DomainError{"combo_dyck_paths", "expected non-negative integer n"});
        }
        return eval_combo_dyck_paths(n);
    }
    if (fn == "numthy_divisors_vec" || fn == "numthy_divisors") {
        const int n = static_cast<int>(arg);
        if (n < 1 || arg != n) {
            return std::unexpected(
                DomainError{fn, "expected positive integer n"});
        }
        return eval_numthy_divisors_vec(n);
    }
    if (fn == "numthy_factor_vec" || fn == "numthy_factor") {
        const int n = static_cast<int>(arg);
        if (n < 2 || arg != n) {
            return std::unexpected(
                DomainError{fn, "expected integer n >= 2"});
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

struct ScalarTripleMatrixCallAssign {
    std::string target;
    std::string callee;
    std::string arg_a;
    std::string arg_b;
    std::string arg_c;
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

struct AxiomExprMatrixCallAssign {
    std::string target;
    std::string callee;
    std::string expr_arg;
    std::string inputs_arg;
};

struct AxiomExprFitnessCallAssign {
    std::string target;
    std::string callee;
    std::string expr_arg;
    std::string inputs_arg;
    std::string targets_arg;
};

bool is_identifier(const std::string& text);
std::optional<std::vector<std::string>> split_call_args(const std::string& cmd);

bool try_parse_axiom_expr_matrix_call_assignment(const std::string& line,
                                                 AxiomExprMatrixCallAssign& assign) {
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
    if (assign.callee != "axiom_evaluate") {
        return false;
    }
    const auto call_args = split_call_args(rhs);
    if (!call_args || call_args->size() != 2) {
        return false;
    }
    assign.expr_arg = trim_copy(call_args->at(0));
    assign.inputs_arg = trim_copy(call_args->at(1));
    return !assign.expr_arg.empty() && !assign.inputs_arg.empty();
}

bool try_parse_axiom_expr_fitness_call_assignment(const std::string& line,
                                                  AxiomExprFitnessCallAssign& assign) {
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
    if (assign.callee != "axiom_mse_fitness" && assign.callee != "axiom_rmse_fitness" &&
        assign.callee != "axiom_gria_fitness") {
        return false;
    }
    const auto call_args = split_call_args(rhs);
    if (!call_args) {
        return false;
    }
    if (assign.callee == "axiom_gria_fitness") {
        if (call_args->size() != 2) {
            return false;
        }
        assign.expr_arg = trim_copy(call_args->at(0));
        assign.inputs_arg = trim_copy(call_args->at(1));
        assign.targets_arg.clear();
        return !assign.expr_arg.empty() && !assign.inputs_arg.empty();
    }
    if (call_args->size() != 3) {
        return false;
    }
    assign.expr_arg = trim_copy(call_args->at(0));
    assign.inputs_arg = trim_copy(call_args->at(1));
    assign.targets_arg = trim_copy(call_args->at(2));
    return !assign.expr_arg.empty() && !assign.inputs_arg.empty() && !assign.targets_arg.empty();
}

bool is_scalar_matrix_mixed_call_callee(const std::string& callee) {
    return callee == "finance_npv" || callee == "info_renyi_entropy" ||
           callee == "info_tsallis_entropy" || callee == "combo_multinomial";
}

bool is_two_scalar_matrix_mixed_call_callee(const std::string& callee) {
    return callee == "cplx_blaschke_product";
}

bool is_matrix_scalar_mixed_call_callee(const std::string& callee) {
    return callee == "finance_historical_var" || callee == "finance_historical_cvar" ||
           callee == "geo_bezier_eval_x" || callee == "geo_bezier_eval_y" ||
           callee == "bwt_decode_vec" || callee == "combo_rank_combination" ||
           callee == "combo_next_comb" || callee == "combo_prev_comb" ||
           callee == "quantum_time_evolution" || callee == "signal_moving_average" ||
           callee == "signal_median_filter" ||
           callee == "signal_upsample" || callee == "signal_downsample" ||
           callee == "signal_decimate" || callee == "signal_interpolate" ||
           callee == "graph_bfs" || callee == "graph_dfs" ||
           callee == "graph_bipartite_match" || callee == "graph_k_core_subgraph" ||
           callee == "stats_percentile" ||
           callee == "stats_ttest" || callee == "stats_trimmed_mean" ||
           callee == "stats_vif" || callee == "stats_variance_inflation_factor" ||
           callee == "stats_acf" ||
           callee == "poly_eval" || callee == "poly_cheb_eval" ||
           callee == "fft_irfft" || callee == "poly_integ" ||
           callee == "poly_shift" || callee == "poly_scale" || callee == "poly_pow" ||
           callee == "poly_cheb_expand" || callee == "gria_ca_step";
}

bool is_matrix_dual_matrix_call_callee(const std::string& callee) {
    return callee == "control_lyap" || callee == "control_dlyap" || callee == "huffman_decode_vec" ||
           callee == "arithmetic_decode_vec" || callee == "ans_decode_vec" ||
           callee == "quantum_op_apply" || callee == "topo_persistence_diagram" ||
           callee == "control_margins" || callee == "control_poles" ||
           callee == "control_zeros" || callee == "control_step_info" ||
           callee == "control_nyquist" || callee == "control_ctrb" ||
           callee == "control_obsv" || callee == "control_ctrb_gram" ||
           callee == "control_obsv_gram" ||            callee == "quantum_commutator" ||
           callee == "quantum_anticommutator" ||
           callee == "quantum_ket_tensor_product" || callee == "quantum_outer" ||
           callee == "poly_add" || callee == "poly_lagrange" ||
           callee == "poly_interp_newton" || callee == "quantum_tensor_product" ||
           callee == "ml_mat_mul" ||
           callee == "ml_linear_fit" || callee == "ml_linear_predict" ||
           callee == "ml_ridge_predict" || callee == "ml_logistic_fit" ||
           callee == "ml_logistic_predict" ||
           callee == "ml_lasso_predict" || callee == "ml_elastic_net_predict" ||
           callee == "ml_knn_predict" || callee == "ml_naive_bayes_predict" ||
           callee == "ml_lda_predict" || callee == "ml_lda_transform" ||
           callee == "ml_qda_predict" || callee == "ml_svm_predict" ||
           callee == "ml_decision_tree_predict" || callee == "ml_random_forest_predict" ||
           callee == "ml_adaboost_predict" || callee == "ml_gradient_boosting_predict" ||
           callee == "poly_mul" || callee == "poly_sub" || callee == "poly_compose" ||
           callee == "signal_convolve" || callee == "signal_correlate" ||
           callee == "signal_sosfilt" || callee == "signal_conv2" ||
           callee == "geo_poly_union" || callee == "geo_poly_intersect" ||
           callee == "geo_poly_diff" || callee == "geo_minkowski_sum" ||
           callee == "geo_clip_polygon";
}

bool is_matrix_triple_matrix_call_callee(const std::string& callee) {
    return callee == "control_place" || callee == "signal_filtfilt" || callee == "signal_filter" ||
           callee == "solve_sylvester";
}

bool is_matrix_quad_matrix_call_callee(const std::string& callee) {
    return callee == "control_lqr" || callee == "control_lqe" || callee == "control_riccati" ||
           callee == "control_dare";
}

bool is_scalar_dual_matrix_call_callee(const std::string& callee) {
    return callee == "ml_accuracy" || callee == "ml_rmse" || callee == "ml_mse" ||
           callee == "ml_r2" || callee == "ml_f1" || callee == "ml_precision" ||
           callee == "ml_recall" || callee == "ml_mae" || callee == "ml_huber" ||
           callee == "ml_hinge" || callee == "ml_binary_crossentropy" ||
           callee == "ml_categorical_crossentropy" || callee == "ml_vec_dot" ||
           callee == "ml_kmeans_inertia" || callee == "ml_roc_auc" ||
           callee == "ml_average_precision" ||
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
           callee == "finance_information_ratio" ||
           callee == "tensorops_inner" ||
           callee == "control_phase_margin" || callee == "control_gain_margin" ||
           callee == "poly_resultant" ||
           callee == "stats_correlation" || callee == "stats_spearman" ||
           callee == "stats_kendall" || callee == "stats_weighted_mean" ||
           callee == "stats_weighted_variance" ||
           callee == "stats_two_sample_ttest" ||
           callee == "stats_chi2_gof" || callee == "graph_is_isomorphic" ||
           callee == "graph_modularity" || callee == "gria_hamming_distance" ||
           callee == "cfd_integrated_mass_3d" || callee == "cfd_integrated_mass_1d" ||
           callee == "cfd_integrated_mass_2d";
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

bool is_scalar_triple_matrix_call_callee(const std::string& callee) {
    return callee == "stats_partial_correlation" || callee == "stats_weighted_correlation" ||
           callee == "izaac_vrf_verify";
}

bool try_parse_scalar_triple_matrix_call_assignment(const std::string& line,
                                                    ScalarTripleMatrixCallAssign& assign) {
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
    if (!is_scalar_triple_matrix_call_callee(assign.callee)) {
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

struct MatrixTwoScalarMixedCallAssign {
    std::string target;
    std::string callee;
    std::string arg_matrix;
    std::string arg_scalar_a;
    std::string arg_scalar_b;
};

bool is_matrix_two_scalar_mixed_call_callee(const std::string& callee) {
    return callee == "poly_root_count" || callee == "quantum_wigner" ||
           callee == "quantum_husimi" || callee == "cfd_integrated_mass_2d" ||
           callee == "izaac_exponential_mechanism";
}

bool try_parse_matrix_two_scalar_mixed_call_assignment(
    const std::string& line, MatrixTwoScalarMixedCallAssign& assign) {
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
    if (!is_matrix_two_scalar_mixed_call_callee(assign.callee)) {
        return false;
    }
    const auto call_args = split_call_args(rhs);
    if (!call_args || call_args->size() != 3) {
        return false;
    }
    assign.arg_matrix = trim_copy(call_args->at(0));
    assign.arg_scalar_a = trim_copy(call_args->at(1));
    assign.arg_scalar_b = trim_copy(call_args->at(2));
    return !assign.arg_matrix.empty() && !assign.arg_scalar_a.empty() &&
           !assign.arg_scalar_b.empty();
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
    if (callee == "ml_roc_auc") {
        return ml::roc_auc(y_pred, y_true);
    }
    if (callee == "ml_average_precision") {
        return ml::average_precision(y_pred, y_true);
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

Result<std::string> eval_sym_expand_string(const std::string& expr_arg) {
    auto expr = parse_sym_quoted_expr(expr_arg, "sym_expand");
    if (!expr) {
        return std::unexpected(expr.error());
    }
    const auto result = sym_expand(std::move(*expr));
    return sym_to_string(result) + "\n";
}

using SymTransform3 = SymExpr (*)(const SymExpr&, const std::string&, const std::string&);

Result<std::string> eval_sym_transform_strings(const std::string& expr_arg, const std::string& var_a_arg,
                                               const std::string& var_b_arg, const char* fn,
                                               SymTransform3 transform) {
    auto expr = parse_sym_quoted_expr(expr_arg, fn);
    if (!expr) {
        return std::unexpected(expr.error());
    }
    std::string var_a;
    std::string var_b;
    if (!parse_quoted_string(var_a_arg, var_a) || var_a.empty() ||
        !parse_quoted_string(var_b_arg, var_b) || var_b.empty()) {
        return std::unexpected(
            DomainError{fn, std::string("expected ") + fn + "(\"expr\", \"var1\", \"var2\")"});
    }
    const auto result = sym_simplify(transform(*expr, var_a, var_b));
    return sym_to_string(result) + "\n";
}

Result<std::string> eval_sym_collect_strings(const std::string& expr_arg, const std::string& var_arg) {
    auto expr = parse_sym_quoted_expr(expr_arg, "sym_collect");
    if (!expr) {
        return std::unexpected(expr.error());
    }
    std::string var_text;
    if (!parse_quoted_string(var_arg, var_text) || var_text.empty()) {
        return std::unexpected(DomainError{"sym_collect", "expected sym_collect(\"expr\", \"var\")"});
    }
    const auto result = sym_collect(*expr, var_text);
    return sym_to_string(result) + "\n";
}

Result<std::string> eval_sym_substitute_strings(const std::string& expr_arg, const std::string& var_arg,
                                                const std::string& repl_arg) {
    auto expr = parse_sym_quoted_expr(expr_arg, "sym_substitute");
    if (!expr) {
        return std::unexpected(expr.error());
    }
    std::string var_text;
    if (!parse_quoted_string(var_arg, var_text) || var_text.empty()) {
        return std::unexpected(DomainError{
            "sym_substitute", "expected sym_substitute(\"expr\", \"var\", \"replacement\")"});
    }
    auto repl = parse_sym_quoted_expr(repl_arg, "sym_substitute");
    if (!repl) {
        return std::unexpected(repl.error());
    }
    const auto result = sym_substitute(*expr, var_text, *repl);
    return sym_to_string(result) + "\n";
}

Result<std::string> eval_sym_limit_strings(const std::string& expr_arg, const std::string& var_arg,
                                           const std::string& point_arg) {
    auto expr = parse_sym_quoted_expr(expr_arg, "sym_limit");
    if (!expr) {
        return std::unexpected(expr.error());
    }
    std::string var_text;
    if (!parse_quoted_string(var_arg, var_text) || var_text.empty()) {
        return std::unexpected(DomainError{"sym_limit", "expected sym_limit(\"expr\", \"var\", point)"});
    }
    double point = 0.0;
    if (!parse_number(trim_copy(point_arg), point)) {
        return std::unexpected(DomainError{"sym_limit", "expected numeric limit point"});
    }
    return std::to_string(sym_limit(*expr, var_text, point)) + "\n";
}

Result<std::string> eval_sym_series_strings(const std::string& expr_arg, const std::string& var_arg,
                                            const std::string& point_arg, const std::string& order_arg) {
    auto expr = parse_sym_quoted_expr(expr_arg, "sym_series");
    if (!expr) {
        return std::unexpected(expr.error());
    }
    std::string var_text;
    if (!parse_quoted_string(var_arg, var_text) || var_text.empty()) {
        return std::unexpected(
            DomainError{"sym_series", "expected sym_series(\"expr\", \"var\", point, order)"});
    }
    double point = 0.0;
    double order_d = 0.0;
    if (!parse_number(trim_copy(point_arg), point) || !parse_number(trim_copy(order_arg), order_d)) {
        return std::unexpected(
            DomainError{"sym_series", "expected numeric point and integer order"});
    }
    const int order = static_cast<int>(order_d);
    if (order < 0 || order_d != order) {
        return std::unexpected(DomainError{"sym_series", "expected non-negative integer order"});
    }
    const auto result = sym_series(*expr, var_text, point, order);
    return sym_to_string(result) + "\n";
}

Result<std::vector<std::string>> parse_sym_semicolon_identifiers(const std::string& arg,
                                                                 const char* fn) {
    std::string text;
    if (!parse_quoted_string(arg, text)) {
        return std::unexpected(
            DomainError{fn, "expected quoted semicolon-separated identifier string"});
    }
    std::vector<std::string> ids;
    std::stringstream ss(text);
    std::string part;
    while (std::getline(ss, part, ';')) {
        part = trim_copy(part);
        if (part.empty()) {
            continue;
        }
        if (!is_identifier(part)) {
            return std::unexpected(DomainError{fn, "expected identifier in semicolon list"});
        }
        ids.push_back(part);
    }
    if (ids.empty()) {
        return std::unexpected(DomainError{fn, "expected at least one identifier"});
    }
    return ids;
}

Result<std::vector<SymExpr>> parse_sym_semicolon_formulas(const std::string& formula_arg,
                                                          const char* fn);

Result<std::string> eval_sym_solve_linear_strings(const std::string& eqs_arg,
                                                  const std::string& vars_arg) {
    auto eqs = parse_sym_semicolon_formulas(eqs_arg, "sym_solve_linear");
    if (!eqs) {
        return std::unexpected(eqs.error());
    }
    auto vars = parse_sym_semicolon_identifiers(vars_arg, "sym_solve_linear");
    if (!vars) {
        return std::unexpected(vars.error());
    }
    const auto result = sym_solve_linear(*eqs, *vars);
    if (!result) {
        return std::unexpected(DomainError{"sym_solve_linear", result.error().message});
    }
    std::ostringstream oss;
    for (const auto& [name, expr] : *result) {
        oss << name << " = " << sym_to_string(expr) << "\n";
    }
    return oss.str();
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

OdeResult ode_trapezoidal_wrapped(OdeFunc f, double t0, double y0, double t_end, size_t steps) {
    return ode_trapezoidal(f, t0, t_end, y0, static_cast<int>(steps));
}

OdeResult ode_rosenbrock23_wrapped(OdeFunc f, double t0, double y0, double t_end, size_t steps) {
    return ode_rosenbrock23(f, t0, y0, t_end, static_cast<int>(steps));
}

Result<Matrix<double>> eval_ode_fixed_step_matrix(const std::string& fn,
                                                  const std::string& formula_arg,
                                                  const std::string& t0_arg,
                                                  const std::string& y0_arg,
                                                  const std::string& t_end_arg,
                                                  const std::string& steps_arg,
                                                  OdeResult (*solver)(OdeFunc, double, double,
                                                                      double, size_t)) {
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
    const OdeResult result =
        solver(f, t0, y0, t_end, static_cast<size_t>(steps_i));
    if (result.t.size() != result.y.size()) {
        return std::unexpected(DomainError{"ode", "internal trajectory size mismatch"});
    }
    Matrix<double> out(result.t.size(), 2);
    for (size_t i = 0; i < result.t.size(); ++i) {
        out(i, 0) = result.t[i];
        out(i, 1) = result.y[i];
    }
    return out;
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

Result<std::string> eval_ode_adaptive_call(
    const char* fn, const std::string& formula_arg, const std::string& t0_arg,
    const std::string& y0_arg, const std::string& t_end_arg, const std::string& rtol_arg,
    const std::string& atol_arg,
    OdeResult (*solver)(OdeFunc, double, double, double, double, double)) {
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
            fn, std::string("expected ") + fn + "(\"formula\", t0, y0, t_end, rtol, atol)"});
    }
    SymExpr parsed = std::move(*expr);
    auto expr_ptr = std::make_shared<SymExpr>(std::move(parsed));
    OdeFunc f = [expr_ptr](double t, double y) {
        return sym_eval(*expr_ptr, {{"t", t}, {"y", y}});
    };
    return format_ode_trajectory(solver(f, t0, y0, t_end, rtol, atol));
}

Result<std::string> eval_ode_trapezoidal_call(const std::string& formula_arg,
                                              const std::string& t0_arg,
                                              const std::string& y0_arg,
                                              const std::string& t_end_arg,
                                              const std::string& steps_arg) {
    constexpr const char* fn = "ode_trapezoidal";
    auto expr = parse_sym_quoted_expr(formula_arg, fn);
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
            fn, "expected ode_trapezoidal(\"formula\", t0, y0, t_end, steps)"});
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
    return format_ode_trajectory(ode_trapezoidal(f, t0, t_end, y0, steps_i));
}

Result<std::string> eval_ode_rosenbrock23_call(const std::string& formula_arg,
                                               const std::string& t0_arg,
                                               const std::string& y0_arg,
                                               const std::string& t_end_arg,
                                               const std::string& steps_arg) {
    constexpr const char* fn = "ode_rosenbrock23";
    auto expr = parse_sym_quoted_expr(formula_arg, fn);
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
            fn, "expected ode_rosenbrock23(\"formula\", t0, y0, t_end, steps)"});
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
    return format_ode_trajectory(ode_rosenbrock23(f, t0, y0, t_end, steps_i));
}

Result<std::string> eval_ode_exponential_euler_call(const std::string& formula_arg,
                                                  const std::string& lambda_arg,
                                                  const std::string& t0_arg,
                                                  const std::string& y0_arg,
                                                  const std::string& t_end_arg,
                                                  const std::string& steps_arg) {
    constexpr const char* fn = "ode_exponential_euler";
    auto expr = parse_sym_quoted_expr(formula_arg, fn);
    if (!expr) {
        return std::unexpected(expr.error());
    }
    double lambda = 0.0;
    double t0 = 0.0;
    double y0 = 0.0;
    double t_end = 0.0;
    double steps_d = 0.0;
    if (!parse_number(trim_copy(lambda_arg), lambda) || !parse_number(trim_copy(t0_arg), t0) ||
        !parse_number(trim_copy(y0_arg), y0) || !parse_number(trim_copy(t_end_arg), t_end) ||
        !parse_number(trim_copy(steps_arg), steps_d)) {
        return std::unexpected(DomainError{
            fn,
            "expected ode_exponential_euler(\"g\", lambda, t0, y0, t_end, steps) with g the "
            "nonlinear remainder in dy/dt = lambda*y + g(t,y)"});
    }
    const int steps_i = static_cast<int>(steps_d);
    if (steps_i < 0 || steps_d != steps_i) {
        return std::unexpected(DomainError{fn, "expected non-negative integer steps"});
    }
    SymExpr parsed = std::move(*expr);
    auto expr_ptr = std::make_shared<SymExpr>(std::move(parsed));
    OdeFunc g = [expr_ptr](double t, double y) {
        return sym_eval(*expr_ptr, {{"t", t}, {"y", y}});
    };
    return format_ode_trajectory(
        ode_exponential_euler(g, lambda, t0, y0, t_end, steps_i));
}

Result<std::vector<double>> parse_comma_separated_numbers(const std::string& row_text,
                                                          const char* fn) {
    std::vector<double> row;
    std::stringstream col_stream(row_text);
    std::string cell;
    while (std::getline(col_stream, cell, ',')) {
        cell = trim_copy(cell);
        if (cell.empty()) {
            continue;
        }
        double value = 0.0;
        if (!parse_number(cell, value)) {
            return std::unexpected(DomainError{fn, "invalid number in vector literal: " + cell});
        }
        row.push_back(value);
    }
    return row;
}

Result<Matrix<double>> parse_bracket_matrix_literal(const std::string& text, const char* fn) {
    std::string s = trim_copy(text);
    if (s.size() < 2 || s.front() != '[' || s.back() != ']') {
        return std::unexpected(DomainError{fn, "expected [ ... ] vector literal"});
    }
    s = trim_copy(s.substr(1, s.size() - 2));
    if (s.empty()) {
        return Matrix<double>(0, 0);
    }

    std::vector<std::vector<double>> rows;
    if (!s.empty() && s.front() == '[') {
        for (size_t i = 0; i < s.size();) {
            while (i < s.size() &&
                   (s[i] == ',' || std::isspace(static_cast<unsigned char>(s[i])))) {
                ++i;
            }
            if (i >= s.size()) {
                break;
            }
            if (s[i] != '[') {
                rows.clear();
                break;
            }
            int depth = 0;
            const size_t start = i;
            for (; i < s.size(); ++i) {
                if (s[i] == '[') {
                    ++depth;
                } else if (s[i] == ']') {
                    --depth;
                    if (depth == 0) {
                        ++i;
                        break;
                    }
                }
            }
            if (depth != 0) {
                return std::unexpected(DomainError{fn, "unbalanced brackets in vector literal"});
            }
            const std::string row_text = trim_copy(s.substr(start + 1, i - start - 2));
            auto row = parse_comma_separated_numbers(row_text, fn);
            if (!row) {
                return std::unexpected(row.error());
            }
            if (row->empty()) {
                return std::unexpected(DomainError{fn, "empty row in vector literal"});
            }
            rows.push_back(std::move(*row));
        }
    }

    if (rows.empty()) {
        std::stringstream row_stream(s);
        std::string row_text;
        while (std::getline(row_stream, row_text, ';')) {
            row_text = trim_copy(row_text);
            if (row_text.empty()) {
                continue;
            }
            auto row = parse_comma_separated_numbers(row_text, fn);
            if (!row) {
                return std::unexpected(row.error());
            }
            rows.push_back(std::move(*row));
        }
    }

    if (rows.empty()) {
        return std::unexpected(DomainError{fn, "no rows found in vector literal"});
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

Result<std::vector<double>> parse_bracket_vector_literal(const std::string& text, const char* fn) {
    auto matrix = parse_bracket_matrix_literal(text, fn);
    if (!matrix) {
        return std::unexpected(matrix.error());
    }
    return matrix_to_coeff_vector(*matrix, fn);
}

std::map<std::string, double> build_optim_env(const std::vector<double>& x) {
    std::map<std::string, double> env;
    for (size_t i = 0; i < x.size(); ++i) {
        env["x" + std::to_string(i)] = x[i];
    }
    return env;
}

Result<std::string> format_optim_result(const OptimResult& result) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "x_opt =\n";
    for (double xi : result.x) {
        oss << "  [" << xi << "]\n";
    }
    oss << "f_val = " << result.f_val << "\n";
    oss << "iterations = " << result.iterations << "\n";
    oss << "converged = " << (result.converged ? 1 : 0) << "\n";
    return oss.str();
}

Result<std::string> format_scalar_optim_result(double x_opt, double f_val) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "x_opt = " << x_opt << "\n";
    oss << "f_val = " << f_val << "\n";
    return oss.str();
}

struct NdOptimInputs {
    std::shared_ptr<SymExpr> expr;
    std::vector<double> x0;
    FuncND f;
};

Result<NdOptimInputs> parse_nd_optim_inputs(const std::string& formula_arg,
                                            const std::string& x0_arg, const char* fn) {
    auto expr = parse_sym_quoted_expr(formula_arg, fn);
    if (!expr) {
        return std::unexpected(expr.error());
    }
    auto x0 = parse_bracket_vector_literal(x0_arg, fn);
    if (!x0) {
        return std::unexpected(x0.error());
    }
    if (x0->empty()) {
        return std::unexpected(DomainError{fn, "expected non-empty initial point vector x0"});
    }
    auto expr_ptr = std::make_shared<SymExpr>(std::move(*expr));
    const size_t dim = x0->size();
    FuncND f = [expr_ptr, dim](const std::vector<double>& x) {
        return sym_eval(*expr_ptr, build_optim_env(x));
    };
    return NdOptimInputs{expr_ptr, std::move(*x0), std::move(f)};
}

GradND make_finite_diff_grad(FuncND f, int n) {
    constexpr double h = 1e-7;
    return [f, n, h](const std::vector<double>& x) {
        std::vector<double> g(static_cast<size_t>(n));
        std::vector<double> xp = x;
        std::vector<double> xm = x;
        for (int i = 0; i < n; ++i) {
            xp[static_cast<size_t>(i)] += h;
            xm[static_cast<size_t>(i)] -= h;
            g[static_cast<size_t>(i)] = (f(xp) - f(xm)) / (2.0 * h);
            xp[static_cast<size_t>(i)] = x[static_cast<size_t>(i)];
            xm[static_cast<size_t>(i)] = x[static_cast<size_t>(i)];
        }
        return g;
    };
}

Result<double> parse_optional_positive_number(const std::string& text, const char* fn,
                                              const char* label, double default_value) {
    if (text.empty()) {
        return default_value;
    }
    double value = 0.0;
    if (!parse_number(trim_copy(text), value) || value <= 0.0) {
        return std::unexpected(DomainError{fn, std::string("expected positive ") + label});
    }
    return value;
}

Result<int> parse_optional_positive_int(const std::string& text, const char* fn, const char* label,
                                        int default_value) {
    if (text.empty()) {
        return default_value;
    }
    double value = 0.0;
    if (!parse_number(trim_copy(text), value) || value < 1.0 || std::floor(value) != value) {
        return std::unexpected(DomainError{fn, std::string("expected positive integer ") + label});
    }
    return static_cast<int>(value);
}

Result<std::string> eval_bfgs_call(const std::string& formula_arg, const std::string& x0_arg,
                                   const std::string& tol_arg, const std::string& max_iter_arg) {
    constexpr const char* fn = "bfgs";
    auto inputs = parse_nd_optim_inputs(formula_arg, x0_arg, fn);
    if (!inputs) {
        return std::unexpected(inputs.error());
    }
    auto tol = parse_optional_positive_number(tol_arg, fn, "tol", 1e-8);
    if (!tol) {
        return std::unexpected(tol.error());
    }
    auto max_iter = parse_optional_positive_int(max_iter_arg, fn, "max_iter", 500);
    if (!max_iter) {
        return std::unexpected(max_iter.error());
    }
    return format_optim_result(bfgs(inputs->f, inputs->x0, *tol, *max_iter));
}

Result<std::string> eval_nelder_mead_call(const std::string& formula_arg, const std::string& x0_arg,
                                          const std::string& tol_arg,
                                          const std::string& max_iter_arg) {
    constexpr const char* fn = "nelder_mead";
    auto inputs = parse_nd_optim_inputs(formula_arg, x0_arg, fn);
    if (!inputs) {
        return std::unexpected(inputs.error());
    }
    auto tol = parse_optional_positive_number(tol_arg, fn, "tol", 1e-8);
    if (!tol) {
        return std::unexpected(tol.error());
    }
    auto max_iter = parse_optional_positive_int(max_iter_arg, fn, "max_iter", 1000);
    if (!max_iter) {
        return std::unexpected(max_iter.error());
    }
    return format_optim_result(nelder_mead(inputs->f, inputs->x0, *tol, *max_iter));
}

Result<std::string> eval_lbfgs_call(const std::string& formula_arg, const std::string& x0_arg,
                                    const std::string& m_arg, const std::string& tol_arg,
                                    const std::string& max_iter_arg) {
    constexpr const char* fn = "lbfgs";
    auto inputs = parse_nd_optim_inputs(formula_arg, x0_arg, fn);
    if (!inputs) {
        return std::unexpected(inputs.error());
    }
    auto m = parse_optional_positive_int(m_arg, fn, "m", 5);
    if (!m) {
        return std::unexpected(m.error());
    }
    auto tol = parse_optional_positive_number(tol_arg, fn, "tol", 1e-8);
    if (!tol) {
        return std::unexpected(tol.error());
    }
    auto max_iter = parse_optional_positive_int(max_iter_arg, fn, "max_iter", 500);
    if (!max_iter) {
        return std::unexpected(max_iter.error());
    }
    return format_optim_result(lbfgs(inputs->f, inputs->x0, *m, *tol, *max_iter));
}

Result<std::string> eval_adam_call(const std::string& formula_arg, const std::string& x0_arg,
                                   const std::string& alpha_arg, const std::string& max_iter_arg) {
    constexpr const char* fn = "adam";
    auto inputs = parse_nd_optim_inputs(formula_arg, x0_arg, fn);
    if (!inputs) {
        return std::unexpected(inputs.error());
    }
    auto alpha = parse_optional_positive_number(alpha_arg, fn, "alpha", 0.001);
    if (!alpha) {
        return std::unexpected(alpha.error());
    }
    auto max_iter = parse_optional_positive_int(max_iter_arg, fn, "max_iter", 1000);
    if (!max_iter) {
        return std::unexpected(max_iter.error());
    }
    return format_optim_result(adam(inputs->f, inputs->x0, *alpha, 0.9, 0.999, *max_iter));
}

Result<std::string> eval_conjugate_gradient_call(const std::string& formula_arg,
                                                 const std::string& x0_arg,
                                                 const std::string& tol_arg,
                                                 const std::string& max_iter_arg) {
    constexpr const char* fn = "conjugate_gradient";
    auto inputs = parse_nd_optim_inputs(formula_arg, x0_arg, fn);
    if (!inputs) {
        return std::unexpected(inputs.error());
    }
    auto tol = parse_optional_positive_number(tol_arg, fn, "tol", 1e-8);
    if (!tol) {
        return std::unexpected(tol.error());
    }
    auto max_iter = parse_optional_positive_int(max_iter_arg, fn, "max_iter", 500);
    if (!max_iter) {
        return std::unexpected(max_iter.error());
    }
    const int n = static_cast<int>(inputs->x0.size());
    auto grad = make_finite_diff_grad(inputs->f, n);
    return format_optim_result(
        conjugate_gradient(inputs->f, grad, inputs->x0, *tol, *max_iter));
}

Result<std::string> eval_rmsprop_call(const std::string& formula_arg, const std::string& x0_arg,
                                      const std::string& alpha_arg,
                                      const std::string& max_iter_arg) {
    constexpr const char* fn = "rmsprop";
    auto inputs = parse_nd_optim_inputs(formula_arg, x0_arg, fn);
    if (!inputs) {
        return std::unexpected(inputs.error());
    }
    auto alpha = parse_optional_positive_number(alpha_arg, fn, "alpha", 0.001);
    if (!alpha) {
        return std::unexpected(alpha.error());
    }
    auto max_iter = parse_optional_positive_int(max_iter_arg, fn, "max_iter", 1000);
    if (!max_iter) {
        return std::unexpected(max_iter.error());
    }
    const int n = static_cast<int>(inputs->x0.size());
    auto grad = make_finite_diff_grad(inputs->f, n);
    return format_optim_result(
        rmsprop(inputs->f, grad, inputs->x0, *alpha, 0.9, 1e-8, *max_iter));
}

Result<std::string> eval_adadelta_call(const std::string& formula_arg, const std::string& x0_arg,
                                       const std::string& lr_arg, const std::string& max_iter_arg) {
    constexpr const char* fn = "adadelta";
    auto inputs = parse_nd_optim_inputs(formula_arg, x0_arg, fn);
    if (!inputs) {
        return std::unexpected(inputs.error());
    }
    auto lr = parse_optional_positive_number(lr_arg, fn, "lr", 1.0);
    if (!lr) {
        return std::unexpected(lr.error());
    }
    auto max_iter = parse_optional_positive_int(max_iter_arg, fn, "max_iter", 1000);
    if (!max_iter) {
        return std::unexpected(max_iter.error());
    }
    const int n = static_cast<int>(inputs->x0.size());
    auto grad = make_finite_diff_grad(inputs->f, n);
    return format_optim_result(
        adadelta(inputs->f, grad, inputs->x0, *lr, 0.95, 1e-6, *max_iter));
}

Result<std::string> eval_golden_section_call(const std::string& formula_arg,
                                             const std::string& a_arg, const std::string& b_arg,
                                             const std::string& tol_arg) {
    constexpr const char* fn = "golden_section";
    auto expr = parse_sym_quoted_expr(formula_arg, fn);
    if (!expr) {
        return std::unexpected(expr.error());
    }
    double a = 0.0;
    double b = 0.0;
    if (!parse_number(trim_copy(a_arg), a) || !parse_number(trim_copy(b_arg), b)) {
        return std::unexpected(
            DomainError{fn, "expected golden_section(\"formula\", a, b[, tol])"});
    }
    auto tol = parse_optional_positive_number(tol_arg, fn, "tol", 1e-8);
    if (!tol) {
        return std::unexpected(tol.error());
    }
    auto expr_ptr = std::make_shared<SymExpr>(std::move(*expr));
    Func1D f = [expr_ptr](double x) { return sym_eval(*expr_ptr, build_optim_env({x})); };
    const double x_opt = golden_section(f, a, b, *tol);
    return format_scalar_optim_result(x_opt, f(x_opt));
}

Result<std::string> eval_levenberg_marquardt_call(const std::string& formulas_arg,
                                                  const std::string& x0_arg,
                                                  const std::string& max_iter_arg,
                                                  const std::string& tol_arg) {
    constexpr const char* fn = "levenberg_marquardt";
    auto exprs = parse_sym_semicolon_formulas(formulas_arg, fn);
    if (!exprs) {
        return std::unexpected(exprs.error());
    }
    auto x0 = parse_bracket_vector_literal(x0_arg, fn);
    if (!x0) {
        return std::unexpected(x0.error());
    }
    if (x0->empty()) {
        return std::unexpected(DomainError{fn, "expected non-empty initial point vector x0"});
    }
    auto max_iter = parse_optional_positive_int(max_iter_arg, fn, "max_iter", 200);
    if (!max_iter) {
        return std::unexpected(max_iter.error());
    }
    auto tol = parse_optional_positive_number(tol_arg, fn, "tol", 1e-10);
    if (!tol) {
        return std::unexpected(tol.error());
    }
    std::vector<std::shared_ptr<SymExpr>> expr_ptrs;
    expr_ptrs.reserve(exprs->size());
    for (auto& expr : *exprs) {
        expr_ptrs.push_back(std::make_shared<SymExpr>(std::move(expr)));
    }
    ResidualFunc residuals = [expr_ptrs](const std::vector<double>& x) {
        const auto env = build_optim_env(x);
        std::vector<double> values;
        values.reserve(expr_ptrs.size());
        for (const auto& expr_ptr : expr_ptrs) {
            values.push_back(sym_eval(*expr_ptr, env));
        }
        return values;
    };
    return format_optim_result(levenberg_marquardt(residuals, *x0, *max_iter, *tol));
}

Result<std::string> eval_cmaes_call(const std::string& formula_arg, const std::string& x0_arg,
                                    const std::string& sigma_arg,
                                    const std::string& max_iter_arg,
                                    const std::string& seed_arg) {
    constexpr const char* fn = "cmaes";
    auto expr = parse_sym_quoted_expr(formula_arg, fn);
    if (!expr) {
        return std::unexpected(expr.error());
    }
    auto x0 = parse_bracket_vector_literal(x0_arg, fn);
    if (!x0) {
        return std::unexpected(x0.error());
    }
    if (x0->empty()) {
        return std::unexpected(DomainError{fn, "expected non-empty initial point vector x0"});
    }
    double sigma = 0.0;
    double max_iter_d = 0.0;
    if (!parse_number(trim_copy(sigma_arg), sigma) ||
        !parse_number(trim_copy(max_iter_arg), max_iter_d)) {
        return std::unexpected(
            DomainError{fn, "expected cmaes(\"formula\", x0, sigma0, max_iter[, seed])"});
    }
    const int max_iter = static_cast<int>(max_iter_d);
    if (max_iter < 1 || max_iter_d != max_iter) {
        return std::unexpected(DomainError{fn, "expected positive integer max_iter"});
    }
    unsigned seed = 42;
    if (!seed_arg.empty()) {
        double seed_d = 0.0;
        if (!parse_number(trim_copy(seed_arg), seed_d) || seed_d < 0.0 ||
            std::floor(seed_d) != seed_d) {
            return std::unexpected(DomainError{fn, "expected non-negative integer seed"});
        }
        seed = static_cast<unsigned>(seed_d);
    }
    SymExpr parsed = std::move(*expr);
    auto expr_ptr = std::make_shared<SymExpr>(std::move(parsed));
    const size_t dim = x0->size();
    FuncND f = [expr_ptr, dim](const std::vector<double>& x) {
        return sym_eval(*expr_ptr, build_optim_env(x));
    };
    return format_optim_result(cmaes(f, *x0, sigma, max_iter, seed, [] {
        return ms::interp::repl_cancel_requested();
    }));
}

axiom::Axiom make_axiom_repl_engine() {
    return axiom::Axiom(
        axiom::EvolutionConfig{.population_size = 2},
        axiom::PrimitiveRegistry::build_from_ms_namespace());
}

Result<axiom::Algorithm> make_axiom_algorithm(const std::string& expr_arg, const char* fn) {
    std::string expr_text;
    if (!parse_quoted_string(expr_arg, expr_text)) {
        return std::unexpected(DomainError{fn, "expected quoted expression string"});
    }
    axiom::Algorithm algo{};
    algo.representation = ms::Sym(expr_text.c_str());
    return algo;
}

Result<Matrix<double>> eval_axiom_evaluate_call(const std::string& expr_arg,
                                                const Matrix<double>& inputs, const char* fn) {
    auto algo = make_axiom_algorithm(expr_arg, fn);
    if (!algo) {
        return std::unexpected(algo.error());
    }
    return make_axiom_repl_engine().evaluate(*algo, inputs);
}

Result<double> eval_axiom_mse_fitness_call(const std::string& expr_arg,
                                           const Matrix<double>& inputs,
                                           const std::vector<double>& targets, const char* fn) {
    auto algo = make_axiom_algorithm(expr_arg, fn);
    if (!algo) {
        return std::unexpected(algo.error());
    }
    return make_axiom_repl_engine().mse_fitness(*algo, inputs, targets);
}

Result<double> eval_axiom_rmse_fitness_call(const std::string& expr_arg,
                                            const Matrix<double>& inputs,
                                            const std::vector<double>& targets, const char* fn) {
    auto algo = make_axiom_algorithm(expr_arg, fn);
    if (!algo) {
        return std::unexpected(algo.error());
    }
    return make_axiom_repl_engine().rmse_fitness(*algo, inputs, targets);
}

Result<double> eval_axiom_gria_fitness_call(const std::string& expr_arg,
                                            const Matrix<double>& data, const char* fn) {
    auto algo = make_axiom_algorithm(expr_arg, fn);
    if (!algo) {
        return std::unexpected(algo.error());
    }
    return make_axiom_repl_engine().gria_fitness(*algo, data);
}

Result<double> eval_axiom_evolve_call(const Matrix<double>& data, size_t population_size,
                                      size_t max_generations) {
    axiom::EvolutionConfig cfg{};
    cfg.population_size = population_size;
    cfg.max_generations = max_generations;
    axiom::Axiom engine(cfg, axiom::PrimitiveRegistry::build_from_ms_namespace());
    const auto evolved = engine
                          .evolve(
                              [&](const axiom::Algorithm& a) {
                                  return engine.gria_fitness(a, data);
                              },
                              [](const axiom::Algorithm& a) { return a.fitness > 0.45; });
    if (!evolved) {
        return std::unexpected(evolved.error());
    }
    return evolved->fitness;
}

Result<std::vector<double>> resolve_axiom_targets(const std::string& text, const char* fn,
                                                    const std::function<Result<Matrix<double>>(const std::string&)>&
                                                        resolve_matrix_arg) {
    auto bracket = parse_bracket_vector_literal(text, fn);
    if (bracket) {
        return *bracket;
    }
    auto matrix = resolve_matrix_arg(text);
    if (!matrix) {
        return std::unexpected(matrix.error());
    }
    return matrix_to_coeff_vector(*matrix, fn);
}

Result<std::vector<std::pair<double, double>>> parse_bounds_pairs_literal(const std::string& text,
                                                                          const char* fn) {
    auto matrix = parse_bracket_matrix_literal(text, fn);
    if (!matrix) {
        return std::unexpected(matrix.error());
    }
    if (matrix->cols() != 2) {
        return std::unexpected(
            DomainError{fn, "expected Nx2 bounds matrix [[lo, hi], ...]"});
    }
    if (matrix->rows() == 0) {
        return std::unexpected(DomainError{fn, "expected non-empty bounds matrix"});
    }
    std::vector<std::pair<double, double>> bounds;
    bounds.reserve(matrix->rows());
    for (size_t i = 0; i < matrix->rows(); ++i) {
        const double lo = (*matrix)(i, 0);
        const double hi = (*matrix)(i, 1);
        if (lo >= hi) {
            return std::unexpected(DomainError{fn, "expected lo < hi for each bounds row"});
        }
        bounds.emplace_back(lo, hi);
    }
    return bounds;
}

Result<unsigned> parse_optional_seed(const std::string& seed_arg, const char* fn) {
    unsigned seed = 42;
    if (seed_arg.empty()) {
        return seed;
    }
    double seed_d = 0.0;
    if (!parse_number(trim_copy(seed_arg), seed_d) || seed_d < 0.0 ||
        std::floor(seed_d) != seed_d) {
        return std::unexpected(DomainError{fn, "expected non-negative integer seed"});
    }
    return static_cast<unsigned>(seed_d);
}

Func1D make_scalar_formula_func(SymExpr expr) {
    auto expr_ptr = std::make_shared<SymExpr>(std::move(expr));
    return [expr_ptr](double x) { return sym_eval(*expr_ptr, build_optim_env({x})); };
}

using BracketRootSolver = double (*)(Func1D, double, double, double, int);

Result<std::string> eval_bracket_root_call(const char* fn, BracketRootSolver solver,
                                           const std::string& formula_arg,
                                           const std::string& a_arg, const std::string& b_arg,
                                           const std::string& tol_arg,
                                           const std::string& max_iter_arg) {
    auto expr = parse_sym_quoted_expr(formula_arg, fn);
    if (!expr) {
        return std::unexpected(expr.error());
    }
    double a = 0.0;
    double b = 0.0;
    if (!parse_number(trim_copy(a_arg), a) || !parse_number(trim_copy(b_arg), b)) {
        return std::unexpected(
            DomainError{fn, std::string("expected ") + fn + "(\"formula\", a, b[, tol[, max_iter]])"});
    }
    auto tol = parse_optional_positive_number(tol_arg, fn, "tol", 1e-10);
    if (!tol) {
        return std::unexpected(tol.error());
    }
    auto max_iter = parse_optional_positive_int(max_iter_arg, fn, "max_iter", 200);
    if (!max_iter) {
        return std::unexpected(max_iter.error());
    }
    auto f = make_scalar_formula_func(std::move(*expr));
    const double x_opt = solver(f, a, b, *tol, *max_iter);
    return format_scalar_optim_result(x_opt, f(x_opt));
}

Result<std::string> eval_secant_call(const std::string& formula_arg, const std::string& x0_arg,
                                     const std::string& x1_arg, const std::string& tol_arg,
                                     const std::string& max_iter_arg) {
    constexpr const char* fn = "secant";
    auto expr = parse_sym_quoted_expr(formula_arg, fn);
    if (!expr) {
        return std::unexpected(expr.error());
    }
    double x0 = 0.0;
    double x1 = 0.0;
    if (!parse_number(trim_copy(x0_arg), x0) || !parse_number(trim_copy(x1_arg), x1)) {
        return std::unexpected(
            DomainError{fn, "expected secant(\"formula\", x0, x1[, tol[, max_iter]])"});
    }
    auto tol = parse_optional_positive_number(tol_arg, fn, "tol", 1e-10);
    if (!tol) {
        return std::unexpected(tol.error());
    }
    auto max_iter = parse_optional_positive_int(max_iter_arg, fn, "max_iter", 100);
    if (!max_iter) {
        return std::unexpected(max_iter.error());
    }
    auto f = make_scalar_formula_func(std::move(*expr));
    const double x_opt = secant(f, x0, x1, *tol, *max_iter);
    return format_scalar_optim_result(x_opt, f(x_opt));
}

Result<std::string> eval_halley_call(const std::string& f_arg, const std::string& df_arg,
                                     const std::string& d2f_arg, const std::string& x0_arg,
                                     const std::string& tol_arg, const std::string& max_iter_arg) {
    constexpr const char* fn = "halley";
    auto f_expr = parse_sym_quoted_expr(f_arg, fn);
    if (!f_expr) {
        return std::unexpected(f_expr.error());
    }
    auto df_expr = parse_sym_quoted_expr(df_arg, fn);
    if (!df_expr) {
        return std::unexpected(df_expr.error());
    }
    auto d2f_expr = parse_sym_quoted_expr(d2f_arg, fn);
    if (!d2f_expr) {
        return std::unexpected(d2f_expr.error());
    }
    double x0 = 0.0;
    if (!parse_number(trim_copy(x0_arg), x0)) {
        return std::unexpected(
            DomainError{fn, "expected halley(\"f\", \"df\", \"d2f\", x0[, tol[, max_iter]])"});
    }
    auto tol = parse_optional_positive_number(tol_arg, fn, "tol", 1e-12);
    if (!tol) {
        return std::unexpected(tol.error());
    }
    auto max_iter = parse_optional_positive_int(max_iter_arg, fn, "max_iter", 50);
    if (!max_iter) {
        return std::unexpected(max_iter.error());
    }
    auto f = make_scalar_formula_func(std::move(*f_expr));
    auto df = make_scalar_formula_func(std::move(*df_expr));
    auto d2f = make_scalar_formula_func(std::move(*d2f_expr));
    const double x_opt = halley(f, df, d2f, x0, *tol, *max_iter);
    return format_scalar_optim_result(x_opt, f(x_opt));
}

Result<std::string> eval_fixed_point_call(const std::string& formula_arg, const std::string& x0_arg,
                                          const std::string& tol_arg,
                                          const std::string& max_iter_arg) {
    constexpr const char* fn = "fixed_point";
    auto expr = parse_sym_quoted_expr(formula_arg, fn);
    if (!expr) {
        return std::unexpected(expr.error());
    }
    double x0 = 0.0;
    if (!parse_number(trim_copy(x0_arg), x0)) {
        return std::unexpected(
            DomainError{fn, "expected fixed_point(\"formula\", x0[, tol[, max_iter]])"});
    }
    auto tol = parse_optional_positive_number(tol_arg, fn, "tol", 1e-10);
    if (!tol) {
        return std::unexpected(tol.error());
    }
    auto max_iter = parse_optional_positive_int(max_iter_arg, fn, "max_iter", 200);
    if (!max_iter) {
        return std::unexpected(max_iter.error());
    }
    auto g = make_scalar_formula_func(std::move(*expr));
    const double x_opt = fixed_point(g, x0, *tol, *max_iter);
    return format_scalar_optim_result(x_opt, g(x_opt));
}

Result<std::string> eval_simulated_annealing_call(const std::string& formula_arg,
                                                  const std::string& x0_arg,
                                                  const std::string& t0_arg,
                                                  const std::string& cooling_arg,
                                                  const std::string& max_iter_arg,
                                                  const std::string& seed_arg) {
    constexpr const char* fn = "simulated_annealing";
    auto inputs = parse_nd_optim_inputs(formula_arg, x0_arg, fn);
    if (!inputs) {
        return std::unexpected(inputs.error());
    }
    auto t0 = parse_optional_positive_number(t0_arg, fn, "T0", 1.0);
    if (!t0) {
        return std::unexpected(t0.error());
    }
    auto cooling = parse_optional_positive_number(cooling_arg, fn, "cooling", 0.995);
    if (!cooling) {
        return std::unexpected(cooling.error());
    }
    if (*cooling >= 1.0) {
        return std::unexpected(DomainError{fn, "expected cooling in (0, 1)"});
    }
    auto max_iter = parse_optional_positive_int(max_iter_arg, fn, "max_iter", 10000);
    if (!max_iter) {
        return std::unexpected(max_iter.error());
    }
    auto seed = parse_optional_seed(seed_arg, fn);
    if (!seed) {
        return std::unexpected(seed.error());
    }
    return format_optim_result(simulated_annealing(inputs->f, inputs->x0, *t0, *cooling, *max_iter,
                                                   *seed));
}

Result<std::string> eval_differential_evolution_call(
    const std::string& formula_arg, const std::string& bounds_arg, const std::string& pop_arg,
    const std::string& f_arg, const std::string& cr_arg, const std::string& max_iter_arg,
    const std::string& seed_arg) {
    constexpr const char* fn = "differential_evolution";
    auto expr = parse_sym_quoted_expr(formula_arg, fn);
    if (!expr) {
        return std::unexpected(expr.error());
    }
    auto bounds = parse_bounds_pairs_literal(bounds_arg, fn);
    if (!bounds) {
        return std::unexpected(bounds.error());
    }
    auto pop = parse_optional_positive_int(pop_arg, fn, "pop", 15);
    if (!pop) {
        return std::unexpected(pop.error());
    }
    auto f_scale = parse_optional_positive_number(f_arg, fn, "F", 0.8);
    if (!f_scale) {
        return std::unexpected(f_scale.error());
    }
    auto cr = parse_optional_positive_number(cr_arg, fn, "CR", 0.9);
    if (!cr) {
        return std::unexpected(cr.error());
    }
    if (*cr > 1.0) {
        return std::unexpected(DomainError{fn, "expected CR in (0, 1]"});
    }
    auto max_iter = parse_optional_positive_int(max_iter_arg, fn, "max_iter", 1000);
    if (!max_iter) {
        return std::unexpected(max_iter.error());
    }
    auto seed = parse_optional_seed(seed_arg, fn);
    if (!seed) {
        return std::unexpected(seed.error());
    }
    auto expr_ptr = std::make_shared<SymExpr>(std::move(*expr));
    const size_t dim = bounds->size();
    FuncND objective = [expr_ptr, dim](const std::vector<double>& x) {
        return sym_eval(*expr_ptr, build_optim_env(x));
    };
    return format_optim_result(
        differential_evolution(objective, *bounds, *pop, *f_scale, *cr, *max_iter, *seed));
}

Result<std::string> eval_particle_swarm_call(const std::string& formula_arg,
                                             const std::string& bounds_arg,
                                             const std::string& n_particles_arg,
                                             const std::string& max_iter_arg,
                                             const std::string& seed_arg) {
    constexpr const char* fn = "particle_swarm";
    auto expr = parse_sym_quoted_expr(formula_arg, fn);
    if (!expr) {
        return std::unexpected(expr.error());
    }
    auto bounds = parse_bounds_pairs_literal(bounds_arg, fn);
    if (!bounds) {
        return std::unexpected(bounds.error());
    }
    auto n_particles = parse_optional_positive_int(n_particles_arg, fn, "n_particles", 30);
    if (!n_particles) {
        return std::unexpected(n_particles.error());
    }
    auto max_iter = parse_optional_positive_int(max_iter_arg, fn, "max_iter", 500);
    if (!max_iter) {
        return std::unexpected(max_iter.error());
    }
    auto seed = parse_optional_seed(seed_arg, fn);
    if (!seed) {
        return std::unexpected(seed.error());
    }
    auto expr_ptr = std::make_shared<SymExpr>(std::move(*expr));
    const size_t dim = bounds->size();
    FuncND objective = [expr_ptr, dim](const std::vector<double>& x) {
        return sym_eval(*expr_ptr, build_optim_env(x));
    };
    return format_optim_result(
        particle_swarm(objective, *bounds, *n_particles, *max_iter, *seed));
}

Result<std::vector<SymExpr>> parse_sym_semicolon_formulas(const std::string& formula_arg,
                                                          const char* fn) {
    std::string formulas_text;
    if (!parse_quoted_string(formula_arg, formulas_text)) {
        return std::unexpected(
            DomainError{fn, "expected quoted semicolon-separated formula string"});
    }
    std::vector<SymExpr> parsed;
    std::stringstream ss(formulas_text);
    std::string part;
    while (std::getline(ss, part, ';')) {
        part = trim_copy(part);
        if (part.empty()) {
            continue;
        }
        auto expr = sym_parse(part);
        if (!expr) {
            return std::unexpected(DomainError{fn, expr.error().message});
        }
        parsed.push_back(std::move(*expr));
    }
    if (parsed.empty()) {
        return std::unexpected(DomainError{fn, "expected at least one formula"});
    }
    return parsed;
}

std::map<std::string, double> build_vec_ode_env(double t, const std::vector<double>& y) {
    std::map<std::string, double> env;
    env["t"] = t;
    for (size_t i = 0; i < y.size(); ++i) {
        env["y" + std::to_string(i)] = y[i];
    }
    return env;
}

std::map<std::string, double> build_vec_accel_env(double t, const std::vector<double>& q) {
    std::map<std::string, double> env;
    env["t"] = t;
    for (size_t i = 0; i < q.size(); ++i) {
        env["q" + std::to_string(i)] = q[i];
    }
    return env;
}

Result<std::string> format_ode_trajectory_vec(const OdeResultVec& result) {
    if (result.t.size() != result.y.size()) {
        return std::unexpected(DomainError{"ode", "internal vector trajectory size mismatch"});
    }
    if (result.t.empty()) {
        return std::string("traj =\n");
    }
    const size_t n_state = result.y.front().size();
    Matrix<double> out(result.t.size(), 1 + n_state);
    for (size_t i = 0; i < result.t.size(); ++i) {
        if (result.y[i].size() != n_state) {
            return std::unexpected(
                DomainError{"ode", "internal vector state size mismatch"});
        }
        out(i, 0) = result.t[i];
        for (size_t j = 0; j < n_state; ++j) {
            out(i, 1 + j) = result.y[i][j];
        }
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

Result<std::string> format_ode_verlet_trajectory(const OdeVerletResult& result) {
    if (result.t.size() != result.q.size() || result.t.size() != result.v.size()) {
        return std::unexpected(DomainError{"ode", "internal Verlet trajectory size mismatch"});
    }
    Matrix<double> out(result.t.size(), 3);
    for (size_t i = 0; i < result.t.size(); ++i) {
        out(i, 0) = result.t[i];
        out(i, 1) = result.q[i];
        out(i, 2) = result.v[i];
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

Result<std::string> format_ode_verlet_trajectory_vec(const OdeVerletResultVec& result) {
    if (result.t.size() != result.q.size() || result.t.size() != result.v.size()) {
        return std::unexpected(DomainError{"ode", "internal vector Verlet trajectory size mismatch"});
    }
    if (result.t.empty()) {
        return std::string("traj =\n");
    }
    const size_t n_dim = result.q.front().size();
    if (result.v.front().size() != n_dim) {
        return std::unexpected(DomainError{"ode", "internal vector Verlet q/v size mismatch"});
    }
    Matrix<double> out(result.t.size(), 1 + 2 * n_dim);
    for (size_t i = 0; i < result.t.size(); ++i) {
        if (result.q[i].size() != n_dim || result.v[i].size() != n_dim) {
            return std::unexpected(DomainError{"ode", "internal vector Verlet state size mismatch"});
        }
        out(i, 0) = result.t[i];
        for (size_t j = 0; j < n_dim; ++j) {
            out(i, 1 + j) = result.q[i][j];
            out(i, 1 + n_dim + j) = result.v[i][j];
        }
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

Result<std::string> eval_ode_verlet_call(const std::string& formula_arg, const std::string& t0_arg,
                                         const std::string& q0_arg, const std::string& v0_arg,
                                         const std::string& t_end_arg, const std::string& steps_arg) {
    constexpr const char* fn = "ode_verlet";
    auto expr = parse_sym_quoted_expr(formula_arg, fn);
    if (!expr) {
        return std::unexpected(expr.error());
    }
    double t0 = 0.0;
    double q0 = 0.0;
    double v0 = 0.0;
    double t_end = 0.0;
    double steps_d = 0.0;
    if (!parse_number(trim_copy(t0_arg), t0) || !parse_number(trim_copy(q0_arg), q0) ||
        !parse_number(trim_copy(v0_arg), v0) || !parse_number(trim_copy(t_end_arg), t_end) ||
        !parse_number(trim_copy(steps_arg), steps_d)) {
        return std::unexpected(DomainError{
            fn, "expected ode_verlet(\"formula\", t0, q0, v0, t_end, steps)"});
    }
    const int steps_i = static_cast<int>(steps_d);
    if (steps_i < 0 || steps_d != steps_i) {
        return std::unexpected(DomainError{fn, "expected non-negative integer steps"});
    }
    SymExpr parsed = std::move(*expr);
    auto expr_ptr = std::make_shared<SymExpr>(std::move(parsed));
    OdeAccelFunc a = [expr_ptr](double t, double q) {
        return sym_eval(*expr_ptr, {{"t", t}, {"q", q}});
    };
    return format_ode_verlet_trajectory(
        ode_verlet(a, t0, q0, v0, t_end, static_cast<size_t>(steps_i)));
}

Result<std::string> eval_ode_vec_fixed_step_call(
    const std::string& fn, const std::string& formula_arg, const std::string& t0_arg,
    const std::string& y0_arg, const std::string& t_end_arg, const std::string& steps_arg,
    OdeResultVec (*solver)(OdeFuncVec, double, const std::vector<double>&, double, size_t)) {
    auto exprs = parse_sym_semicolon_formulas(formula_arg, fn.c_str());
    if (!exprs) {
        return std::unexpected(exprs.error());
    }
    auto y0 = parse_bracket_vector_literal(y0_arg, fn.c_str());
    if (!y0) {
        return std::unexpected(y0.error());
    }
    if (exprs->size() != y0->size()) {
        return std::unexpected(DomainError{
            fn.c_str(),
            "formula count must match initial-condition vector length"});
    }
    double t0 = 0.0;
    double t_end = 0.0;
    double steps_d = 0.0;
    if (!parse_number(trim_copy(t0_arg), t0) || !parse_number(trim_copy(t_end_arg), t_end) ||
        !parse_number(trim_copy(steps_arg), steps_d)) {
        return std::unexpected(DomainError{
            fn.c_str(),
            std::string("expected ") + fn + "(\"f0;f1;...\", t0, y0, t_end, steps)"});
    }
    const int steps_i = static_cast<int>(steps_d);
    if (steps_i < 0 || steps_d != steps_i) {
        return std::unexpected(DomainError{fn.c_str(), "expected non-negative integer steps"});
    }
    auto exprs_ptr = std::make_shared<std::vector<SymExpr>>(std::move(*exprs));
    OdeFuncVec f = [exprs_ptr](double t, const std::vector<double>& y) {
        const auto env = build_vec_ode_env(t, y);
        std::vector<double> out;
        out.reserve(exprs_ptr->size());
        for (const auto& expr : *exprs_ptr) {
            out.push_back(sym_eval(expr, env));
        }
        return out;
    };
    return format_ode_trajectory_vec(
        solver(f, t0, *y0, t_end, static_cast<size_t>(steps_i)));
}

Result<std::string> eval_ode_rk45_vec_call(const std::string& formula_arg, const std::string& t0_arg,
                                           const std::string& y0_arg, const std::string& t_end_arg,
                                           const std::string& rtol_arg, const std::string& atol_arg) {
    constexpr const char* fn = "ode_rk45_vec";
    auto exprs = parse_sym_semicolon_formulas(formula_arg, fn);
    if (!exprs) {
        return std::unexpected(exprs.error());
    }
    auto y0 = parse_bracket_vector_literal(y0_arg, fn);
    if (!y0) {
        return std::unexpected(y0.error());
    }
    if (exprs->size() != y0->size()) {
        return std::unexpected(DomainError{
            fn, "formula count must match initial-condition vector length"});
    }
    double t0 = 0.0;
    double t_end = 0.0;
    double rtol = 0.0;
    double atol = 0.0;
    if (!parse_number(trim_copy(t0_arg), t0) || !parse_number(trim_copy(t_end_arg), t_end) ||
        !parse_number(trim_copy(rtol_arg), rtol) || !parse_number(trim_copy(atol_arg), atol)) {
        return std::unexpected(DomainError{
            fn, "expected ode_rk45_vec(\"f0;f1;...\", t0, y0, t_end, rtol, atol)"});
    }
    auto exprs_ptr = std::make_shared<std::vector<SymExpr>>(std::move(*exprs));
    OdeFuncVec f = [exprs_ptr](double t, const std::vector<double>& y) {
        const auto env = build_vec_ode_env(t, y);
        std::vector<double> out;
        out.reserve(exprs_ptr->size());
        for (const auto& expr : *exprs_ptr) {
            out.push_back(sym_eval(expr, env));
        }
        return out;
    };
    return format_ode_trajectory_vec(ode_rk45_vec(f, t0, *y0, t_end, rtol, atol));
}

Result<std::string> eval_ode_rosenbrock23_vec_call(const std::string& formula_arg,
                                                   const std::string& t0_arg,
                                                   const std::string& y0_arg,
                                                   const std::string& t_end_arg,
                                                   const std::string& steps_arg) {
    constexpr const char* fn = "ode_rosenbrock23_vec";
    auto exprs = parse_sym_semicolon_formulas(formula_arg, fn);
    if (!exprs) {
        return std::unexpected(exprs.error());
    }
    auto y0 = parse_bracket_vector_literal(y0_arg, fn);
    if (!y0) {
        return std::unexpected(y0.error());
    }
    if (exprs->size() != y0->size()) {
        return std::unexpected(DomainError{
            fn, "formula count must match initial-condition vector length"});
    }
    double t0 = 0.0;
    double t_end = 0.0;
    double steps_d = 0.0;
    if (!parse_number(trim_copy(t0_arg), t0) || !parse_number(trim_copy(t_end_arg), t_end) ||
        !parse_number(trim_copy(steps_arg), steps_d)) {
        return std::unexpected(DomainError{
            fn, "expected ode_rosenbrock23_vec(\"f0;f1;...\", t0, y0, t_end, steps)"});
    }
    const int steps_i = static_cast<int>(steps_d);
    if (steps_i < 0 || steps_d != steps_i) {
        return std::unexpected(DomainError{fn, "expected non-negative integer steps"});
    }
    auto exprs_ptr = std::make_shared<std::vector<SymExpr>>(std::move(*exprs));
    OdeFuncVec f = [exprs_ptr](double t, const std::vector<double>& y) {
        const auto env = build_vec_ode_env(t, y);
        std::vector<double> out;
        out.reserve(exprs_ptr->size());
        for (const auto& expr : *exprs_ptr) {
            out.push_back(sym_eval(expr, env));
        }
        return out;
    };
    return format_ode_trajectory_vec(ode_rosenbrock23_vec(f, t0, *y0, t_end, steps_i));
}

Result<std::string> eval_ode_verlet_vec_call(const std::string& formula_arg,
                                             const std::string& t0_arg, const std::string& q0_arg,
                                             const std::string& v0_arg, const std::string& t_end_arg,
                                             const std::string& steps_arg) {
    constexpr const char* fn = "ode_verlet_vec";
    auto exprs = parse_sym_semicolon_formulas(formula_arg, fn);
    if (!exprs) {
        return std::unexpected(exprs.error());
    }
    auto q0 = parse_bracket_vector_literal(q0_arg, fn);
    if (!q0) {
        return std::unexpected(q0.error());
    }
    auto v0 = parse_bracket_vector_literal(v0_arg, fn);
    if (!v0) {
        return std::unexpected(v0.error());
    }
    if (exprs->size() != q0->size() || q0->size() != v0->size()) {
        return std::unexpected(DomainError{
            fn, "formula count must match q0 and v0 vector lengths"});
    }
    double t0 = 0.0;
    double t_end = 0.0;
    double steps_d = 0.0;
    if (!parse_number(trim_copy(t0_arg), t0) || !parse_number(trim_copy(t_end_arg), t_end) ||
        !parse_number(trim_copy(steps_arg), steps_d)) {
        return std::unexpected(DomainError{
            fn, "expected ode_verlet_vec(\"a0;a1;...\", t0, q0, v0, t_end, steps)"});
    }
    const int steps_i = static_cast<int>(steps_d);
    if (steps_i < 0 || steps_d != steps_i) {
        return std::unexpected(DomainError{fn, "expected non-negative integer steps"});
    }
    auto exprs_ptr = std::make_shared<std::vector<SymExpr>>(std::move(*exprs));
    OdeAccelFuncVec a = [exprs_ptr](double t, const std::vector<double>& q) {
        const auto env = build_vec_accel_env(t, q);
        std::vector<double> out;
        out.reserve(exprs_ptr->size());
        for (const auto& expr : *exprs_ptr) {
            out.push_back(sym_eval(expr, env));
        }
        return out;
    };
    return format_ode_verlet_trajectory_vec(
        ode_verlet_vec(a, t0, *q0, *v0, t_end, static_cast<size_t>(steps_i)));
}

std::map<std::string, double> build_dae_env(double t, const std::vector<double>& y,
                                            const std::vector<double>& z) {
    auto env = build_vec_ode_env(t, y);
    for (size_t i = 0; i < z.size(); ++i) {
        env["z" + std::to_string(i)] = z[i];
    }
    return env;
}

Result<std::string> format_dae_trajectory(const DaeResult& result) {
    if (result.t.size() != result.y.size() || result.t.size() != result.z.size()) {
        return std::unexpected(DomainError{"ode", "internal DAE trajectory size mismatch"});
    }
    std::ostringstream oss;
    oss << "converged = " << (result.converged ? "true" : "false") << "\n";
    if (result.t.empty()) {
        oss << "y_traj =\n";
        oss << "z_traj =\n";
        return oss.str();
    }
    const size_t n_y = result.y.front().size();
    const size_t n_z = result.z.front().size();
    Matrix<double> y_out(result.t.size(), 1 + n_y);
    Matrix<double> z_out(result.t.size(), 1 + n_z);
    for (size_t i = 0; i < result.t.size(); ++i) {
        if (result.y[i].size() != n_y || result.z[i].size() != n_z) {
            return std::unexpected(DomainError{"ode", "internal DAE state size mismatch"});
        }
        y_out(i, 0) = result.t[i];
        z_out(i, 0) = result.t[i];
        for (size_t j = 0; j < n_y; ++j) {
            y_out(i, 1 + j) = result.y[i][j];
        }
        for (size_t j = 0; j < n_z; ++j) {
            z_out(i, 1 + j) = result.z[i][j];
        }
    }
    oss << "y_traj =\n";
    oss << std::fixed << std::setprecision(6);
    for (size_t i = 0; i < y_out.rows(); ++i) {
        oss << "  [";
        for (size_t j = 0; j < y_out.cols(); ++j) {
            if (j > 0) {
                oss << ", ";
            }
            oss << y_out(i, j);
        }
        oss << "]\n";
    }
    oss << "z_traj =\n";
    for (size_t i = 0; i < z_out.rows(); ++i) {
        oss << "  [";
        for (size_t j = 0; j < z_out.cols(); ++j) {
            if (j > 0) {
                oss << ", ";
            }
            oss << z_out(i, j);
        }
        oss << "]\n";
    }
    return oss.str();
}

Result<std::string> format_ode_bvp_trajectory(const OdeBvpResult& result) {
    if (result.t.size() != result.y.size() || result.t.size() != result.yp.size()) {
        return std::unexpected(DomainError{"ode", "internal BVP trajectory size mismatch"});
    }
    std::ostringstream oss;
    oss << "converged = " << (result.converged ? "true" : "false") << "\n";
    oss << "iterations = " << result.iterations << "\n";
    if (result.t.empty()) {
        oss << "traj =\n";
        return oss.str();
    }
    Matrix<double> out(result.t.size(), 3);
    for (size_t i = 0; i < result.t.size(); ++i) {
        out(i, 0) = result.t[i];
        out(i, 1) = result.y[i];
        out(i, 2) = result.yp[i];
    }
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

Result<std::string> format_ode_event_trajectory(const OdeEventResult& result) {
    if (result.t.size() != result.y.size()) {
        return std::unexpected(DomainError{"ode", "internal event trajectory size mismatch"});
    }
    if (result.event_times.size() != result.event_values.size()) {
        return std::unexpected(DomainError{"ode", "internal event value size mismatch"});
    }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "traj =\n";
    for (size_t i = 0; i < result.t.size(); ++i) {
        oss << "  [" << result.t[i] << ", " << result.y[i] << "]\n";
    }
    oss << "event_count = " << result.event_times.size() << "\n";
    if (!result.event_times.empty()) {
        oss << "events =\n";
        for (size_t i = 0; i < result.event_times.size(); ++i) {
            oss << "  [" << result.event_times[i] << ", " << result.event_values[i] << "]\n";
        }
    }
    oss << "event_values =\n";
    for (size_t i = 0; i < result.event_values.size(); ++i) {
        oss << "  [" << result.event_values[i] << "]\n";
    }
    return oss.str();
}

Result<std::string> eval_ode_dae_index1_call(const std::string& diff_formula_arg,
                                             const std::string& alg_formula_arg,
                                             const std::string& t0_arg,
                                             const std::string& y0_arg,
                                             const std::string& z0_arg,
                                             const std::string& t_end_arg,
                                             const std::string& steps_arg) {
    constexpr const char* fn = "ode_dae_index1";
    auto diff_exprs = parse_sym_semicolon_formulas(diff_formula_arg, fn);
    if (!diff_exprs) {
        return std::unexpected(diff_exprs.error());
    }
    auto alg_exprs = parse_sym_semicolon_formulas(alg_formula_arg, fn);
    if (!alg_exprs) {
        return std::unexpected(alg_exprs.error());
    }
    auto y0 = parse_bracket_vector_literal(y0_arg, fn);
    if (!y0) {
        return std::unexpected(y0.error());
    }
    auto z0 = parse_bracket_vector_literal(z0_arg, fn);
    if (!z0) {
        return std::unexpected(z0.error());
    }
    if (diff_exprs->size() != y0->size()) {
        return std::unexpected(DomainError{
            fn, "differential formula count must match initial y vector length"});
    }
    if (alg_exprs->size() != z0->size()) {
        return std::unexpected(DomainError{
            fn, "algebraic formula count must match initial z vector length"});
    }
    double t0 = 0.0;
    double t_end = 0.0;
    double steps_d = 0.0;
    if (!parse_number(trim_copy(t0_arg), t0) || !parse_number(trim_copy(t_end_arg), t_end) ||
        !parse_number(trim_copy(steps_arg), steps_d)) {
        return std::unexpected(DomainError{
            fn,
            "expected ode_dae_index1(\"f0;f1;...\", \"g0;g1;...\", t0, y0, z0, t_end, steps)"});
    }
    const int steps_i = static_cast<int>(steps_d);
    if (steps_i < 0 || steps_d != steps_i) {
        return std::unexpected(DomainError{fn, "expected non-negative integer steps"});
    }
    auto diff_exprs_ptr = std::make_shared<std::vector<SymExpr>>(std::move(*diff_exprs));
    auto alg_exprs_ptr = std::make_shared<std::vector<SymExpr>>(std::move(*alg_exprs));
    DaeDiffFunc f = [diff_exprs_ptr](double t, const std::vector<double>& y,
                                     const std::vector<double>& z) {
        const auto env = build_dae_env(t, y, z);
        std::vector<double> out;
        out.reserve(diff_exprs_ptr->size());
        for (const auto& expr : *diff_exprs_ptr) {
            out.push_back(sym_eval(expr, env));
        }
        return out;
    };
    DaeAlgFunc g = [alg_exprs_ptr](double t, const std::vector<double>& y,
                                   const std::vector<double>& z) {
        const auto env = build_dae_env(t, y, z);
        std::vector<double> out;
        out.reserve(alg_exprs_ptr->size());
        for (const auto& expr : *alg_exprs_ptr) {
            out.push_back(sym_eval(expr, env));
        }
        return out;
    };
    return format_dae_trajectory(
        ode_dae_index1(f, g, t0, *y0, *z0, t_end, static_cast<size_t>(steps_i)));
}

Result<std::string> eval_ode_bvp_shooting_call(const std::string& formula_arg,
                                               const std::string& t0_arg,
                                               const std::string& y_a_arg,
                                               const std::string& t_end_arg,
                                               const std::string& y_b_arg,
                                               const std::string& steps_arg) {
    constexpr const char* fn = "ode_bvp_shooting";
    auto expr = parse_sym_quoted_expr(formula_arg, fn);
    if (!expr) {
        return std::unexpected(expr.error());
    }
    double t0 = 0.0;
    double y_a = 0.0;
    double t_end = 0.0;
    double y_b = 0.0;
    double steps_d = 0.0;
    if (!parse_number(trim_copy(t0_arg), t0) || !parse_number(trim_copy(y_a_arg), y_a) ||
        !parse_number(trim_copy(t_end_arg), t_end) || !parse_number(trim_copy(y_b_arg), y_b) ||
        !parse_number(trim_copy(steps_arg), steps_d)) {
        return std::unexpected(DomainError{
            fn,
            "expected ode_bvp_shooting(\"formula\", t0, y_a, t_end, y_b, steps) "
            "with env {t, y, yp}"});
    }
    const int steps_i = static_cast<int>(steps_d);
    if (steps_i < 0 || steps_d != steps_i) {
        return std::unexpected(DomainError{fn, "expected non-negative integer steps"});
    }
    SymExpr parsed = std::move(*expr);
    auto expr_ptr = std::make_shared<SymExpr>(std::move(parsed));
    OdeBvpFunc f = [expr_ptr](double t, double y, double yp) {
        return sym_eval(*expr_ptr, {{"t", t}, {"y", y}, {"yp", yp}});
    };
    return format_ode_bvp_trajectory(
        ode_bvp_shooting(f, t0, y_a, t_end, y_b, static_cast<size_t>(steps_i)));
}

Result<std::string> eval_ode_dde_fixed_step_call(const std::string& formula_arg,
                                                 const std::string& history_formula_arg,
                                                 const std::string& t0_arg,
                                                 const std::string& t_end_arg,
                                                 const std::string& tau_arg,
                                                 const std::string& steps_arg) {
    constexpr const char* fn = "ode_dde_fixed_step";
    auto expr = parse_sym_quoted_expr(formula_arg, fn);
    if (!expr) {
        return std::unexpected(expr.error());
    }
    auto hist_expr = parse_sym_quoted_expr(history_formula_arg, fn);
    if (!hist_expr) {
        return std::unexpected(hist_expr.error());
    }
    double t0 = 0.0;
    double t_end = 0.0;
    double tau = 0.0;
    double steps_d = 0.0;
    if (!parse_number(trim_copy(t0_arg), t0) || !parse_number(trim_copy(t_end_arg), t_end) ||
        !parse_number(trim_copy(tau_arg), tau) || !parse_number(trim_copy(steps_arg), steps_d)) {
        return std::unexpected(DomainError{
            fn,
            "expected ode_dde_fixed_step(\"f\", \"history\", t0, t_end, tau, steps) "
            "with f env {t, y, ydelay} and history env {t}"});
    }
    const int steps_i = static_cast<int>(steps_d);
    if (steps_i < 0 || steps_d != steps_i) {
        return std::unexpected(DomainError{fn, "expected non-negative integer steps"});
    }
    SymExpr parsed = std::move(*expr);
    SymExpr hist_parsed = std::move(*hist_expr);
    auto expr_ptr = std::make_shared<SymExpr>(std::move(parsed));
    auto hist_ptr = std::make_shared<SymExpr>(std::move(hist_parsed));
    auto f = [expr_ptr](double t, double y, double ydelay) {
        return sym_eval(*expr_ptr, {{"t", t}, {"y", y}, {"ydelay", ydelay}});
    };
    auto history = [hist_ptr](double t) {
        return sym_eval(*hist_ptr, {{"t", t}});
    };
    return format_ode_trajectory(
        ode_dde_fixed_step(f, history, t0, t_end, tau, static_cast<size_t>(steps_i)));
}

Result<std::string> eval_ode_event_detect_call(const std::string& formula_arg,
                                               const std::string& event_formula_arg,
                                               const std::string& t0_arg,
                                               const std::string& y0_arg,
                                               const std::string& t_end_arg,
                                               const std::string& steps_arg) {
    constexpr const char* fn = "ode_event_detect";
    auto expr = parse_sym_quoted_expr(formula_arg, fn);
    if (!expr) {
        return std::unexpected(expr.error());
    }
    auto event_expr = parse_sym_quoted_expr(event_formula_arg, fn);
    if (!event_expr) {
        return std::unexpected(event_expr.error());
    }
    double t0 = 0.0;
    double y0 = 0.0;
    double t_end = 0.0;
    double steps_d = 0.0;
    if (!parse_number(trim_copy(t0_arg), t0) || !parse_number(trim_copy(y0_arg), y0) ||
        !parse_number(trim_copy(t_end_arg), t_end) ||
        !parse_number(trim_copy(steps_arg), steps_d)) {
        return std::unexpected(DomainError{
            fn,
            "expected ode_event_detect(\"f\", \"event\", t0, y0, t_end, steps) "
            "with env {t, y} for both formulas"});
    }
    const int steps_i = static_cast<int>(steps_d);
    if (steps_i < 0 || steps_d != steps_i) {
        return std::unexpected(DomainError{fn, "expected non-negative integer steps"});
    }
    SymExpr parsed = std::move(*expr);
    SymExpr event_parsed = std::move(*event_expr);
    auto expr_ptr = std::make_shared<SymExpr>(std::move(parsed));
    auto event_ptr = std::make_shared<SymExpr>(std::move(event_parsed));
    OdeFunc f = [expr_ptr](double t, double y) {
        return sym_eval(*expr_ptr, {{"t", t}, {"y", y}});
    };
    OdeFunc event_g = [event_ptr](double t, double y) {
        return sym_eval(*event_ptr, {{"t", t}, {"y", y}});
    };
    return format_ode_event_trajectory(
        ode_event_detect(f, event_g, t0, y0, t_end, static_cast<size_t>(steps_i)));
}

std::optional<Result<std::string>> try_eval_sym_command(const std::string& cmd) {
    const auto open = cmd.find('(');
    if (open == std::string::npos || cmd.empty() || cmd.back() != ')') {
        return std::nullopt;
    }
    const std::string fn = lower(trim_copy(cmd.substr(0, open)));
    if (fn != "sym_diff" && fn != "sym_simplify" && fn != "sym_integrate" && fn != "sym_eval" &&
        fn != "sym_expand" && fn != "sym_collect" && fn != "sym_substitute" && fn != "sym_limit" &&
        fn != "sym_series" && fn != "sym_solve_linear" && fn != "sym_laplace" &&
        fn != "sym_ilaplace" && fn != "sym_mellin" && fn != "sym_imellin" &&
        fn != "sym_hankel" && fn != "sym_ihankel" &&
        fn != "sym_fourier" && fn != "sym_ifourier" &&
        fn != "sym_ztransform" && fn != "sym_iztransform" && fn != "sym_dsolve") {
        return std::nullopt;
    }
    const auto args = split_call_args(cmd);
    if (!args) {
        return std::unexpected(DomainError{fn, "invalid call syntax"});
    }
    if (fn == "sym_simplify" || fn == "sym_expand") {
        if (args->size() != 1) {
            return std::unexpected(DomainError{fn, std::string("expected ") + fn + "(\"expr\")"});
        }
        if (fn == "sym_simplify") {
            return eval_sym_simplify_string(args->at(0));
        }
        return eval_sym_expand_string(args->at(0));
    }
    if (fn == "sym_diff" || fn == "sym_integrate" || fn == "sym_eval" || fn == "sym_collect" ||
        fn == "sym_solve_linear") {
        if (args->size() != 2) {
            return std::unexpected(
                DomainError{fn, std::string("expected ") + fn + "(\"expr\", \"arg\")"});
        }
        if (fn == "sym_diff") {
            return eval_sym_diff_strings(args->at(0), args->at(1));
        }
        if (fn == "sym_integrate") {
            return eval_sym_integrate_strings(args->at(0), args->at(1));
        }
        if (fn == "sym_eval") {
            return eval_sym_eval_strings(args->at(0), args->at(1));
        }
        if (fn == "sym_collect") {
            return eval_sym_collect_strings(args->at(0), args->at(1));
        }
        return eval_sym_solve_linear_strings(args->at(0), args->at(1));
    }
    if (fn == "sym_laplace" || fn == "sym_ilaplace" || fn == "sym_mellin" || fn == "sym_imellin" ||
        fn == "sym_hankel" || fn == "sym_ihankel" ||
        fn == "sym_fourier" || fn == "sym_ifourier" ||
        fn == "sym_ztransform" || fn == "sym_iztransform" || fn == "sym_dsolve") {
        if (args->size() != 3) {
            return std::unexpected(
                DomainError{fn, std::string("expected ") + fn + "(\"expr\", \"var1\", \"var2\")"});
        }
        if (fn == "sym_laplace") {
            return eval_sym_transform_strings(args->at(0), args->at(1), args->at(2), fn.c_str(), sym_laplace);
        }
        if (fn == "sym_ilaplace") {
            return eval_sym_transform_strings(args->at(0), args->at(1), args->at(2), fn.c_str(), sym_ilaplace);
        }
        if (fn == "sym_mellin") {
            return eval_sym_transform_strings(args->at(0), args->at(1), args->at(2), fn.c_str(), sym_mellin);
        }
        if (fn == "sym_imellin") {
            return eval_sym_transform_strings(args->at(0), args->at(1), args->at(2), fn.c_str(), sym_imellin);
        }
        if (fn == "sym_hankel") {
            return eval_sym_transform_strings(args->at(0), args->at(1), args->at(2), fn.c_str(), sym_hankel);
        }
        if (fn == "sym_ihankel") {
            return eval_sym_transform_strings(args->at(0), args->at(1), args->at(2), fn.c_str(), sym_ihankel);
        }
        if (fn == "sym_fourier") {
            return eval_sym_transform_strings(args->at(0), args->at(1), args->at(2), fn.c_str(), sym_fourier);
        }
        if (fn == "sym_ifourier") {
            return eval_sym_transform_strings(args->at(0), args->at(1), args->at(2), fn.c_str(), sym_ifourier);
        }
        if (fn == "sym_ztransform") {
            return eval_sym_transform_strings(args->at(0), args->at(1), args->at(2), fn.c_str(), sym_ztransform);
        }
        if (fn == "sym_iztransform") {
            return eval_sym_transform_strings(args->at(0), args->at(1), args->at(2), fn.c_str(), sym_iztransform);
        }
        return eval_sym_transform_strings(args->at(0), args->at(1), args->at(2), fn.c_str(), sym_dsolve);
    }
    if (fn == "sym_substitute" || fn == "sym_limit") {
        if (args->size() != 3) {
            return std::unexpected(
                DomainError{fn, std::string("expected ") + fn + "(\"expr\", \"var\", arg)"});
        }
        if (fn == "sym_substitute") {
            return eval_sym_substitute_strings(args->at(0), args->at(1), args->at(2));
        }
        return eval_sym_limit_strings(args->at(0), args->at(1), args->at(2));
    }
    if (fn == "sym_series") {
        if (args->size() != 4) {
            return std::unexpected(DomainError{
                fn, "expected sym_series(\"expr\", \"var\", point, order)"});
        }
        return eval_sym_series_strings(args->at(0), args->at(1), args->at(2), args->at(3));
    }
    return std::nullopt;
}

std::optional<Result<std::string>> try_eval_crypto_command(const std::string& cmd) {
    const auto open_paren = cmd.find('(');
    if (open_paren == std::string::npos || cmd.empty() || cmd.back() != ')') {
        return std::nullopt;
    }
    const std::string fn = lower(trim_copy(cmd.substr(0, open_paren)));
    const auto call_args = split_call_args(cmd);
    if (!call_args) {
        return std::nullopt;
    }
    if (fn == "crypto_aes128_encrypt_block") {
        if (call_args->size() != 2) {
            return std::unexpected(DomainError{
                fn, "expected crypto_aes128_encrypt_block(key_hex, block_hex)"});
        }
        return eval_crypto_aes128_encrypt_block(call_args->at(0), call_args->at(1));
    }
    if (fn == "crypto_aes128_decrypt_block") {
        if (call_args->size() != 2) {
            return std::unexpected(DomainError{
                fn, "expected crypto_aes128_decrypt_block(key_hex, block_hex)"});
        }
        return eval_crypto_aes128_decrypt_block(call_args->at(0), call_args->at(1));
    }
    if (fn == "crypto_aes256_encrypt_block") {
        if (call_args->size() != 2) {
            return std::unexpected(DomainError{
                fn, "expected crypto_aes256_encrypt_block(key_hex, block_hex)"});
        }
        return eval_crypto_aes256_encrypt_block(call_args->at(0), call_args->at(1));
    }
    if (fn == "crypto_aes256_decrypt_block") {
        if (call_args->size() != 2) {
            return std::unexpected(DomainError{
                fn, "expected crypto_aes256_decrypt_block(key_hex, block_hex)"});
        }
        return eval_crypto_aes256_decrypt_block(call_args->at(0), call_args->at(1));
    }
    if (fn == "crypto_aes128_cbc_encrypt") {
        if (call_args->size() != 3) {
            return std::unexpected(DomainError{
                fn, "expected crypto_aes128_cbc_encrypt(key_hex, iv_hex, plaintext_hex)"});
        }
        return eval_crypto_aes128_cbc_encrypt(call_args->at(0), call_args->at(1),
                                              call_args->at(2));
    }
    if (fn == "crypto_aes128_cbc_decrypt") {
        if (call_args->size() != 3) {
            return std::unexpected(DomainError{
                fn, "expected crypto_aes128_cbc_decrypt(key_hex, iv_hex, ciphertext_hex)"});
        }
        return eval_crypto_aes128_cbc_decrypt(call_args->at(0), call_args->at(1),
                                              call_args->at(2));
    }
    if (fn == "crypto_aes256_cbc_encrypt") {
        if (call_args->size() != 3) {
            return std::unexpected(DomainError{
                fn, "expected crypto_aes256_cbc_encrypt(key_hex, iv_hex, plaintext_hex)"});
        }
        return eval_crypto_aes256_cbc_encrypt(call_args->at(0), call_args->at(1),
                                              call_args->at(2));
    }
    if (fn == "crypto_aes256_cbc_decrypt") {
        if (call_args->size() != 3) {
            return std::unexpected(DomainError{
                fn, "expected crypto_aes256_cbc_decrypt(key_hex, iv_hex, ciphertext_hex)"});
        }
        return eval_crypto_aes256_cbc_decrypt(call_args->at(0), call_args->at(1),
                                              call_args->at(2));
    }
    if (fn == "crypto_aes128_gcm_encrypt") {
        if (call_args->size() != 4) {
            return std::unexpected(DomainError{
                fn, "expected crypto_aes128_gcm_encrypt(key_hex, iv_hex, aad_hex, plaintext_hex)"});
        }
        return eval_crypto_aes128_gcm_encrypt(call_args->at(0), call_args->at(1),
                                              call_args->at(2), call_args->at(3));
    }
    if (fn == "crypto_aes128_gcm_decrypt") {
        if (call_args->size() != 5) {
            return std::unexpected(DomainError{
                fn,
                "expected crypto_aes128_gcm_decrypt(key_hex, iv_hex, aad_hex, ciphertext_hex, "
                "tag_hex)"});
        }
        return eval_crypto_aes128_gcm_decrypt(call_args->at(0), call_args->at(1),
                                              call_args->at(2), call_args->at(3),
                                              call_args->at(4));
    }
    if (fn == "crypto_aes256_gcm_encrypt") {
        if (call_args->size() != 4) {
            return std::unexpected(DomainError{
                fn, "expected crypto_aes256_gcm_encrypt(key_hex, iv_hex, aad_hex, plaintext_hex)"});
        }
        return eval_crypto_aes256_gcm_encrypt(call_args->at(0), call_args->at(1),
                                              call_args->at(2), call_args->at(3));
    }
    if (fn == "crypto_aes256_gcm_decrypt") {
        if (call_args->size() != 5) {
            return std::unexpected(DomainError{
                fn,
                "expected crypto_aes256_gcm_decrypt(key_hex, iv_hex, aad_hex, ciphertext_hex, "
                "tag_hex)"});
        }
        return eval_crypto_aes256_gcm_decrypt(call_args->at(0), call_args->at(1),
                                              call_args->at(2), call_args->at(3),
                                              call_args->at(4));
    }
    if (fn == "crypto_chacha20") {
        if (call_args->size() != 4) {
            return std::unexpected(DomainError{
                fn, "expected crypto_chacha20(key_hex, nonce_hex, counter, data_hex)"});
        }
        return eval_crypto_chacha20(call_args->at(0), call_args->at(1), call_args->at(2),
                                    call_args->at(3));
    }
    if (fn == "crypto_chacha20_poly1305_encrypt") {
        if (call_args->size() != 4) {
            return std::unexpected(DomainError{
                fn,
                "expected crypto_chacha20_poly1305_encrypt(key_hex, nonce_hex, aad_hex, "
                "plaintext_hex)"});
        }
        return eval_crypto_chacha20_poly1305_encrypt(call_args->at(0), call_args->at(1),
                                                     call_args->at(2), call_args->at(3));
    }
    if (fn == "crypto_chacha20_poly1305_decrypt") {
        if (call_args->size() != 5) {
            return std::unexpected(DomainError{
                fn,
                "expected crypto_chacha20_poly1305_decrypt(key_hex, nonce_hex, aad_hex, "
                "ciphertext_hex, tag_hex)"});
        }
        return eval_crypto_chacha20_poly1305_decrypt(call_args->at(0), call_args->at(1),
                                                     call_args->at(2), call_args->at(3),
                                                     call_args->at(4));
    }
    if (fn == "crypto_x25519_keypair") {
        if (call_args->size() != 1) {
            return std::unexpected(DomainError{
                fn, "expected crypto_x25519_keypair(hex_priv) -> hex public key"});
        }
        return eval_crypto_x25519_keypair(call_args->at(0));
    }
    if (fn == "crypto_x25519_shared") {
        if (call_args->size() != 2) {
            return std::unexpected(DomainError{
                fn, "expected crypto_x25519_shared(hex_priv, hex_pub)"});
        }
        return eval_crypto_x25519_shared(call_args->at(0), call_args->at(1));
    }
    if (fn == "crypto_ed25519_keypair") {
        if (call_args->size() != 1) {
            return std::unexpected(DomainError{
                fn, "expected crypto_ed25519_keypair(hex_seed) -> hex public key"});
        }
        return eval_crypto_ed25519_keypair(call_args->at(0));
    }
    if (fn == "crypto_ed25519_sign") {
        if (call_args->size() != 2) {
            return std::unexpected(DomainError{
                fn, "expected crypto_ed25519_sign(hex_seed_or_sk, hex_msg)"});
        }
        return eval_crypto_ed25519_sign(call_args->at(0), call_args->at(1));
    }
    if (fn == "crypto_ed25519_verify") {
        if (call_args->size() != 3) {
            return std::unexpected(DomainError{
                fn, "expected crypto_ed25519_verify(hex_pub, hex_msg, hex_sig)"});
        }
        return eval_crypto_ed25519_verify(call_args->at(0), call_args->at(1),
                                          call_args->at(2));
    }
    if (fn == "crypto_constant_time_eq") {
        if (call_args->size() != 2) {
            return std::unexpected(DomainError{
                fn, "expected crypto_constant_time_eq(hex_a, hex_b)"});
        }
        return eval_crypto_constant_time_eq(call_args->at(0), call_args->at(1));
    }
    if (fn == "crypto_random_bytes") {
        if (call_args->size() != 1) {
            return std::unexpected(DomainError{fn, "expected crypto_random_bytes(n)"});
        }
        return eval_crypto_random_bytes(call_args->at(0));
    }
    if (fn == "crypto_hkdf_sha256") {
        if (call_args->size() != 4) {
            return std::unexpected(DomainError{
                fn, "expected crypto_hkdf_sha256(hex_ikm, hex_salt, hex_info, len)"});
        }
        return eval_crypto_hkdf_sha256(call_args->at(0), call_args->at(1),
                                       call_args->at(2), call_args->at(3));
    }
    if (fn == "crypto_hkdf_sha512") {
        if (call_args->size() != 4) {
            return std::unexpected(DomainError{
                fn, "expected crypto_hkdf_sha512(hex_ikm, hex_salt, hex_info, len)"});
        }
        return eval_crypto_hkdf_sha512(call_args->at(0), call_args->at(1),
                                       call_args->at(2), call_args->at(3));
    }
    if (fn == "crypto_sha256") {
        if (call_args->size() != 1) {
            return std::unexpected(DomainError{fn, "expected crypto_sha256(hex_data)"});
        }
        return eval_crypto_sha256(call_args->at(0));
    }
    if (fn == "crypto_to_hex") {
        if (call_args->size() != 1) {
            return std::unexpected(DomainError{fn, "expected crypto_to_hex(hex_data)"});
        }
        return eval_crypto_to_hex(call_args->at(0));
    }
    if (fn == "crypto_sha512") {
        if (call_args->size() != 1) {
            return std::unexpected(DomainError{fn, "expected crypto_sha512(hex_data)"});
        }
        return eval_crypto_sha512(call_args->at(0));
    }
    if (fn == "crypto_hmac_sha256") {
        if (call_args->size() != 2) {
            return std::unexpected(DomainError{
                fn, "expected crypto_hmac_sha256(hex_key, hex_data)"});
        }
        return eval_crypto_hmac_sha256(call_args->at(0), call_args->at(1));
    }
    if (fn == "crypto_hmac_sha512") {
        if (call_args->size() != 2) {
            return std::unexpected(DomainError{
                fn, "expected crypto_hmac_sha512(hex_key, hex_data)"});
        }
        return eval_crypto_hmac_sha512(call_args->at(0), call_args->at(1));
    }
    if (fn == "crypto_pbkdf2_sha256") {
        if (call_args->size() != 4) {
            return std::unexpected(DomainError{
                fn, "expected crypto_pbkdf2_sha256(hex_pass, hex_salt, iter, dklen)"});
        }
        return eval_crypto_pbkdf2_sha256(call_args->at(0), call_args->at(1),
                                         call_args->at(2), call_args->at(3));
    }
    if (fn == "crypto_pbkdf2_hmac_sha512") {
        if (call_args->size() != 4) {
            return std::unexpected(DomainError{
                fn, "expected crypto_pbkdf2_hmac_sha512(hex_pass, hex_salt, iter, dklen)"});
        }
        return eval_crypto_pbkdf2_hmac_sha512(call_args->at(0), call_args->at(1),
                                              call_args->at(2), call_args->at(3));
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

bool is_identifier_view(std::string_view text) {
    text = trim_view(text);
    if (text.empty()) {
        return false;
    }
    const unsigned char first = static_cast<unsigned char>(text.front());
    if (!std::isalpha(first) && text.front() != '_') {
        return false;
    }
    for (char c : text) {
        const unsigned char uc = static_cast<unsigned char>(c);
        if (!std::isalnum(uc) && c != '_') {
            return false;
        }
    }
    return true;
}

bool parse_scalar_operand_view(std::string_view text, ScalarOperand& out) {
    text = trim_view(text);
    double value = 0.0;
    if (parse_number_view(text, value)) {
        out.is_literal = true;
        out.literal = value;
        out.name.clear();
        return true;
    }
    if (is_identifier_view(text)) {
        out.is_literal = false;
        out.literal = 0.0;
        out.name.assign(text.begin(), text.end());
        return true;
    }
    return false;
}

std::optional<std::pair<std::string_view, std::string_view>> parse_scalar_unary_call_view(
    std::string_view expr) {
    expr = trim_view(expr);
    const auto open = expr.find('(');
    if (open == std::string_view::npos || open == 0) {
        return std::nullopt;
    }
    const std::string_view name = trim_view(expr.substr(0, open));
    if (!is_identifier_view(name)) {
        return std::nullopt;
    }

    int depth = 0;
    size_t close = std::string_view::npos;
    for (size_t i = open; i < expr.size(); ++i) {
        if (expr[i] == '(') {
            ++depth;
        } else if (expr[i] == ')') {
            --depth;
            if (depth == 0) {
                close = i;
                break;
            }
        }
    }
    if (close == std::string_view::npos || close + 1 != expr.size()) {
        return std::nullopt;
    }

    const std::string_view arg = trim_view(expr.substr(open + 1, close - open - 1));
    if (arg.empty()) {
        return std::nullopt;
    }
    return std::pair{name, arg};
}

size_t split_scalar_call_args_view(std::string_view args_text, std::string_view* out, size_t cap) {
    size_t count = 0;
    size_t start = 0;
    int paren_depth = 0;
    int bracket_depth = 0;
    for (size_t i = 0; i <= args_text.size(); ++i) {
        const char c = i < args_text.size() ? args_text[i] : ',';
        if (i < args_text.size()) {
            if (c == '(') {
                ++paren_depth;
            } else if (c == ')') {
                --paren_depth;
            } else if (c == '[') {
                ++bracket_depth;
            } else if (c == ']') {
                --bracket_depth;
            }
        }
        if ((c == ',' && paren_depth == 0 && bracket_depth == 0) || i == args_text.size()) {
            if (count < cap) {
                out[count] = trim_view(args_text.substr(start, i - start));
            }
            ++count;
            start = i + 1;
        }
    }
    return count;
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
            fn == "cuda_add" ||
            fn == "solve" || fn == "lsq" || fn == "bicgstab" || fn == "cg" || fn == "gmres" ||
            fn == "jacobi" ||
            fn == "qmr" || fn == "lsqr" ||
            fn == "tfqmr" || fn == "lsmr" ||
            fn == "dist_solve" || fn == "dist_cg" || fn == "dist_gmres" || fn == "dist_jacobi" || fn == "dist_bicgstab" || fn == "dist_minres" || fn == "dist_qmr" || fn == "dist_tfqmr" || fn == "dist_lsmr" || fn == "dist_lsqr" || fn == "dist_matmul" || fn == "transpose" || fn == "chol" ||
            fn == "det" ||
            fn == "trace" || fn == "norm" || fn == "rank" || fn == "matrix_rank" ||
            fn == "cond" || fn == "lu" ||
            fn == "cuda_lu" ||
            fn == "qr" || fn == "svd" || fn == "eig_sym" || fn == "eig" || fn == "ldl" ||
            fn == "hess" || fn == "schur" ||
            fn == "bidiag" || fn == "solve_sylvester" || fn == "minres" ||
            fn == "zeros" || fn == "eye" || fn == "ones" ||
            fn == "rand" || fn == "randn" || fn == "expm" || fn == "sqrtm" ||
            fn == "logm" || fn == "tril" || fn == "triu" || fn == "diag" || fn == "cosm" ||
            fn == "sinm" || fn == "funm" || fn == "precond_diag" || fn == "precond_ssor" ||
            fn == "inv" ||
            fn == "pinv" || fn == "null" || fn == "orth" ||
            fn == "kron" || fn == "repmat" || fn == "linspace" ||
            fn == "rgb2gray" || fn == "gray2rgb" || fn == "rgb2hsv" || fn == "hsv2rgb" ||
            fn == "sobel" || fn == "sobel_x" || fn == "sobel_y" || fn == "prewitt" ||
            fn == "scharr" || fn == "roberts" ||
            fn == "imfilter" || fn == "dft_magnitude" || fn == "laplacian_of_gaussian" ||
            fn == "imgaussfilt" || fn == "medfilt2" || fn == "boxfilter" ||
            fn == "imdilate" || fn == "imerode" || fn == "imopen" || fn == "imclose" ||
            fn == "imtophat" || fn == "imbothat" || fn == "imgradient_morph" ||
            fn == "imadjust" || fn == "imhist" ||
            fn == "bilateral" || fn == "canny" ||
            fn == "laplacian" || fn == "histeq" ||
            fn == "sharpen" || fn == "threshold_otsu" || fn == "imresize" ||
            fn == "imflip" || fn == "imrotate90" || fn == "threshold_binary" ||
            fn == "adapthisteq" || fn == "impad" || fn == "radon" || fn == "iradon" ||
            fn == "label_components" || fn == "watershed" || fn == "slic" ||
            fn == "hough_lines" || fn == "hough_circles" || fn == "harris" ||
            fn == "shi_tomasi" ||
            fn == "rle_encode_vec" || fn == "rle_decode_vec" ||
            fn == "mtf_encode_vec" || fn == "mtf_decode_vec" ||
            fn == "lzw_encode_vec" || fn == "lzw_decode_vec" ||
            fn == "huffman_encode_vec" || fn == "huffman_decode_vec" ||
            fn == "arithmetic_encode_vec" || fn == "arithmetic_decode_vec" ||
            fn == "ans_encode_vec" || fn == "ans_decode_vec" ||
            fn == "golomb_rice_encode_vec" || fn == "golomb_rice_decode_vec" ||
            fn == "wavelet_compress_vec" || fn == "wavelet_decompress_vec" ||
            fn == "lz77_encode_vec" || fn == "lz77_decode_vec" ||
            fn == "bzip2_compress_vec" || fn == "bzip2_decompress_vec" ||
            fn == "bwt_encode_vec" || fn == "bwt_decode_vec" ||
            fn == "delta_encode_vec" || fn == "delta_decode_vec" ||
            fn == "control_is_controllable" || fn == "control_is_observable" ||
            fn == "control_lyap" || fn == "control_lqr" || fn == "control_lqe" ||
            fn == "control_place" ||
            fn == "control_dlyap" || fn == "control_riccati" || fn == "control_dare" ||
            fn == "control_ctrb" || fn == "control_obsv" ||
            fn == "control_ctrb_gram" || fn == "control_obsv_gram" ||
            fn == "control_bode_mag_db" || fn == "control_bode_phase" ||
            fn == "control_bode" || fn == "control_poles" || fn == "control_zeros" ||
            fn == "control_step_info" || fn == "control_nyquist" ||
            fn == "control_step_response" || fn == "control_impulse_response" ||
            fn == "control_kalman_predict" || fn == "control_kalman_update" ||
            fn == "control_kalman_predict_cov" || fn == "control_kalman_update_cov" ||
            fn == "control_tf2ss" || fn == "control_c2d" || fn == "control_c2d_b" ||
            fn == "control_c2d_tustin" || fn == "control_c2d_euler" ||
            fn == "control_ss2tf" || fn == "control_d2c" ||
            fn == "control_d2c_tustin" || fn == "control_d2c_euler" ||
            fn == "control_series" || fn == "control_parallel" || fn == "control_feedback" ||
            fn == "control_c2d_tf" || fn == "control_d2c_tf" ||
            fn == "control_c2d_tf_tustin" || fn == "control_d2c_tf_tustin" ||
            fn == "topo_bottleneck_distance" ||
            fn == "topo_wasserstein_distance" ||
            fn == "topo_persistence_diagram" ||
            fn == "quantum_op_apply" ||
            fn == "graph_diameter" ||
            fn == "graph_radius" ||
            fn == "graph_algebraic_connectivity" ||
            fn == "graph_betweenness" ||
            fn == "graph_closeness" ||
            fn == "graph_degree_centrality" ||
            fn == "graph_louvain" || fn == "graph_eigenvector_centrality" ||
            fn == "graph_katz_centrality" || fn == "graph_adjacency_spectrum" ||
            fn == "graph_laplacian" ||
            fn == "graph_normalised_laplacian" ||
            fn == "graph_eccentricity" ||
            fn == "graph_articulation_points" ||
            fn == "graph_biconnected_components" || fn == "graph_eulerian_path" ||
            fn == "graph_hamiltonian_path" || fn == "graph_tsp_heuristic" ||
            fn == "graph_bridges" || fn == "graph_maximum_matching" ||
            fn == "graph_transitive_closure" ||
            fn == "finance_min_variance_portfolio" ||
            fn == "graph_is_bipartite" ||
            fn == "graph_is_dag" || fn == "graph_is_connected" ||
            fn == "graph_is_strongly_connected" ||
            fn == "graph_is_tree" || fn == "graph_is_planar" ||
            fn == "graph_topological_sort" ||
            fn == "graph_greedy_colour" || fn == "graph_k_core_decomposition" ||
            fn == "graph_euler_circuit" ||
            fn == "graph_chromatic_number" ||
            fn == "graph_scc" || fn == "graph_connected_components" ||
            fn == "count_components" ||
            fn == "poly_discriminant" ||
            fn == "stats_mean" || fn == "stats_median" ||
            fn == "stats_stddev" || fn == "stats_skewness" ||
            fn == "stats_kurtosis" || fn == "stats_var" || fn == "stats_mode" ||
            fn == "stats_geometric_mean" || fn == "stats_harmonic_mean" ||
            fn == "stats_rms" || fn == "stats_mad" || fn == "stats_iqr" ||
            fn == "stats_min_value" || fn == "stats_max_value" ||
            fn == "count_components" ||
            fn == "fft_rfft" || fn == "fft_dft" || fn == "fft_goertzel" || fn == "geo_delaunay_2d" ||
            fn == "geo_voronoi" || fn == "geo_convex_hull" ||
            fn == "geo_upper_hull" || fn == "geo_lower_hull" ||
            fn == "geo_triangulate_polygon" || fn == "geo_convex_hull_3d" ||
            fn == "geo_min_bounding_rect" ||
            fn == "geo_kdtree_knn" || fn == "geo_kdtree_range" ||
            fn == "geo_kdtree_3d_knn" || fn == "geo_kdtree_3d_range" ||
            fn == "geo_bezier_subdivide" ||
            fn == "topo_pairwise_distances" ||
            fn == "combo_next_perm" || fn == "combo_prev_perm" || fn == "numthy_convergents" ||
            fn == "numthy_factor_exp" || fn == "numthy_farey" || fn == "numthy_lucas_sequence" ||
            fn == "sph_harm" ||
            fn == "numthy_stern_brocot" || fn == "numthy_pell_solve" ||
            fn == "numthy_quadratic_residues" ||
            fn == "ml_mat_transpose" || fn == "ml_mat_mul" ||
            fn == "ml_linear_fit" || fn == "ml_linear_predict" ||
            fn == "ml_ridge_fit" || fn == "ml_ridge_predict" ||
            fn == "ml_logistic_fit" || fn == "ml_logistic_predict" ||
            fn == "ml_lasso_fit" || fn == "ml_lasso_predict" ||
            fn == "ml_elastic_net_fit" || fn == "ml_elastic_net_predict" ||
            fn == "ml_knn_fit" || fn == "ml_knn_predict" ||
            fn == "ml_naive_bayes_fit" || fn == "ml_naive_bayes_predict" ||
            fn == "ml_lda_fit" || fn == "ml_lda_predict" || fn == "ml_lda_transform" ||
            fn == "ml_qda_fit" || fn == "ml_qda_predict" ||
            fn == "ml_svm_fit" || fn == "ml_svm_predict" ||
            fn == "ml_pca_fit" || fn == "ml_pca_transform" || fn == "ml_pca_fit_transform" || fn == "ml_kmeans_fit" || fn == "ml_kmeans_predict" ||
            fn == "ml_decision_tree_fit" || fn == "ml_decision_tree_predict" || fn == "ml_random_forest_fit" || fn == "ml_random_forest_predict" || fn == "ml_adaboost_fit" || fn == "ml_adaboost_predict" || fn == "ml_gradient_boosting_fit" || fn == "ml_gradient_boosting_predict" ||
            fn == "ml_gmm_fit" || fn == "ml_gmm_predict" || fn == "ml_gmm_predict_proba" || fn == "ml_dbscan_fit" || fn == "ml_spectral_clustering" ||
            fn == "ml_isolation_forest_fit" || fn == "ml_isolation_forest_score" || fn == "ml_agglomerative_fit" || fn == "ml_tsne_fit" ||
            fn == "ml_standard_scaler_fit" || fn == "ml_standard_scaler_transform" || fn == "ml_minmax_scaler_fit" || fn == "ml_minmax_scaler_transform" ||
            fn == "ml_confusion_matrix" || fn == "ml_roc_curve" || fn == "ml_precision_recall_curve" || fn == "poly_deriv" ||
            fn == "poly_eval" || fn == "poly_cheb_eval" || fn == "poly_cheb_expand" ||
            fn == "poly_integ" || fn == "poly_add" ||
            fn == "poly_lagrange" || fn == "poly_interp_newton" ||
            fn == "poly_roots" || fn == "poly_fit" || fn == "poly_interp_hermite" ||
            fn == "poly_gcd" || fn == "poly_squarefree" ||
            fn == "poly_factor" || fn == "poly_rational_roots" ||
            fn == "poly_factor_rational" || fn == "poly_partial_fractions" ||
            fn == "poly_monic" || fn == "poly_reverse" ||
            fn == "poly_shift" || fn == "poly_scale" || fn == "poly_pow" ||
            fn == "poly_lcm" || fn == "poly_div_quot" || fn == "poly_mod" ||
            fn == "poly_eval_at" || fn == "poly_sylvester" ||
            fn == "poly_mul" || fn == "poly_sub" || fn == "poly_compose" ||
            fn == "geo_poly_union" || fn == "geo_poly_intersect" || fn == "geo_poly_diff" ||
            fn == "geo_minkowski_sum" || fn == "geo_clip_polygon" ||
            fn == "fft_irfft" || fn == "fft_ifft" || fn == "fft_fft2" || fn == "fft_dct2" || fn == "fft_idct2" || fn == "fft_dst2" || fn == "ifft2" || fn == "idst2" || fn == "kruskal_wallis" || fn == "stats_shapiro_wilk" || fn == "stats_one_way_anova" || fn == "stats_mann_whitney_u" || fn == "stats_wilcoxon_signed_rank" || fn == "stats_friedman" || fn == "stats_ks_2sample" || fn == "stats_jarque_bera" || fn == "stats_ljung_box" || fn == "fftshift" || fn == "ifftshift" || fn == "fftfreq" || fn == "rfftfreq" ||
            fn == "fft_irfft" || fn == "fft_ifft" || fn == "fft_fft2" || fn == "fft_dct2" || fn == "fft_idct2" || fn == "fft_dst2" || fn == "ifft2" || fn == "idst2" || fn == "kruskal_wallis" || fn == "stats_shapiro_wilk" || fn == "stats_one_way_anova" || fn == "stats_levene" || fn == "stats_bartlett" || fn == "stats_fligner" || fn == "stats_mann_whitney_u" || fn == "stats_wilcoxon_signed_rank" || fn == "fftshift" || fn == "ifftshift" || fn == "fftfreq" || fn == "rfftfreq" ||
            fn == "graph_floyd_warshall" || fn == "graph_mst_kruskal" ||
            fn == "graph_mst_prim" || fn == "graph_min_arborescence" ||
            fn == "graph_scc" || fn == "graph_connected_components" ||
            fn == "info_channel_capacity_input" ||
            fn == "signal_convolve" || fn == "signal_correlate" ||
            fn == "signal_xcorr" || fn == "signal_xcov" || fn == "signal_autocorr" ||
            fn == "signal_lms" || fn == "signal_lms_weights" ||
            fn == "signal_sosfilt" || fn == "signal_conv2" ||
            fn == "signal_deconv" ||
            fn == "signal_filtfilt" || fn == "signal_filter" ||
            fn == "graph_max_flow" || fn == "graph_min_cut" ||
            fn == "diffgeo_surface_normal_sphere" ||
            fn == "poly_resultant" ||
            fn == "stats_correlation" || fn == "stats_spearman" ||
            fn == "stats_kendall" || fn == "stats_weighted_mean" ||
            fn == "stats_weighted_variance" ||
            fn == "stats_two_sample_ttest" ||
            fn == "stats_chi2_gof" || fn == "graph_is_isomorphic" ||
            fn == "graph_modularity" ||
            fn == "signal_moving_average" || fn == "signal_upsample" ||
            fn == "signal_downsample" || fn == "signal_decimate" ||
            fn == "signal_interpolate" || fn == "signal_resample" ||
            fn == "signal_savgol" || fn == "signal_median_filter" ||
            fn == "graph_bfs" || fn == "graph_dfs" ||
            fn == "graph_bipartite_match" || fn == "graph_k_core_subgraph" ||
            fn == "stats_percentile" || fn == "graph_dfs" ||
            fn == "finance_historical_var" || fn == "finance_historical_cvar" ||
            fn == "stats_percentile" || fn == "stats_ttest" || fn == "stats_trimmed_mean" ||
            fn == "stats_vif" || fn == "stats_variance_inflation_factor" ||
            fn == "stats_acf" ||
            fn == "stats_linear_regression" || fn == "stats_pacf" ||
            fn == "stats_arfit" || fn == "stats_multiple_regression" ||
            fn == "stats_kde" ||
            fn == "stats_bootstrap_ci" || fn == "stats_bootstrap_mean" ||
            fn == "stats_partial_correlation" || fn == "stats_weighted_correlation" ||
            fn == "signal_lowpass" || fn == "signal_butterworth" || fn == "signal_highpass" ||
            fn == "signal_bandpass" || fn == "signal_cheby1" || fn == "signal_cheby2" ||
            fn == "signal_firwin" || fn == "signal_firwin_highpass" ||
            fn == "signal_periodogram" || fn == "signal_welch_psd" ||
            fn == "signal_coherence" ||
            fn == "signal_spectrogram" ||
            fn == "signal_envelope" || fn == "signal_hilbert" ||
            fn == "signal_czt" || fn == "signal_czt_zoom" ||
            fn == "signal_instantaneous_phase" || fn == "signal_instantaneous_freq" ||
            fn == "signal_unwrap" ||
            fn == "quantum_commutator" || fn == "quantum_tensor_product" ||
            fn == "quantum_ket_tensor_product" || fn == "quantum_outer" ||
            fn == "quantum_dagger" || fn == "quantum_matmul_dm" ||
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
            fn == "combo_gray_code" ||
            fn == "combo_dyck_paths" ||
            fn == "combo_necklaces" || fn == "combo_bracelets" ||
            fn == "combo_lyndon_words" || fn == "combo_de_bruijn_sequence" ||
            fn == "combo_motzkin_paths" || fn == "combo_set_partitions" ||
            fn == "combo_restricted_partitions" ||
            fn == "numthy_crt" || fn == "geo_centroid_x" || fn == "geo_centroid_y" ||
            fn == "bwt_primary_index" ||
            fn == "quantum_ket_superposition" || fn == "quantum_ghz_state" ||
            fn == "quantum_w_state" || fn == "quantum_bell_state" || fn == "quantum_coherent_state" ||
            fn == "numthy_divisors_vec" ||
            fn == "numthy_divisors" ||
            fn == "numthy_factor_vec" ||
            fn == "numthy_factor" ||
            fn == "combo_unrank_permutation" || fn == "combo_unrank_combination" ||
            fn == "ml_accuracy" || fn == "ml_rmse" || fn == "ml_mse" ||
            fn == "ml_r2" || fn == "ml_f1" || fn == "ml_precision" ||
            fn == "ml_recall" || fn == "ml_mae" || fn == "ml_huber" ||
            fn == "ml_hinge" || fn == "ml_binary_crossentropy" ||
            fn == "ml_categorical_crossentropy" || fn == "ml_vec_dot" ||
            fn == "bigint" || fn == "bigint_factorial" || fn == "bigint_fib" ||
            fn == "bigint_gcd" ||
            fn == "sym_diff" || fn == "sym_integrate" || fn == "sym_eval" || fn == "sym_simplify" ||
            fn == "sym_expand" || fn == "sym_collect" || fn == "sym_substitute" ||
            fn == "sym_limit" || fn == "sym_series" || fn == "sym_solve_linear" ||
            fn == "sym_laplace" || fn == "sym_ilaplace" || fn == "sym_mellin" || fn == "sym_imellin" ||
            fn == "sym_hankel" || fn == "sym_ihankel" ||
            fn == "sym_fourier" ||
            fn == "sym_ifourier" || fn == "sym_ztransform" || fn == "sym_iztransform" ||
            fn == "sym_dsolve" ||
            fn == "graph_pagerank" || fn == "graph_dijkstra_dist" ||
            fn == "graph_bellman_ford_dist" || fn == "graph_max_flow" ||
            fn == "graph_min_cut" ||
            fn == "graph_astar" ||
            fn == "geo_convex_hull_area" || fn == "geo_polygon_area" ||
            fn == "geo_polygon_perimeter" || fn == "geo_signed_area" ||
            fn == "geo_moment_of_inertia" || fn == "geo_point_in_polygon" ||
            fn == "ml_vec_norm" ||
            fn == "quantum_hadamard" ||
            fn == "quantum_ket_normalise" ||
            fn == "quantum_density_matrix" ||
            fn == "quantum_ket_basis" || fn == "quantum_fock_state" ||
            fn == "quantum_grover_search" ||
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
            fn == "finance_portfolio_variance" || fn == "finance_information_ratio" ||
            fn == "finance_treynor" ||
            fn == "info_entropy" || fn == "info_lz_complexity" || fn == "info_mutual_info" ||
            fn == "info_blahut_arimoto" || fn == "info_channel_capacity" ||
            fn == "info_normalized_entropy" ||
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
            fn == "quantum_schmidt_rank" || fn == "quantum_uncertainty" ||
           fn == "quantum_schmidt_number" ||
            fn == "quantum_grover_optimal_iterations" || fn == "quantum_purity" ||
            fn == "quantum_wigner" || fn == "quantum_husimi" ||
            fn == "quantum_partial_trace" || fn == "quantum_schrodinger" ||
            fn == "quantum_schrodinger_final" || fn == "quantum_grover_search" ||
            fn == "pde_heat_1d" || fn == "pde_heat_1d_cn" || fn == "pde_heat_2d" ||
            fn == "pde_heat_2d_cn_adi" || fn == "pde_wave_1d" || fn == "pde_wave_2d" ||
            fn == "pde_advection_1d" || fn == "pde_advection_1d_lax_wendroff" ||
            fn == "pde_reaction_diffusion_1d" || fn == "pde_poisson_2d" ||
            fn == "pde_poisson_1d" || fn == "pde_laplace_2d" || fn == "pde_helmholtz_2d" ||
            fn == "pde_burgers_1d" ||
            fn == "fem_poisson1d" || fn == "fem_poisson2d" || fn == "fem_poisson3d" ||
            fn == "fem_mesh2d_rectangular" || fn == "fem_mesh2d" ||
            fn == "fem_stiffness_2d" || fn == "assemble_stiffness_2d" ||
            fn == "fem_load_2d" || fn == "fem_apply_dirichlet" || fn == "fem_solve" ||
            fn == "cfd_advection1d" ||
            fn == "cfd_advection2d" ||
            fn == "cfd_advection3d" ||
            fn == "cfd_grid2d" ||
            fn == "cfd_square_pulse_2d" ||
            fn == "cfd_upwind_step_2d" ||
            fn == "ode_euler" || fn == "ode_rk4" || fn == "ode_rk2" || fn == "ode_midpoint" ||
            fn == "ode_rk45" || fn == "ode_rk23" || fn == "ode_cashkarp" ||
            fn == "ode_backward_euler" || fn == "ode_trapezoidal" ||
            fn == "ode_rosenbrock23" || fn == "ode_adams_bashforth2" ||
            fn == "ode_bdf2" || fn == "ode_exponential_euler" || fn == "ode_verlet" ||
            fn == "quantum_time_evolution" ||
            fn == "info_joint_entropy" ||
            fn == "cplx_power_series_eval" || fn == "cplx_winding_number" ||
            fn == "topo_vietoris_rips_betti0" || fn == "topo_betti_curve" ||
            fn == "info_conditional_entropy" || fn == "info_sample_entropy" ||
            fn == "info_permutation_entropy" || fn == "info_transfer_entropy" ||
            fn == "geo_kdtree_nearest" || fn == "geo_kdtree_3d_nearest" ||
            fn == "stats_ztest" || fn == "stats_ks_norm" || fn == "stats_bootstrap_mean" ||
            fn == "stats_partial_correlation" || fn == "stats_weighted_correlation" ||
            fn == "finance_binomial_call" ||
            fn == "finance_binomial_put" || fn == "finance_geo_asian_call" ||
            fn == "finance_geo_asian_put" || fn == "finance_bs_delta" ||
            fn == "finance_bs_theta" || fn == "finance_bs_rho" ||
            fn == "finance_portfolio_return" || fn == "finance_portfolio_variance" ||
            fn == "finance_information_ratio" || fn == "finance_treynor" ||
            fn == "finance_min_variance_portfolio" || fn == "finance_max_sharpe_portfolio" ||
            fn == "finance_efficient_frontier" || fn == "finance_max_sharpe" ||
            fn == "finance_bl_implied_returns" || fn == "finance_bl_posterior_returns" ||
            fn == "finance_bl_posterior_returns_default_omega" ||
            fn == "finance_merton_implied_asset_params" ||
            fn == "finance_heston_call" || fn == "finance_heston_put" ||
            fn == "cmaes" || fn == "bfgs" || fn == "nelder_mead" || fn == "lbfgs" ||
            fn == "adam" || fn == "conjugate_gradient" || fn == "rmsprop" ||
            fn == "adadelta" || fn == "golden_section" || fn == "levenberg_marquardt" ||
            fn == "bisection" || fn == "brentq" || fn == "secant" || fn == "halley" ||
            fn == "fixed_point" || fn == "illinois" || fn == "simulated_annealing" ||
            fn == "differential_evolution" || fn == "particle_swarm" ||
            fn == "gria_settling_time" || fn == "gria_hamming_distance" ||
            fn == "run_backtest_sharpe" ||
            fn == "run_backtest_max_drawdown" || fn == "run_backtest_total_return" ||
            fn == "cellai_energy" || fn == "gria_langton_lambda" || fn == "gria_alpha_ca" ||
            fn == "cellai_boltzmann_weights" ||
            fn == "cellai_cell_to_cypha_features" ||
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

Result<double> eval_scalar_call_cached(std::string_view fn_name, std::span<const double> args) {
    if (args.size() == 1) {
        const double arg = args[0];
        if (iequals(fn_name, "sin")) {
            return std::sin(arg);
        }
        if (iequals(fn_name, "cos")) {
            return std::cos(arg);
        }
        if (iequals(fn_name, "tan")) {
            return std::tan(arg);
        }
        if (iequals(fn_name, "asin")) {
            return std::asin(arg);
        }
        if (iequals(fn_name, "acos")) {
            return std::acos(arg);
        }
        if (iequals(fn_name, "atan")) {
            return std::atan(arg);
        }
        if (iequals(fn_name, "sinh")) {
            return std::sinh(arg);
        }
        if (iequals(fn_name, "cosh")) {
            return std::cosh(arg);
        }
        if (iequals(fn_name, "tanh")) {
            return std::tanh(arg);
        }
        if (iequals(fn_name, "sqrt")) {
            return std::sqrt(arg);
        }
        if (iequals(fn_name, "abs")) {
            return std::fabs(arg);
        }
        if (iequals(fn_name, "exp")) {
            return std::exp(arg);
        }
        if (iequals(fn_name, "log")) {
            return std::log(arg);
        }
        if (iequals(fn_name, "log10")) {
            return std::log10(arg);
        }
        if (iequals(fn_name, "floor")) {
            return std::floor(arg);
        }
        if (iequals(fn_name, "ceil")) {
            return std::ceil(arg);
        }
        if (iequals(fn_name, "erf")) {
            return ms::erf(arg);
        }
    }
    if (args.size() == 2) {
        if (iequals(fn_name, "pow")) {
            return std::pow(args[0], args[1]);
        }
        if (iequals(fn_name, "min")) {
            return std::fmin(args[0], args[1]);
        }
        if (iequals(fn_name, "max")) {
            return std::fmax(args[0], args[1]);
        }
        if (iequals(fn_name, "atan2")) {
            return std::atan2(args[0], args[1]);
        }
    }

    const std::string& fn_lower = g_scalar_fn_cache.lookup_lowered(fn_name);
    g_scalar_call_arg_buf.assign(args.begin(), args.end());
    return Interpreter::eval_scalar_call(fn_lower, g_scalar_call_arg_buf);
}

Result<double> eval_scalar_expr_impl(const SessionState& state, std::string_view expr_text) {
    std::string_view expr = strip_outer_parens_view(expr_text);
    if (expr.empty()) {
        return std::unexpected(DomainError{"eval", "empty expression"});
    }

    if (is_literal_arith_expr(expr)) {
        return eval_literal_arith(expr);
    }

    if (expr.front() == '-') {
        auto inner = eval_scalar_expr_impl(state, expr.substr(1));
        if (!inner) {
            return std::unexpected(inner.error());
        }
        return -(*inner);
    }
    if (expr.front() == '+') {
        return eval_scalar_expr_impl(state, expr.substr(1));
    }

    if (const auto call = parse_scalar_unary_call_view(expr)) {
        std::string_view arg_views[16];
        const size_t arg_count = split_scalar_call_args_view(call->second, arg_views, 16);
        if (arg_count == 0) {
            return std::unexpected(DomainError{"eval", "invalid scalar expression"});
        }
        if (arg_count > 16) {
            const std::string legacy_expr(expr);
            if (const auto legacy_call = parse_scalar_call(legacy_expr)) {
                std::vector<double> arg_values;
                arg_values.reserve(legacy_call->second.size());
                for (const auto& arg_text : legacy_call->second) {
                    auto arg = eval_scalar_expr_impl(state, std::string_view{arg_text});
                    if (!arg) {
                        return std::unexpected(arg.error());
                    }
                    arg_values.push_back(*arg);
                }
                return eval_scalar_call_cached(legacy_call->first,
                                               std::span<const double>(arg_values));
            }
            return std::unexpected(DomainError{"eval", "invalid scalar expression"});
        }
        std::array<double, 16> arg_values{};
        for (size_t i = 0; i < arg_count; ++i) {
            if (arg_views[i].empty()) {
                return std::unexpected(DomainError{"eval", "invalid scalar expression"});
            }
            auto arg = eval_scalar_expr_impl(state, arg_views[i]);
            if (!arg) {
                return std::unexpected(arg.error());
            }
            arg_values[i] = *arg;
        }
        return eval_scalar_call_cached(call->first, std::span(arg_values.data(), arg_count));
    }

    ScalarOperand single;
    if (parse_scalar_operand_view(expr, single)) {
        return resolve_scalar_operand(state, single);
    }

    const auto op_pos = find_scalar_binop_view(expr);
    if (!op_pos) {
        return std::unexpected(DomainError{"eval", "invalid scalar expression"});
    }

    auto left = eval_scalar_expr_impl(state, trim_view(expr.substr(0, op_pos->first)));
    if (!left) {
        return std::unexpected(left.error());
    }
    auto right = eval_scalar_expr_impl(state, trim_view(expr.substr(op_pos->first + 1)));
    if (!right) {
        return std::unexpected(right.error());
    }
    return Interpreter::eval_scalar_op(op_pos->second, *left, *right);
}

Result<double> eval_scalar_expr(const SessionState& state, const std::string& expr_text) {
    return eval_scalar_expr_impl(state, std::string_view{expr_text});
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

const char* format_compute_class(gria::ComputeClass cls) {
    switch (cls) {
    case gria::ComputeClass::Reversible:
        return "reversible";
    case gria::ComputeClass::Critical:
        return "critical";
    case gria::ComputeClass::Irreversible:
        return "irreversible";
    }
    return "reversible";
}

Result<void> require_session_rng(const char* fn) {
    if (!izaac::session_active()) {
        return std::unexpected(
            DomainError{fn, "session RNG not active; run 'izaac seed <n>' first"});
    }
    return {};
}

std::string format_nig_params(const cypha::NIGParams& params) {
    std::ostringstream out;
    out << "mu=" << params.mu << " alpha=" << params.alpha << " beta=" << params.beta
        << " delta=" << params.delta << "\n";
    return out.str();
}

const char* session_object_kind_name(const SessionObject& object) {
    if (std::holds_alternative<izaac::bloom::BloomFilter>(object)) {
        return "bloom";
    }
    if (std::holds_alternative<izaac::ratelimit::TokenBucket>(object)) {
        return "tokenbucket";
    }
    if (std::holds_alternative<cellai::CellMemory>(object)) {
        return "cellmemory";
    }
    if (std::holds_alternative<cypha::DifModel>(object)) {
        return "difmodel";
    }
    if (std::holds_alternative<izaac::consensus::Cluster>(object)) {
        return "cluster";
    }
    if (std::holds_alternative<tensorops::CPDecomposition>(object)) {
        return "cp";
    }
    if (std::holds_alternative<tensorops::NMFDecomposition>(object)) {
        return "nmf";
    }
    if (std::holds_alternative<tensorops::TTDecomposition>(object)) {
        return "tt";
    }
    return "tucker";
}

template <typename T>
const char* session_object_type_label() {
    if constexpr (std::is_same_v<T, izaac::bloom::BloomFilter>) {
        return "BloomFilter";
    }
    if constexpr (std::is_same_v<T, izaac::ratelimit::TokenBucket>) {
        return "TokenBucket";
    }
    if constexpr (std::is_same_v<T, cellai::CellMemory>) {
        return "CellMemory";
    }
    if constexpr (std::is_same_v<T, cypha::DifModel>) {
        return "DifModel";
    }
    if constexpr (std::is_same_v<T, izaac::consensus::Cluster>) {
        return "Cluster";
    }
    if constexpr (std::is_same_v<T, tensorops::CPDecomposition>) {
        return "CPDecomposition";
    }
    if constexpr (std::is_same_v<T, tensorops::TuckerDecomposition>) {
        return "TuckerDecomposition";
    }
    if constexpr (std::is_same_v<T, tensorops::NMFDecomposition>) {
        return "NMFDecomposition";
    }
    if constexpr (std::is_same_v<T, tensorops::TTDecomposition>) {
        return "TTDecomposition";
    }
    return "session object";
}

const char* format_node_role(izaac::consensus::NodeRole role) {
    switch (role) {
    case izaac::consensus::NodeRole::Follower:
        return "Follower";
    case izaac::consensus::NodeRole::Candidate:
        return "Candidate";
    case izaac::consensus::NodeRole::Leader:
        return "Leader";
    }
    return "Follower";
}

template <typename T>
Result<void> require_session_object_type(
    std::map<std::string, SessionObject>& registry,
    const std::string& handle,
    const char* fn,
    T*& out) {
    const auto it = registry.find(handle);
    if (it == registry.end()) {
        return std::unexpected(DomainError{fn, "session object not found: " + handle});
    }
    if (!std::holds_alternative<T>(it->second)) {
        return std::unexpected(DomainError{
            fn, std::string("session object '") + handle + "' is not a " +
                    session_object_type_label<T>()});
    }
    out = &std::get<T>(it->second);
    return {};
}

Result<std::vector<int>> parse_bracket_int_vector_literal(const std::string& text, const char* fn) {
    auto values = parse_bracket_vector_literal(text, fn);
    if (!values) {
        return std::unexpected(values.error());
    }
    if (values->empty()) {
        return std::unexpected(DomainError{fn, "expected non-empty integer vector literal"});
    }
    std::vector<int> out;
    out.reserve(values->size());
    for (double value : *values) {
        if (value < 1.0 || std::floor(value) != value) {
            return std::unexpected(DomainError{fn, "expected positive integer vector literal"});
        }
        out.push_back(static_cast<int>(value));
    }
    return out;
}

Result<void> parse_session_handle(const std::string& text, const char* fn, std::string& handle) {
    handle = trim_copy(text);
    if (!is_identifier(handle)) {
        return std::unexpected(DomainError{fn, "expected identifier handle"});
    }
    return {};
}

std::span<const uint8_t> string_item_bytes(const std::string& item) {
    return std::span<const uint8_t>(
        reinterpret_cast<const uint8_t*>(item.data()), item.size());
}

} // namespace ms::interp::detail
