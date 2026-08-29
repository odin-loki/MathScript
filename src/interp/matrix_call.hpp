#pragma once

#include "ms/interp/repl_engine.hpp"
#include <cmath>
#include <cstddef>
#include <string_view>

namespace ms::interp {

constexpr std::size_t kMaxReplMatrixElems = 262144;
inline const char* kReplMatrixTooLarge = "matrix too large (max 262144 elements)";

inline bool repl_elems_allowed(std::size_t rows, std::size_t cols) {
    return rows == 0 || cols == 0 ||
           (rows <= kMaxReplMatrixElems && cols <= kMaxReplMatrixElems &&
            rows <= kMaxReplMatrixElems / cols);
}

inline bool repl_dims_allowed(double m_d, double n_d, std::size_t& rows, std::size_t& cols) {
    if (!(m_d >= 1.0) || !(n_d >= 1.0) || m_d != std::floor(m_d) || n_d != std::floor(n_d)) {
        return false;
    }
    if (m_d > static_cast<double>(kMaxReplMatrixElems) ||
        n_d > static_cast<double>(kMaxReplMatrixElems)) {
        return false;
    }
    rows = static_cast<std::size_t>(m_d);
    cols = static_cast<std::size_t>(n_d);
    return repl_elems_allowed(rows, cols);
}

struct MatrixCallCtx {
    Interpreter& interp;
    explicit MatrixCallCtx(Interpreter& i) : interp(i) {}
    Result<Matrix<double>> resolve_operand(const std::string& text) {
        return interp.eval_matrix_operand(text);
    }
    SessionState& state() { return interp.state_; }
    const SessionState& state() const { return interp.state_; }
    auto& session_objects() { return interp.session_objects_; }
    const auto& session_objects() const { return interp.session_objects_; }
};

using MatrixCallHandler = Result<Matrix<double>> (*)(Interpreter&, const MatrixCallAssign&);

void register_matrix_call(std::string_view name, MatrixCallHandler handler);
void ensure_matrix_calls_registered();
bool is_matrix_call_callee(const std::string& callee);
Result<Matrix<double>> dispatch_matrix_call(Interpreter& interp, const MatrixCallAssign& assign);

} // namespace ms::interp
