#pragma once

#include "ms/interp/repl_engine.hpp"
#include <cmath>
#include <cstddef>
#include <string>
#include <string_view>

namespace ms::interp {

// Declared here so MatrixCallCtx's argument helpers can use them without this
// header pulling in repl_engine_internal.hpp, which every handler already
// includes for its own body. Definitions live in repl_engine_internal.cpp.
namespace detail {
bool parse_number(const std::string& text, double& value);
Result<double> eval_scalar_expr(const SessionState& state, const std::string& expr_text);
} // namespace detail

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

    // The argument helpers below used to be stamped into every generated handler
    // as a local lambda, whether or not it used them: parse_uint64_arg was
    // defined in 480 translation units and called in one. They are identical in
    // every copy, so they belong here once.

    /// A numeric literal, or a scalar expression evaluated in the session.
    Result<double> parse_scalar_arg(const std::string& arg_text, const char* fn) {
        double value = 0.0;
        if (detail::parse_number(arg_text, value)) {
            return value;
        }
        auto expr = detail::eval_scalar_expr(state(), arg_text);
        if (!expr) {
            return std::unexpected(DomainError{fn, "expected numeric scalar argument"});
        }
        return *expr;
    }

    /// A dimension: a positive integer, exactly representable as the double it
    /// arrived as.
    static Result<std::size_t> parse_positive_size_arg(double value, const char* fn,
                                                       const char* label) {
        const int i = static_cast<int>(value);
        if (i < 1 || value != static_cast<double>(i)) {
            return std::unexpected(DomainError{fn, label});
        }
        return static_cast<std::size_t>(i);
    }

    /// A non-negative whole number.
    static Result<std::uint64_t> parse_uint64_arg(double value, const char* fn,
                                                  const char* label) {
        if (value < 0.0 || value != std::floor(value)) {
            return std::unexpected(DomainError{fn, label});
        }
        return static_cast<std::uint64_t>(value);
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
