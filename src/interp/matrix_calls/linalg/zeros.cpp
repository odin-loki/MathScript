#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_zeros(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);
    auto resolve_operand = [&ctx](const std::string& text) { return ctx.resolve_operand(text); };
    auto parse_scalar_arg = [&ctx](const std::string& arg_text, const char* fn) -> Result<double> {
        double value = 0.0;
        if (parse_number(arg_text, value)) return value;
        auto expr = eval_scalar_expr(ctx.state(), arg_text);
        if (!expr) {
            return std::unexpected(DomainError{fn, "expected numeric scalar argument"});
        }
        return *expr;
    };
    auto parse_positive_size_arg = [](double value, const char* fn, const char* label) -> Result<std::size_t> {
        const int i = static_cast<int>(value);
        if (i < 1 || value != static_cast<double>(i)) {
            return std::unexpected(DomainError{fn, label});
        }
        return static_cast<std::size_t>(i);
    };
    auto parse_uint64_arg = [](double value, const char* fn, const char* label) -> Result<uint64_t> {
        if (value < 0.0 || value != std::floor(value)) {
            return std::unexpected(DomainError{fn, label});
        }
        return static_cast<uint64_t>(value);
    };

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if ((assign.callee == "zeros" || assign.callee == "eye" || assign.callee == "ones") &&
               (assign.args.size() == 1 || assign.args.size() == 2)) {
        double m_d = 0.0, n_d = 0.0;
        if (!parse_number(assign.args[0], m_d)) {
            // try resolving as scalar variable
            auto it = ctx.state().scalars.find(assign.args[0]);
            if (it != ctx.state().scalars.end()) {
                m_d = it->second;
            } else {
                return std::unexpected(DomainError{assign.callee, "expected numeric size argument"});
            }
        }
        if (assign.args.size() == 2) {
            if (!parse_number(assign.args[1], n_d)) {
                auto it = ctx.state().scalars.find(assign.args[1]);
                if (it != ctx.state().scalars.end()) {
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
    }

    return result;
}

void ms_register_matrix_call_zeros() {
    register_matrix_call("zeros", &handle_zeros);
}

} // namespace ms::interp
