#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_randn(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if ((assign.callee == "rand" || assign.callee == "randn") && assign.args.size() == 2) {
        double m_d = 0.0, n_d = 0.0;
        if (!parse_number(assign.args[0], m_d)) {
            auto it = ctx.state().scalars.find(assign.args[0]);
            if (it != ctx.state().scalars.end()) m_d = it->second;
            else return std::unexpected(DomainError{assign.callee, "expected numeric size"});
        }
        if (!parse_number(assign.args[1], n_d)) {
            auto it = ctx.state().scalars.find(assign.args[1]);
            if (it != ctx.state().scalars.end()) n_d = it->second;
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
    }

    return result;
}

void ms_register_matrix_call_randn() {
    register_matrix_call("randn", &handle_randn);
}

} // namespace ms::interp
