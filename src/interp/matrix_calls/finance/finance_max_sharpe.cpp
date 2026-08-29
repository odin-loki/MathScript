#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_finance_max_sharpe(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "finance_max_sharpe" && assign.args.size() == 3) {
        auto cov = resolve_operand(assign.args[0]);
        if (!cov) {
            return std::unexpected(cov.error());
        }
        auto mu = resolve_operand(assign.args[1]);
        if (!mu) {
            return std::unexpected(mu.error());
        }
        double risk_free = 0.0;
        if (!parse_number(assign.args[2], risk_free)) {
            return std::unexpected(DomainError{
                "finance_max_sharpe", "expected finance_max_sharpe(cov, mu, risk_free)"});
        }
        auto w = eval_finance_max_sharpe(*cov, *mu, risk_free);
        if (!w) {
            return std::unexpected(w.error());
        }
        result = *w;
    }

    return result;
}

void ms_register_matrix_call_finance_max_sharpe() {
    register_matrix_call("finance_max_sharpe", &handle_finance_max_sharpe);
}

} // namespace ms::interp
