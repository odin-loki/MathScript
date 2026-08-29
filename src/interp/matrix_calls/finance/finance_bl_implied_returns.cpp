#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_finance_bl_implied_returns(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "finance_bl_implied_returns" && assign.args.size() == 3) {
        auto cov = resolve_operand(assign.args[0]);
        if (!cov) {
            return std::unexpected(cov.error());
        }
        auto w_mkt = resolve_operand(assign.args[1]);
        if (!w_mkt) {
            return std::unexpected(w_mkt.error());
        }
        double delta = 0.0;
        if (!parse_number(assign.args[2], delta)) {
            return std::unexpected(DomainError{
                "finance_bl_implied_returns",
                "expected finance_bl_implied_returns(cov, w_mkt, delta)"});
        }
        auto pi = eval_finance_bl_implied_returns(*cov, *w_mkt, delta);
        if (!pi) {
            return std::unexpected(pi.error());
        }
        result = *pi;
    }

    return result;
}

void ms_register_matrix_call_finance_bl_implied_returns() {
    register_matrix_call("finance_bl_implied_returns", &handle_finance_bl_implied_returns);
}

} // namespace ms::interp
