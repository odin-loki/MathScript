#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_finance_bl_posterior_returns_default_omega(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "finance_bl_posterior_returns_default_omega" &&
               assign.args.size() == 5) {
        auto pi = resolve_operand(assign.args[0]);
        if (!pi) {
            return std::unexpected(pi.error());
        }
        auto cov = resolve_operand(assign.args[1]);
        if (!cov) {
            return std::unexpected(cov.error());
        }
        auto P = resolve_operand(assign.args[2]);
        if (!P) {
            return std::unexpected(P.error());
        }
        auto Q = resolve_operand(assign.args[3]);
        if (!Q) {
            return std::unexpected(Q.error());
        }
        double tau = 0.0;
        if (!parse_number(assign.args[4], tau)) {
            auto tau_expr = eval_scalar_expr(ctx.state(), assign.args[4]);
            if (!tau_expr) {
                return std::unexpected(DomainError{
                    "finance_bl_posterior_returns_default_omega",
                    "expected finance_bl_posterior_returns_default_omega(pi, cov, P, Q, tau)"});
            }
            tau = *tau_expr;
        }
        auto post = eval_finance_bl_posterior_returns_default_omega(*pi, *cov, *P, *Q, tau);
        if (!post) {
            return std::unexpected(post.error());
        }
        result = *post;
    }

    return result;
}

void ms_register_matrix_call_finance_bl_posterior_returns_default_omega() {
    register_matrix_call("finance_bl_posterior_returns_default_omega", &handle_finance_bl_posterior_returns_default_omega);
}

} // namespace ms::interp
