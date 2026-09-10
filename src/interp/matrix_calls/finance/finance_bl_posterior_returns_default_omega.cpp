#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_finance_bl_posterior_returns_default_omega(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "finance_bl_posterior_returns_default_omega" &&
               assign.args.size() == 5) {
        auto pi = ctx.resolve_operand(assign.args[0]);
        if (!pi) {
            return std::unexpected(pi.error());
        }
        auto cov = ctx.resolve_operand(assign.args[1]);
        if (!cov) {
            return std::unexpected(cov.error());
        }
        auto P = ctx.resolve_operand(assign.args[2]);
        if (!P) {
            return std::unexpected(P.error());
        }
        auto Q = ctx.resolve_operand(assign.args[3]);
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
