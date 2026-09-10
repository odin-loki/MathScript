#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_finance_efficient_frontier(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "finance_efficient_frontier" && assign.args.size() == 3) {
        auto cov = ctx.resolve_operand(assign.args[0]);
        if (!cov) {
            return std::unexpected(cov.error());
        }
        auto mu = ctx.resolve_operand(assign.args[1]);
        if (!mu) {
            return std::unexpected(mu.error());
        }
        double target_return = 0.0;
        if (!parse_number(assign.args[2], target_return)) {
            return std::unexpected(DomainError{
                "finance_efficient_frontier",
                "expected finance_efficient_frontier(cov, mu, target_return)"});
        }
        auto w = eval_finance_efficient_frontier(*cov, *mu, target_return);
        if (!w) {
            return std::unexpected(w.error());
        }
        result = *w;
    }

    return result;
}

void ms_register_matrix_call_finance_efficient_frontier() {
    register_matrix_call("finance_efficient_frontier", &handle_finance_efficient_frontier);
}

} // namespace ms::interp
