#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_finance_min_variance_portfolio(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "finance_min_variance_portfolio" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto w = eval_finance_min_variance_portfolio(*matrix);
        if (!w) {
            return std::unexpected(w.error());
        }
        result = *w;
    }

    return result;
}

void ms_register_matrix_call_finance_min_variance_portfolio() {
    register_matrix_call("finance_min_variance_portfolio", &handle_finance_min_variance_portfolio);
}

} // namespace ms::interp
