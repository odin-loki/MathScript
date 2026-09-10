#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_ml_lasso_fit(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "ml_lasso_fit" && assign.args.size() == 3) {
        auto X = ctx.resolve_operand(assign.args[0]);
        if (!X) {
            return std::unexpected(X.error());
        }
        auto y = ctx.resolve_operand(assign.args[1]);
        if (!y) {
            return std::unexpected(y.error());
        }
        auto alpha = ctx.parse_scalar_arg(assign.args[2], "ml_lasso_fit");
        if (!alpha) {
            return std::unexpected(alpha.error());
        }
        auto fitted = eval_ml_lasso_fit(*X, *y, *alpha);
        if (!fitted) {
            return std::unexpected(fitted.error());
        }
        result = *fitted;
    }

    return result;
}

void ms_register_matrix_call_ml_lasso_fit() {
    register_matrix_call("ml_lasso_fit", &handle_ml_lasso_fit);
}

} // namespace ms::interp
