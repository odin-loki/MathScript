#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_ml_ridge_fit(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "ml_ridge_fit" && assign.args.size() == 3) {
        auto X = ctx.resolve_operand(assign.args[0]);
        if (!X) {
            return std::unexpected(X.error());
        }
        auto y = ctx.resolve_operand(assign.args[1]);
        if (!y) {
            return std::unexpected(y.error());
        }
        double alpha = 0.0;
        if (!parse_number(assign.args[2], alpha)) {
            return std::unexpected(DomainError{"ml_ridge_fit", "expected ml_ridge_fit(X, y, alpha)"});
        }
        auto fitted = eval_ml_ridge_fit(*X, *y, alpha);
        if (!fitted) {
            return std::unexpected(fitted.error());
        }
        result = *fitted;
    }

    return result;
}

void ms_register_matrix_call_ml_ridge_fit() {
    register_matrix_call("ml_ridge_fit", &handle_ml_ridge_fit);
}

} // namespace ms::interp
