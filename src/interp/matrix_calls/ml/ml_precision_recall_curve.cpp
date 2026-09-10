#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_ml_precision_recall_curve(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "ml_precision_recall_curve" && assign.args.size() == 2) {
        auto y_pred = ctx.resolve_operand(assign.args[0]);
        if (!y_pred) {
            return std::unexpected(y_pred.error());
        }
        auto y_true = ctx.resolve_operand(assign.args[1]);
        if (!y_true) {
            return std::unexpected(y_true.error());
        }
        result = eval_ml_precision_recall_curve(*y_pred, *y_true);
    }

    return result;
}

void ms_register_matrix_call_ml_precision_recall_curve() {
    register_matrix_call("ml_precision_recall_curve", &handle_ml_precision_recall_curve);
}

} // namespace ms::interp
