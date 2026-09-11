// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_ml_confusion_matrix(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "ml_confusion_matrix" &&
        (assign.args.size() == 2 || assign.args.size() == 3)) {
        auto y_pred = ctx.resolve_operand(assign.args[0]);
        if (!y_pred) {
            return std::unexpected(y_pred.error());
        }
        auto y_true = ctx.resolve_operand(assign.args[1]);
        if (!y_true) {
            return std::unexpected(y_true.error());
        }
        double threshold = 0.5;
        if (assign.args.size() == 3) {
            auto thr = ctx.parse_scalar_arg(assign.args[2], "ml_confusion_matrix");
            if (!thr) {
                return std::unexpected(thr.error());
            }
            threshold = *thr;
        }
        result = eval_ml_confusion_matrix(*y_pred, *y_true, threshold);
    }

    return result;
}

void ms_register_matrix_call_ml_confusion_matrix() {
    register_matrix_call("ml_confusion_matrix", &handle_ml_confusion_matrix);
}

} // namespace ms::interp
