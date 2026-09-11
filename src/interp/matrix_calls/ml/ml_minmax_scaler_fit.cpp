// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_ml_minmax_scaler_fit(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "ml_minmax_scaler_fit" && assign.args.size() == 1) {
        auto X = ctx.resolve_operand(assign.args[0]);
        if (!X) {
            return std::unexpected(X.error());
        }
        auto fitted = eval_ml_minmax_scaler_fit(*X);
        if (!fitted) {
            return std::unexpected(fitted.error());
        }
        result = *fitted;
    }

    return result;
}

void ms_register_matrix_call_ml_minmax_scaler_fit() {
    register_matrix_call("ml_minmax_scaler_fit", &handle_ml_minmax_scaler_fit);
}

} // namespace ms::interp
