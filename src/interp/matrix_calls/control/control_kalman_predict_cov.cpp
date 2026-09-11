// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_control_kalman_predict_cov(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "control_kalman_predict_cov" && assign.args.size() == 4) {
        auto x_m = ctx.resolve_operand(assign.args[0]);
        if (!x_m) {
            return std::unexpected(x_m.error());
        }
        auto P_m = ctx.resolve_operand(assign.args[1]);
        if (!P_m) {
            return std::unexpected(P_m.error());
        }
        auto A_m = ctx.resolve_operand(assign.args[2]);
        if (!A_m) {
            return std::unexpected(A_m.error());
        }
        auto Q_m = ctx.resolve_operand(assign.args[3]);
        if (!Q_m) {
            return std::unexpected(Q_m.error());
        }
        auto predicted = eval_control_kalman_predict_cov(*x_m, *P_m, *A_m, *Q_m);
        if (!predicted) {
            return std::unexpected(predicted.error());
        }
        result = *predicted;
    }

    return result;
}

void ms_register_matrix_call_control_kalman_predict_cov() {
    register_matrix_call("control_kalman_predict_cov", &handle_control_kalman_predict_cov);
}

} // namespace ms::interp
