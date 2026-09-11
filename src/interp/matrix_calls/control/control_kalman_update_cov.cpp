// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_control_kalman_update_cov(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "control_kalman_update_cov" && assign.args.size() == 5) {
        auto x_m = ctx.resolve_operand(assign.args[0]);
        if (!x_m) {
            return std::unexpected(x_m.error());
        }
        auto P_m = ctx.resolve_operand(assign.args[1]);
        if (!P_m) {
            return std::unexpected(P_m.error());
        }
        auto z_m = ctx.resolve_operand(assign.args[2]);
        if (!z_m) {
            return std::unexpected(z_m.error());
        }
        auto H_m = ctx.resolve_operand(assign.args[3]);
        if (!H_m) {
            return std::unexpected(H_m.error());
        }
        auto R_m = ctx.resolve_operand(assign.args[4]);
        if (!R_m) {
            return std::unexpected(R_m.error());
        }
        auto updated = eval_control_kalman_update_cov(*x_m, *P_m, *z_m, *H_m, *R_m);
        if (!updated) {
            return std::unexpected(updated.error());
        }
        result = *updated;
    }

    return result;
}

void ms_register_matrix_call_control_kalman_update_cov() {
    register_matrix_call("control_kalman_update_cov", &handle_control_kalman_update_cov);
}

} // namespace ms::interp
