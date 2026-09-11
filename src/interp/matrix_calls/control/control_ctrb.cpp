// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_control_ctrb(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "control_ctrb" && assign.args.size() == 2) {
        auto A_m = ctx.resolve_operand(assign.args[0]);
        if (!A_m) {
            return std::unexpected(A_m.error());
        }
        auto B_m = ctx.resolve_operand(assign.args[1]);
        if (!B_m) {
            return std::unexpected(B_m.error());
        }
        auto value = eval_control_ctrb(*A_m, *B_m);
        if (!value) {
            return std::unexpected(value.error());
        }
        result = *value;
    }

    return result;
}

void ms_register_matrix_call_control_ctrb() {
    register_matrix_call("control_ctrb", &handle_control_ctrb);
}

} // namespace ms::interp
