// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_control_ss2tf(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "control_ss2tf" && assign.args.size() == 1) {
        auto ss_m = ctx.resolve_operand(assign.args[0]);
        if (!ss_m) {
            return std::unexpected(ss_m.error());
        }
        result = eval_control_ss2tf(*ss_m);
    }

    return result;
}

void ms_register_matrix_call_control_ss2tf() {
    register_matrix_call("control_ss2tf", &handle_control_ss2tf);
}

} // namespace ms::interp
