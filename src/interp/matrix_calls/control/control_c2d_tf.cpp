// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_control_c2d_tf(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "control_c2d_tf" && assign.args.size() == 3) {
        auto num_m = ctx.resolve_operand(assign.args[0]);
        if (!num_m) {
            return std::unexpected(num_m.error());
        }
        auto den_m = ctx.resolve_operand(assign.args[1]);
        if (!den_m) {
            return std::unexpected(den_m.error());
        }
        double Ts = 0.0;
        if (!parse_number(assign.args[2], Ts)) {
            auto ts_expr = eval_scalar_expr(ctx.state(), assign.args[2]);
            if (!ts_expr) {
                return std::unexpected(DomainError{"control_c2d_tf", "expected positive Ts"});
            }
            Ts = *ts_expr;
        }
        result = eval_control_c2d_tf(*num_m, *den_m, Ts, control::DiscretizationMethod::ZOH,
                                     "control_c2d_tf");
    }

    return result;
}

void ms_register_matrix_call_control_c2d_tf() {
    register_matrix_call("control_c2d_tf", &handle_control_c2d_tf);
}

} // namespace ms::interp
