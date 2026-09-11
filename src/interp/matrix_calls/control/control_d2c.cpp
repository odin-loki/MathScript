// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_control_d2c(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "control_d2c" && assign.args.size() == 5) {
        auto A_m = ctx.resolve_operand(assign.args[0]);
        if (!A_m) {
            return std::unexpected(A_m.error());
        }
        auto B_m = ctx.resolve_operand(assign.args[1]);
        if (!B_m) {
            return std::unexpected(B_m.error());
        }
        auto C_m = ctx.resolve_operand(assign.args[2]);
        if (!C_m) {
            return std::unexpected(C_m.error());
        }
        auto D_m = ctx.resolve_operand(assign.args[3]);
        if (!D_m) {
            return std::unexpected(D_m.error());
        }
        double Ts = 0.0;
        if (!parse_number(assign.args[4], Ts)) {
            auto ts_expr = eval_scalar_expr(ctx.state(), assign.args[4]);
            if (!ts_expr) {
                return std::unexpected(DomainError{"control_d2c", "expected positive Ts"});
            }
            Ts = *ts_expr;
        }
        result = eval_control_d2c(*A_m, *B_m, *C_m, *D_m, Ts,
                                  control::DiscretizationMethod::ZOH, "control_d2c");
    }

    return result;
}

void ms_register_matrix_call_control_d2c() {
    register_matrix_call("control_d2c", &handle_control_d2c);
}

} // namespace ms::interp
