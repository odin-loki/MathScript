// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_signal_savgol(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "signal_savgol" && assign.args.size() == 3) {
        auto x = ctx.resolve_operand(assign.args[0]);
        if (!x) {
            return std::unexpected(x.error());
        }
        auto window_val = ctx.parse_scalar_arg(assign.args[1], "signal_savgol");
        if (!window_val) {
            return std::unexpected(window_val.error());
        }
        auto poly_val = ctx.parse_scalar_arg(assign.args[2], "signal_savgol");
        if (!poly_val) {
            return std::unexpected(poly_val.error());
        }
        result = eval_signal_savgol_wp(*x, *window_val, *poly_val);
    }

    return result;
}

void ms_register_matrix_call_signal_savgol() {
    register_matrix_call("signal_savgol", &handle_signal_savgol);
}

} // namespace ms::interp
