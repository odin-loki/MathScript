// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_signal_sosfilt(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "signal_sosfilt" && assign.args.size() == 2) {
        auto sos = ctx.resolve_operand(assign.args[0]);
        if (!sos) {
            return std::unexpected(sos.error());
        }
        auto x = ctx.resolve_operand(assign.args[1]);
        if (!x) {
            return std::unexpected(x.error());
        }
        result = eval_signal_sosfilt(*sos, *x);
    }

    return result;
}

void ms_register_matrix_call_signal_sosfilt() {
    register_matrix_call("signal_sosfilt", &handle_signal_sosfilt);
}

} // namespace ms::interp
