// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_signal_hilbert(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "signal_hilbert" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto analytic = eval_signal_hilbert(*matrix);
        if (!analytic) {
            return std::unexpected(analytic.error());
        }
        result = *analytic;
    }

    return result;
}

void ms_register_matrix_call_signal_hilbert() {
    register_matrix_call("signal_hilbert", &handle_signal_hilbert);
}

} // namespace ms::interp
