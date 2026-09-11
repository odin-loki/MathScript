// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_signal_envelope(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "signal_envelope" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto env = eval_signal_envelope(*matrix);
        if (!env) {
            return std::unexpected(env.error());
        }
        result = *env;
    }

    return result;
}

void ms_register_matrix_call_signal_envelope() {
    register_matrix_call("signal_envelope", &handle_signal_envelope);
}

} // namespace ms::interp
