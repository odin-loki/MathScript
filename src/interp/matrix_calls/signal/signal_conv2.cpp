// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_signal_conv2(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "signal_conv2" && assign.args.size() == 2) {
        auto A = ctx.resolve_operand(assign.args[0]);
        if (!A) {
            return std::unexpected(A.error());
        }
        auto K = ctx.resolve_operand(assign.args[1]);
        if (!K) {
            return std::unexpected(K.error());
        }
        result = eval_signal_conv2(*A, *K);
    }

    return result;
}

void ms_register_matrix_call_signal_conv2() {
    register_matrix_call("signal_conv2", &handle_signal_conv2);
}

} // namespace ms::interp
