// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_fft_idct2(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "fft_idct2" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto signal = eval_fft_idct2(*matrix);
        if (!signal) {
            return std::unexpected(signal.error());
        }
        result = *signal;
    }

    return result;
}

void ms_register_matrix_call_fft_idct2() {
    register_matrix_call("fft_idct2", &handle_fft_idct2);
}

} // namespace ms::interp
