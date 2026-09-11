// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_lzw_decode_vec(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "lzw_decode_vec" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto decoded = eval_lzw_decode_vec(*matrix);
        if (!decoded) {
            return std::unexpected(decoded.error());
        }
        result = *decoded;
    }

    return result;
}

void ms_register_matrix_call_lzw_decode_vec() {
    register_matrix_call("lzw_decode_vec", &handle_lzw_decode_vec);
}

} // namespace ms::interp
