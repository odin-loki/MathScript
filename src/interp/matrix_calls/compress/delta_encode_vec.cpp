// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_delta_encode_vec(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "delta_encode_vec" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto bytes = matrix_to_bytes(*matrix, "delta_encode_vec");
        if (!bytes) {
            return std::unexpected(bytes.error());
        }
        result = bytes_to_matrix_col(compress::delta_encode(*bytes));
    }

    return result;
}

void ms_register_matrix_call_delta_encode_vec() {
    register_matrix_call("delta_encode_vec", &handle_delta_encode_vec);
}

} // namespace ms::interp
