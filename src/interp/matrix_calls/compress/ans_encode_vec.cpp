// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_ans_encode_vec(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "ans_encode_vec" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto encoded = eval_ans_encode_vec(*matrix);
        if (!encoded) {
            return std::unexpected(encoded.error());
        }
        result = *encoded;
    }

    return result;
}

void ms_register_matrix_call_ans_encode_vec() {
    register_matrix_call("ans_encode_vec", &handle_ans_encode_vec);
}

} // namespace ms::interp
