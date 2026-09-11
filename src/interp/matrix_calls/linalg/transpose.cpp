// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_transpose(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "transpose" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        const auto transposed = transpose(*matrix);
        Matrix<double> stored(transposed.rows(), transposed.cols());
        for (size_t i = 0; i < transposed.rows(); ++i) {
            for (size_t j = 0; j < transposed.cols(); ++j) {
                stored(i, j) = transposed(i, j);
            }
        }
        result = stored;
    }

    return result;
}

void ms_register_matrix_call_transpose() {
    register_matrix_call("transpose", &handle_transpose);
}

} // namespace ms::interp
