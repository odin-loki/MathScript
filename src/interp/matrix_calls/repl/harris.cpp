// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_harris(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "harris" && assign.args.size() >= 1 && assign.args.size() <= 3) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        double k = 0.04;
        double threshold = 0.01;
        if (assign.args.size() >= 2 && !parse_number(assign.args[1], k)) {
            return std::unexpected(
                DomainError{"harris", "expected harris(M), harris(M, k), or harris(M, k, threshold)"});
        }
        if (assign.args.size() == 3 && !parse_number(assign.args[2], threshold)) {
            return std::unexpected(
                DomainError{"harris", "expected harris(M), harris(M, k), or harris(M, k, threshold)"});
        }
        result = eval_harris(*matrix, static_cast<float>(k), static_cast<float>(threshold));
    }

    return result;
}

void ms_register_matrix_call_harris() {
    register_matrix_call("harris", &handle_harris);
}

} // namespace ms::interp
