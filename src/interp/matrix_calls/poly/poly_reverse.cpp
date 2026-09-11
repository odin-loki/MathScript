// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_poly_reverse(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "poly_reverse" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto reversed = eval_poly_reverse(*matrix);
        if (!reversed) {
            return std::unexpected(reversed.error());
        }
        result = *reversed;
    }

    return result;
}

void ms_register_matrix_call_poly_reverse() {
    register_matrix_call("poly_reverse", &handle_poly_reverse);
}

} // namespace ms::interp
