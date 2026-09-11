// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_poly_deriv(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "poly_deriv" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto deriv = eval_poly_deriv(*matrix);
        if (!deriv) {
            return std::unexpected(deriv.error());
        }
        result = *deriv;
    }

    return result;
}

void ms_register_matrix_call_poly_deriv() {
    register_matrix_call("poly_deriv", &handle_poly_deriv);
}

} // namespace ms::interp
