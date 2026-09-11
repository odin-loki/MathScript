// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_poly_factor_rational(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "poly_factor_rational" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto fact = eval_poly_factor_rational(*matrix, 1e-6);
        if (!fact) {
            return std::unexpected(fact.error());
        }
        result = *fact;
    }

    return result;
}

void ms_register_matrix_call_poly_factor_rational() {
    register_matrix_call("poly_factor_rational", &handle_poly_factor_rational);
}

} // namespace ms::interp
