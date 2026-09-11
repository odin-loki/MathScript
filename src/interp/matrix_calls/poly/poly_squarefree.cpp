// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_poly_squarefree(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "poly_squarefree" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto sf = eval_poly_squarefree(*matrix);
        if (!sf) {
            return std::unexpected(sf.error());
        }
        result = *sf;
    }

    return result;
}

void ms_register_matrix_call_poly_squarefree() {
    register_matrix_call("poly_squarefree", &handle_poly_squarefree);
}

} // namespace ms::interp
