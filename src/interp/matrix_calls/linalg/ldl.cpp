// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_ldl(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "ldl" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto decomp = ldl(*matrix);
        if (!decomp) {
            return std::unexpected(decomp.error());
        }
        result = decomp->L;
    }

    return result;
}

void ms_register_matrix_call_ldl() {
    register_matrix_call("ldl", &handle_ldl);
}

} // namespace ms::interp
