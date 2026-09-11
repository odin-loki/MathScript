// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_mat_col(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    if (assign.callee != "mat_col" || assign.args.size() != 2) {
        return std::unexpected(DomainError{"mat_col", "expected mat_col(A, j)"});
    }
    auto A = ctx.resolve_operand(assign.args[0]);
    if (!A) {
        return std::unexpected(A.error());
    }
    auto index = ctx.parse_scalar_arg(assign.args[1], "mat_col");
    if (!index) {
        return std::unexpected(index.error());
    }
    return eval_mat_col(*A, *index);
}

void ms_register_matrix_call_mat_col() {
    register_matrix_call("mat_col", &handle_mat_col);
}

} // namespace ms::interp
