// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_jacobi(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "jacobi" && assign.args.size() == 2) {
        auto left = ctx.resolve_operand(assign.args[0]);
        if (!left) {
            return std::unexpected(left.error());
        }
        auto right = ctx.resolve_operand(assign.args[1]);
        if (!right) {
            return std::unexpected(right.error());
        }
        result = jacobi(*left, *right);
    }

    return result;
}

void ms_register_matrix_call_jacobi() {
    register_matrix_call("jacobi", &handle_jacobi);
}

} // namespace ms::interp
