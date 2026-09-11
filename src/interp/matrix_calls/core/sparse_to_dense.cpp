// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_sparse_to_dense(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "sparse_to_dense" && assign.args.size() == 1) {
        auto packed = ctx.resolve_operand(assign.args[0]);
        if (!packed) {
            return std::unexpected(packed.error());
        }
        result = eval_sparse_to_dense(*packed);
    }

    return result;
}

void ms_register_matrix_call_sparse_to_dense() {
    register_matrix_call("sparse_to_dense", &handle_sparse_to_dense);
}

} // namespace ms::interp
