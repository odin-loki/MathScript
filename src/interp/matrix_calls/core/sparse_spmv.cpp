// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_sparse_spmv(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "sparse_spmv" && assign.args.size() == 2) {
        auto packed = ctx.resolve_operand(assign.args[0]);
        if (!packed) {
            return std::unexpected(packed.error());
        }
        auto x = ctx.resolve_operand(assign.args[1]);
        if (!x) {
            return std::unexpected(x.error());
        }
        result = eval_sparse_spmv(*packed, *x);
    }

    return result;
}

void ms_register_matrix_call_sparse_spmv() {
    register_matrix_call("sparse_spmv", &handle_sparse_spmv);
}

} // namespace ms::interp
