// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_fem_apply_dirichlet(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "fem_apply_dirichlet" && assign.args.size() == 4) {
        auto K = ctx.resolve_operand(assign.args[0]);
        if (!K) {
            return std::unexpected(K.error());
        }
        auto f = ctx.resolve_operand(assign.args[1]);
        if (!f) {
            return std::unexpected(f.error());
        }
        auto nodes = ctx.resolve_operand(assign.args[2]);
        if (!nodes) {
            return std::unexpected(nodes.error());
        }
        auto values = ctx.resolve_operand(assign.args[3]);
        if (!values) {
            return std::unexpected(values.error());
        }
        result = eval_fem_apply_dirichlet(*K, *f, *nodes, *values);
    }

    return result;
}

void ms_register_matrix_call_fem_apply_dirichlet() {
    register_matrix_call("fem_apply_dirichlet", &handle_fem_apply_dirichlet);
}

} // namespace ms::interp
