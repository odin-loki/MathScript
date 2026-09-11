// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_graph_closeness(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "graph_closeness" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto cc = eval_graph_closeness(*matrix);
        if (!cc) {
            return std::unexpected(cc.error());
        }
        result = *cc;
    }

    return result;
}

void ms_register_matrix_call_graph_closeness() {
    register_matrix_call("graph_closeness", &handle_graph_closeness);
}

} // namespace ms::interp
