// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_graph_mst_kruskal(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "graph_mst_kruskal" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto edges = eval_graph_mst_kruskal(*matrix);
        if (!edges) {
            return std::unexpected(edges.error());
        }
        result = *edges;
    }

    return result;
}

void ms_register_matrix_call_graph_mst_kruskal() {
    register_matrix_call("graph_mst_kruskal", &handle_graph_mst_kruskal);
}

} // namespace ms::interp
