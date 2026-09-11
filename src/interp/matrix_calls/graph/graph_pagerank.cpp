// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_graph_pagerank(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "graph_pagerank" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto G = graph_from_adjacency(*matrix, "graph_pagerank");
        if (!G) {
            return std::unexpected(G.error());
        }
        result = vector_to_column(graph::pagerank(*G));
    }

    return result;
}

void ms_register_matrix_call_graph_pagerank() {
    register_matrix_call("graph_pagerank", &handle_graph_pagerank);
}

} // namespace ms::interp
