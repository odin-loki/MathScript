// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_graph_tsp_heuristic(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "graph_tsp_heuristic" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto tour = eval_graph_tsp_heuristic(*matrix);
        if (!tour) {
            return std::unexpected(tour.error());
        }
        result = *tour;
    }

    return result;
}

void ms_register_matrix_call_graph_tsp_heuristic() {
    register_matrix_call("graph_tsp_heuristic", &handle_graph_tsp_heuristic);
}

} // namespace ms::interp
