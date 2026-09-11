// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_graph_katz_centrality(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "graph_katz_centrality" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto kc = eval_graph_katz_centrality(*matrix);
        if (!kc) {
            return std::unexpected(kc.error());
        }
        result = *kc;
    }

    return result;
}

void ms_register_matrix_call_graph_katz_centrality() {
    register_matrix_call("graph_katz_centrality", &handle_graph_katz_centrality);
}

} // namespace ms::interp
