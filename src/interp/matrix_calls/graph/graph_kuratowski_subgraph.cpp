// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_graph_kuratowski_subgraph(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "graph_kuratowski_subgraph" && assign.args.size() == 1) {
        auto arg0 = ctx.resolve_operand(assign.args[0]);
        if (!arg0) {
            return std::unexpected(arg0.error());
        }
        auto out = eval_graph_kuratowski_subgraph(*arg0);
        if (!out) {
            return std::unexpected(out.error());
        }
        result = *out;
    }

    return result;
}

void ms_register_matrix_call_graph_kuratowski_subgraph() {
    register_matrix_call("graph_kuratowski_subgraph", &handle_graph_kuratowski_subgraph);
}

} // namespace ms::interp
