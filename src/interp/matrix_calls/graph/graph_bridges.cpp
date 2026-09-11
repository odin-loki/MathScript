// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_graph_bridges(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "graph_bridges" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto br = eval_graph_bridges(*matrix);
        if (!br) {
            return std::unexpected(br.error());
        }
        result = *br;
    }

    return result;
}

void ms_register_matrix_call_graph_bridges() {
    register_matrix_call("graph_bridges", &handle_graph_bridges);
}

} // namespace ms::interp
