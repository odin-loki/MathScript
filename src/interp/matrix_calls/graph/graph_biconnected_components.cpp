// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_graph_biconnected_components(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "graph_biconnected_components" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto bcc = eval_graph_biconnected_components(*matrix);
        if (!bcc) {
            return std::unexpected(bcc.error());
        }
        result = *bcc;
    }

    return result;
}

void ms_register_matrix_call_graph_biconnected_components() {
    register_matrix_call("graph_biconnected_components", &handle_graph_biconnected_components);
}

} // namespace ms::interp
