// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_graph_min_arborescence(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "graph_min_arborescence" && assign.args.size() == 2) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        double root_d = 0.0;
        if (!parse_number(assign.args[1], root_d)) {
            auto root_expr = eval_scalar_expr(ctx.state(), assign.args[1]);
            if (!root_expr) {
                return std::unexpected(DomainError{
                    "graph_min_arborescence", "expected graph_min_arborescence(A, root)"});
            }
            root_d = *root_expr;
        }
        const int root = static_cast<int>(root_d);
        if (root < 0 || root_d != root) {
            return std::unexpected(
                DomainError{"graph_min_arborescence", "expected non-negative integer root"});
        }
        auto arb = eval_graph_min_arborescence(*matrix, root);
        if (!arb) {
            return std::unexpected(arb.error());
        }
        result = *arb;
    }

    return result;
}

void ms_register_matrix_call_graph_min_arborescence() {
    register_matrix_call("graph_min_arborescence", &handle_graph_min_arborescence);
}

} // namespace ms::interp
