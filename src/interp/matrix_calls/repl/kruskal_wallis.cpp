// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_kruskal_wallis(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "kruskal_wallis" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto stats = eval_kruskal_wallis(*matrix);
        if (!stats) {
            return std::unexpected(stats.error());
        }
        result = *stats;
    }

    return result;
}

void ms_register_matrix_call_kruskal_wallis() {
    register_matrix_call("kruskal_wallis", &handle_kruskal_wallis);
}

} // namespace ms::interp
