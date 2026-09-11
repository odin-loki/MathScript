// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_stats_shapiro_wilk(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "stats_shapiro_wilk" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        result = eval_stats_shapiro_wilk(*matrix);
    }

    return result;
}

void ms_register_matrix_call_stats_shapiro_wilk() {
    register_matrix_call("stats_shapiro_wilk", &handle_stats_shapiro_wilk);
}

} // namespace ms::interp
