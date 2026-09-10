// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_stats_one_way_anova(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "stats_one_way_anova" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        result = eval_stats_one_way_anova(*matrix);
    }

    return result;
}

void ms_register_matrix_call_stats_one_way_anova() {
    register_matrix_call("stats_one_way_anova", &handle_stats_one_way_anova);
}

} // namespace ms::interp
