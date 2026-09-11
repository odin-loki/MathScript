// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_stats_arfit(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "stats_arfit" && assign.args.size() == 2) {
        auto x = ctx.resolve_operand(assign.args[0]);
        if (!x) {
            return std::unexpected(x.error());
        }
        double p_d = 0.0;
        if (!parse_number(assign.args[1], p_d)) {
            return std::unexpected(
                DomainError{"stats_arfit", "expected stats_arfit(x, p)"});
        }
        const int p = static_cast<int>(p_d);
        if (p < 1 || p_d != p) {
            return std::unexpected(
                DomainError{"stats_arfit", "expected positive integer p"});
        }
        result = eval_stats_arfit(*x, p);
    }

    return result;
}

void ms_register_matrix_call_stats_arfit() {
    register_matrix_call("stats_arfit", &handle_stats_arfit);
}

} // namespace ms::interp
