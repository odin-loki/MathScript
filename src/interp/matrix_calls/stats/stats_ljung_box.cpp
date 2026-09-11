// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_stats_ljung_box(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "stats_ljung_box" && assign.args.size() == 2) {
        auto x = ctx.resolve_operand(assign.args[0]);
        if (!x) {
            return std::unexpected(x.error());
        }
        double max_lag_d = 0.0;
        if (!parse_number(assign.args[1], max_lag_d)) {
            return std::unexpected(
                DomainError{"stats_ljung_box", "expected stats_ljung_box(x, max_lag)"});
        }
        const int max_lag = static_cast<int>(max_lag_d);
        if (max_lag < 1 || max_lag_d != max_lag) {
            return std::unexpected(
                DomainError{"stats_ljung_box", "expected positive integer max_lag"});
        }
        result = eval_stats_ljung_box(*x, max_lag);
    }

    return result;
}

void ms_register_matrix_call_stats_ljung_box() {
    register_matrix_call("stats_ljung_box", &handle_stats_ljung_box);
}

} // namespace ms::interp
