// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_run_backtest_equity(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "run_backtest_equity" && assign.args.size() == 3) {
        auto prices = ctx.resolve_operand(assign.args[0]);
        if (!prices) {
            return std::unexpected(prices.error());
        }
        auto positions = ctx.resolve_operand(assign.args[1]);
        if (!positions) {
            return std::unexpected(positions.error());
        }
        auto capital = ctx.parse_scalar_arg(assign.args[2], "run_backtest_equity");
        if (!capital) {
            return std::unexpected(capital.error());
        }
        result = eval_run_backtest_equity(*prices, *positions, *capital);
    }

    return result;
}

void ms_register_matrix_call_run_backtest_equity() {
    register_matrix_call("run_backtest_equity", &handle_run_backtest_equity);
}

} // namespace ms::interp
