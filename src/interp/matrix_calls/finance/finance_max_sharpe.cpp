// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_finance_max_sharpe(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "finance_max_sharpe" && assign.args.size() == 3) {
        auto cov = ctx.resolve_operand(assign.args[0]);
        if (!cov) {
            return std::unexpected(cov.error());
        }
        auto mu = ctx.resolve_operand(assign.args[1]);
        if (!mu) {
            return std::unexpected(mu.error());
        }
        double risk_free = 0.0;
        if (!parse_number(assign.args[2], risk_free)) {
            return std::unexpected(DomainError{
                "finance_max_sharpe", "expected finance_max_sharpe(cov, mu, risk_free)"});
        }
        auto w = eval_finance_max_sharpe(*cov, *mu, risk_free);
        if (!w) {
            return std::unexpected(w.error());
        }
        result = *w;
    }

    return result;
}

void ms_register_matrix_call_finance_max_sharpe() {
    register_matrix_call("finance_max_sharpe", &handle_finance_max_sharpe);
}

} // namespace ms::interp
