// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_finance_bl_implied_returns(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "finance_bl_implied_returns" && assign.args.size() == 3) {
        auto cov = ctx.resolve_operand(assign.args[0]);
        if (!cov) {
            return std::unexpected(cov.error());
        }
        auto w_mkt = ctx.resolve_operand(assign.args[1]);
        if (!w_mkt) {
            return std::unexpected(w_mkt.error());
        }
        double delta = 0.0;
        if (!parse_number(assign.args[2], delta)) {
            return std::unexpected(DomainError{
                "finance_bl_implied_returns",
                "expected finance_bl_implied_returns(cov, w_mkt, delta)"});
        }
        auto pi = eval_finance_bl_implied_returns(*cov, *w_mkt, delta);
        if (!pi) {
            return std::unexpected(pi.error());
        }
        result = *pi;
    }

    return result;
}

void ms_register_matrix_call_finance_bl_implied_returns() {
    register_matrix_call("finance_bl_implied_returns", &handle_finance_bl_implied_returns);
}

} // namespace ms::interp
