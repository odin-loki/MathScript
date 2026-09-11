// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_signal_autocorr(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "signal_autocorr" && assign.args.size() == 2) {
        auto x = ctx.resolve_operand(assign.args[0]);
        if (!x) {
            return std::unexpected(x.error());
        }
        auto max_lag_val = ctx.parse_scalar_arg(assign.args[1], "signal_autocorr");
        if (!max_lag_val) {
            return std::unexpected(max_lag_val.error());
        }
        const int max_lag = static_cast<int>(*max_lag_val);
        if (max_lag < 0 || *max_lag_val != max_lag) {
            return std::unexpected(
                DomainError{"signal_autocorr", "expected non-negative integer max_lag"});
        }
        result = eval_signal_autocorr(*x, max_lag);
    }

    return result;
}

void ms_register_matrix_call_signal_autocorr() {
    register_matrix_call("signal_autocorr", &handle_signal_autocorr);
}

} // namespace ms::interp
