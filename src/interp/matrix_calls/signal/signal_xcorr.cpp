#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_signal_xcorr(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if ((assign.callee == "signal_xcorr" || assign.callee == "signal_xcov") &&
               assign.args.size() == 3) {
        auto a = ctx.resolve_operand(assign.args[0]);
        if (!a) {
            return std::unexpected(a.error());
        }
        auto b = ctx.resolve_operand(assign.args[1]);
        if (!b) {
            return std::unexpected(b.error());
        }
        auto max_lag_val = ctx.parse_scalar_arg(assign.args[2], assign.callee.c_str());
        if (!max_lag_val) {
            return std::unexpected(max_lag_val.error());
        }
        const int max_lag = static_cast<int>(*max_lag_val);
        if (max_lag < 0 || *max_lag_val != max_lag) {
            return std::unexpected(
                DomainError{assign.callee, "expected non-negative integer max_lag"});
        }
        if (assign.callee == "signal_xcorr") {
            result = eval_signal_xcorr(*a, *b, max_lag);
        } else {
            result = eval_signal_xcov(*a, *b, max_lag);
        }
    }

    return result;
}

void ms_register_matrix_call_signal_xcorr() {
    register_matrix_call("signal_xcorr", &handle_signal_xcorr);
}

} // namespace ms::interp
