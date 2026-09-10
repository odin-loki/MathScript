#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_stats_pacf(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "stats_pacf" && assign.args.size() == 2) {
        auto x = ctx.resolve_operand(assign.args[0]);
        if (!x) {
            return std::unexpected(x.error());
        }
        double max_lag_d = 0.0;
        if (!parse_number(assign.args[1], max_lag_d)) {
            return std::unexpected(
                DomainError{"stats_pacf", "expected stats_pacf(x, max_lag)"});
        }
        const int max_lag = static_cast<int>(max_lag_d);
        if (max_lag < 0 || max_lag_d != max_lag) {
            return std::unexpected(
                DomainError{"stats_pacf", "expected non-negative integer max_lag"});
        }
        result = eval_stats_pacf(*x, max_lag);
    }

    return result;
}

void ms_register_matrix_call_stats_pacf() {
    register_matrix_call("stats_pacf", &handle_stats_pacf);
}

} // namespace ms::interp
