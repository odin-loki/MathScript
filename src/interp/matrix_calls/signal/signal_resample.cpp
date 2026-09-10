#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_signal_resample(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "signal_resample" && assign.args.size() == 3) {
        auto x = ctx.resolve_operand(assign.args[0]);
        if (!x) {
            return std::unexpected(x.error());
        }
        auto p_val = ctx.parse_scalar_arg(assign.args[1], "signal_resample");
        if (!p_val) {
            return std::unexpected(p_val.error());
        }
        auto q_val = ctx.parse_scalar_arg(assign.args[2], "signal_resample");
        if (!q_val) {
            return std::unexpected(q_val.error());
        }
        result = eval_signal_resample_pq(*x, *p_val, *q_val);
    }

    return result;
}

void ms_register_matrix_call_signal_resample() {
    register_matrix_call("signal_resample", &handle_signal_resample);
}

} // namespace ms::interp
