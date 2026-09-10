#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_signal_lms_weights(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if ((assign.callee == "signal_lms" || assign.callee == "signal_lms_weights") &&
               assign.args.size() == 4) {
        auto x = ctx.resolve_operand(assign.args[0]);
        if (!x) {
            return std::unexpected(x.error());
        }
        auto d = ctx.resolve_operand(assign.args[1]);
        if (!d) {
            return std::unexpected(d.error());
        }
        auto filter_length_val = ctx.parse_scalar_arg(assign.args[2], assign.callee.c_str());
        if (!filter_length_val) {
            return std::unexpected(filter_length_val.error());
        }
        auto mu_val = ctx.parse_scalar_arg(assign.args[3], assign.callee.c_str());
        if (!mu_val) {
            return std::unexpected(mu_val.error());
        }
        if (assign.callee == "signal_lms") {
            result = eval_signal_lms(*x, *d, *filter_length_val, *mu_val);
        } else {
            result = eval_signal_lms_weights(*x, *d, *filter_length_val, *mu_val);
        }
    }

    return result;
}

void ms_register_matrix_call_signal_lms_weights() {
    register_matrix_call("signal_lms_weights", &handle_signal_lms_weights);
}

} // namespace ms::interp
