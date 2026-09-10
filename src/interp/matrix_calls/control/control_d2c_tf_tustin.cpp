#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_control_d2c_tf_tustin(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if ((assign.callee == "control_c2d_tf_tustin" ||
                assign.callee == "control_d2c_tf_tustin") &&
               assign.args.size() == 3) {
        auto num_m = ctx.resolve_operand(assign.args[0]);
        if (!num_m) {
            return std::unexpected(num_m.error());
        }
        auto den_m = ctx.resolve_operand(assign.args[1]);
        if (!den_m) {
            return std::unexpected(den_m.error());
        }
        double Ts = 0.0;
        if (!parse_number(assign.args[2], Ts)) {
            auto ts_expr = eval_scalar_expr(ctx.state(), assign.args[2]);
            if (!ts_expr) {
                return std::unexpected(DomainError{assign.callee, "expected positive Ts"});
            }
            Ts = *ts_expr;
        }
        const char* fn = assign.callee.c_str();
        if (assign.callee == "control_c2d_tf_tustin") {
            result = eval_control_c2d_tf(*num_m, *den_m, Ts, control::DiscretizationMethod::Tustin,
                                         fn);
        } else {
            result = eval_control_d2c_tf(*num_m, *den_m, Ts, control::DiscretizationMethod::Tustin,
                                         fn);
        }
    }

    return result;
}

void ms_register_matrix_call_control_d2c_tf_tustin() {
    register_matrix_call("control_d2c_tf_tustin", &handle_control_d2c_tf_tustin);
}

} // namespace ms::interp
