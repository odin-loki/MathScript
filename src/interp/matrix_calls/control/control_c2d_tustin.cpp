#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_control_c2d_tustin(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if ((assign.callee == "control_c2d_tustin" || assign.callee == "control_c2d_euler" ||
                assign.callee == "control_d2c_tustin" || assign.callee == "control_d2c_euler") &&
               assign.args.size() == 5) {
        auto A_m = ctx.resolve_operand(assign.args[0]);
        if (!A_m) {
            return std::unexpected(A_m.error());
        }
        auto B_m = ctx.resolve_operand(assign.args[1]);
        if (!B_m) {
            return std::unexpected(B_m.error());
        }
        auto C_m = ctx.resolve_operand(assign.args[2]);
        if (!C_m) {
            return std::unexpected(C_m.error());
        }
        auto D_m = ctx.resolve_operand(assign.args[3]);
        if (!D_m) {
            return std::unexpected(D_m.error());
        }
        double Ts = 0.0;
        if (!parse_number(assign.args[4], Ts)) {
            auto ts_expr = eval_scalar_expr(ctx.state(), assign.args[4]);
            if (!ts_expr) {
                return std::unexpected(DomainError{assign.callee, "expected positive Ts"});
            }
            Ts = *ts_expr;
        }
        const char* fn = assign.callee.c_str();
        const control::DiscretizationMethod method =
            (assign.callee == "control_c2d_tustin" || assign.callee == "control_d2c_tustin")
                ? control::DiscretizationMethod::Tustin
                : control::DiscretizationMethod::Euler;
        if (assign.callee == "control_c2d_tustin" || assign.callee == "control_c2d_euler") {
            result = eval_control_c2d(*A_m, *B_m, *C_m, *D_m, Ts, method, fn);
        } else {
            result = eval_control_d2c(*A_m, *B_m, *C_m, *D_m, Ts, method, fn);
        }
    }

    return result;
}

void ms_register_matrix_call_control_c2d_tustin() {
    register_matrix_call("control_c2d_tustin", &handle_control_c2d_tustin);
}

} // namespace ms::interp
