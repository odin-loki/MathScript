#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_control_step_response(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "control_step_response" &&
               (assign.args.size() == 2 || assign.args.size() == 3 || assign.args.size() == 4)) {
        auto num_m = ctx.resolve_operand(assign.args[0]);
        if (!num_m) {
            return std::unexpected(num_m.error());
        }
        auto den_m = ctx.resolve_operand(assign.args[1]);
        if (!den_m) {
            return std::unexpected(den_m.error());
        }
        auto t_end = ctx.parse_scalar_arg(assign.args.size() >= 3 ? assign.args[2] : "10",
                                      "control_step_response");
        if (!t_end) {
            return std::unexpected(t_end.error());
        }
        int n_pts = 500;
        if (assign.args.size() == 4) {
            auto n_pts_val = ctx.parse_scalar_arg(assign.args[3], "control_step_response");
            if (!n_pts_val) {
                return std::unexpected(n_pts_val.error());
            }
            if (*n_pts_val < 2.0 || std::floor(*n_pts_val) != *n_pts_val) {
                return std::unexpected(
                    DomainError{"control_step_response", "expected integer n_pts >= 2"});
            }
            n_pts = static_cast<int>(*n_pts_val);
        }
        result = eval_control_step_response(*num_m, *den_m, *t_end, n_pts);
    }

    return result;
}

void ms_register_matrix_call_control_step_response() {
    register_matrix_call("control_step_response", &handle_control_step_response);
}

} // namespace ms::interp
