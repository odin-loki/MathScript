#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_control_bode(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "control_bode" && assign.args.size() == 3) {
        auto num_m = ctx.resolve_operand(assign.args[0]);
        if (!num_m) {
            return std::unexpected(num_m.error());
        }
        auto den_m = ctx.resolve_operand(assign.args[1]);
        if (!den_m) {
            return std::unexpected(den_m.error());
        }
        auto w_val = ctx.parse_scalar_arg(assign.args[2], "control_bode");
        if (!w_val) {
            return std::unexpected(w_val.error());
        }
        auto bode = eval_control_bode(*num_m, *den_m, *w_val);
        if (!bode) {
            return std::unexpected(bode.error());
        }
        result = *bode;
    }

    return result;
}

void ms_register_matrix_call_control_bode() {
    register_matrix_call("control_bode", &handle_control_bode);
}

} // namespace ms::interp
