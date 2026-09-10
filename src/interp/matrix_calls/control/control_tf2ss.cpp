#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_control_tf2ss(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "control_tf2ss" && assign.args.size() == 2) {
        auto num_m = ctx.resolve_operand(assign.args[0]);
        if (!num_m) {
            return std::unexpected(num_m.error());
        }
        auto den_m = ctx.resolve_operand(assign.args[1]);
        if (!den_m) {
            return std::unexpected(den_m.error());
        }
        auto ss = eval_control_tf2ss(*num_m, *den_m);
        if (!ss) {
            return std::unexpected(ss.error());
        }
        result = *ss;
    }

    return result;
}

void ms_register_matrix_call_control_tf2ss() {
    register_matrix_call("control_tf2ss", &handle_control_tf2ss);
}

} // namespace ms::interp
