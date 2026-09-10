#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_control_obsv(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "control_obsv" && assign.args.size() == 2) {
        auto A_m = ctx.resolve_operand(assign.args[0]);
        if (!A_m) {
            return std::unexpected(A_m.error());
        }
        auto C_m = ctx.resolve_operand(assign.args[1]);
        if (!C_m) {
            return std::unexpected(C_m.error());
        }
        auto value = eval_control_obsv(*A_m, *C_m);
        if (!value) {
            return std::unexpected(value.error());
        }
        result = *value;
    }

    return result;
}

void ms_register_matrix_call_control_obsv() {
    register_matrix_call("control_obsv", &handle_control_obsv);
}

} // namespace ms::interp
