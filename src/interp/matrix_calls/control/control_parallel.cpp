#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_control_parallel(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "control_parallel" && assign.args.size() == 4) {
        auto num1 = ctx.resolve_operand(assign.args[0]);
        if (!num1) {
            return std::unexpected(num1.error());
        }
        auto den1 = ctx.resolve_operand(assign.args[1]);
        if (!den1) {
            return std::unexpected(den1.error());
        }
        auto num2 = ctx.resolve_operand(assign.args[2]);
        if (!num2) {
            return std::unexpected(num2.error());
        }
        auto den2 = ctx.resolve_operand(assign.args[3]);
        if (!den2) {
            return std::unexpected(den2.error());
        }
        result = eval_control_parallel(*num1, *den1, *num2, *den2);
    }

    return result;
}

void ms_register_matrix_call_control_parallel() {
    register_matrix_call("control_parallel", &handle_control_parallel);
}

} // namespace ms::interp
