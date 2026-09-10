#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_quantum_op_apply(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "quantum_op_apply" && assign.args.size() == 2) {
        auto op = ctx.resolve_operand(assign.args[0]);
        if (!op) {
            return std::unexpected(op.error());
        }
        auto psi = ctx.resolve_operand(assign.args[1]);
        if (!psi) {
            return std::unexpected(psi.error());
        }
        result = eval_quantum_op_apply(*op, *psi);
    }

    return result;
}

void ms_register_matrix_call_quantum_op_apply() {
    register_matrix_call("quantum_op_apply", &handle_quantum_op_apply);
}

} // namespace ms::interp
