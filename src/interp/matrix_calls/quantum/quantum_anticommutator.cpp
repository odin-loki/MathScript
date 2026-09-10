#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_quantum_anticommutator(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "quantum_anticommutator" && assign.args.size() == 2) {
        auto A = ctx.resolve_operand(assign.args[0]);
        if (!A) {
            return std::unexpected(A.error());
        }
        auto B = ctx.resolve_operand(assign.args[1]);
        if (!B) {
            return std::unexpected(B.error());
        }
        result = eval_quantum_anticommutator(*A, *B);
    }

    return result;
}

void ms_register_matrix_call_quantum_anticommutator() {
    register_matrix_call("quantum_anticommutator", &handle_quantum_anticommutator);
}

} // namespace ms::interp
