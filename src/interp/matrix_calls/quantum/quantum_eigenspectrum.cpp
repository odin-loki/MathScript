#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_quantum_eigenspectrum(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "quantum_eigenspectrum" && assign.args.size() == 1) {
        auto H = ctx.resolve_operand(assign.args[0]);
        if (!H) {
            return std::unexpected(H.error());
        }
        result = eval_quantum_eigenspectrum(*H);
    }

    return result;
}

void ms_register_matrix_call_quantum_eigenspectrum() {
    register_matrix_call("quantum_eigenspectrum", &handle_quantum_eigenspectrum);
}

} // namespace ms::interp
