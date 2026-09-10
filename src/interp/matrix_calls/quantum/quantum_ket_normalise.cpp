#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_quantum_ket_normalise(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "quantum_ket_normalise" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        result = eval_quantum_ket_normalise_matrix(*matrix);
    }

    return result;
}

void ms_register_matrix_call_quantum_ket_normalise() {
    register_matrix_call("quantum_ket_normalise", &handle_quantum_ket_normalise);
}

} // namespace ms::interp
