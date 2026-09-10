#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_quantum_ket_tensor_product(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "quantum_ket_tensor_product" && assign.args.size() == 2) {
        auto psi1 = ctx.resolve_operand(assign.args[0]);
        if (!psi1) {
            return std::unexpected(psi1.error());
        }
        auto psi2 = ctx.resolve_operand(assign.args[1]);
        if (!psi2) {
            return std::unexpected(psi2.error());
        }
        result = eval_quantum_ket_tensor_product(*psi1, *psi2);
    }

    return result;
}

void ms_register_matrix_call_quantum_ket_tensor_product() {
    register_matrix_call("quantum_ket_tensor_product", &handle_quantum_ket_tensor_product);
}

} // namespace ms::interp
