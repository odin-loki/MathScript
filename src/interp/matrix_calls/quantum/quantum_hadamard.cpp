// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_quantum_hadamard(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "quantum_hadamard" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto psi = matrix_to_ket2(*matrix, "quantum_hadamard");
        if (!psi) {
            return std::unexpected(psi.error());
        }
        const auto out = quantum::op_apply(quantum::hadamard(), *psi);
        Matrix<double> col(2, 1);
        col(0, 0) = out[0].real();
        col(1, 0) = out[1].real();
        result = col;
    }

    return result;
}

void ms_register_matrix_call_quantum_hadamard() {
    register_matrix_call("quantum_hadamard", &handle_quantum_hadamard);
}

} // namespace ms::interp
