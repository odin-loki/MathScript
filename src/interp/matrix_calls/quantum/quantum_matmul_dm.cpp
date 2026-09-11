// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_quantum_matmul_dm(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "quantum_matmul_dm" && assign.args.size() == 2) {
        auto A = ctx.resolve_operand(assign.args[0]);
        if (!A) {
            return std::unexpected(A.error());
        }
        auto B = ctx.resolve_operand(assign.args[1]);
        if (!B) {
            return std::unexpected(B.error());
        }
        result = eval_quantum_matmul_dm(*A, *B);
    }

    return result;
}

void ms_register_matrix_call_quantum_matmul_dm() {
    register_matrix_call("quantum_matmul_dm", &handle_quantum_matmul_dm);
}

} // namespace ms::interp
