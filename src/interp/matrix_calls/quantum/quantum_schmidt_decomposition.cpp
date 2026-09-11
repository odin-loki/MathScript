// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_quantum_schmidt_decomposition(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "quantum_schmidt_decomposition" && assign.args.size() == 3) {
        auto psi = ctx.resolve_operand(assign.args[0]);
        if (!psi) {
            return std::unexpected(psi.error());
        }
        auto dim_a_val = ctx.parse_scalar_arg(assign.args[1], "quantum_schmidt_decomposition");
        if (!dim_a_val) {
            return std::unexpected(dim_a_val.error());
        }
        auto dim_b_val = ctx.parse_scalar_arg(assign.args[2], "quantum_schmidt_decomposition");
        if (!dim_b_val) {
            return std::unexpected(dim_b_val.error());
        }
        const int dim_a = static_cast<int>(*dim_a_val);
        const int dim_b = static_cast<int>(*dim_b_val);
        if (dim_a < 1 || dim_b < 1 || *dim_a_val != dim_a || *dim_b_val != dim_b) {
            return std::unexpected(DomainError{
                "quantum_schmidt_decomposition", "expected positive integer dim_a and dim_b"});
        }
        result = eval_quantum_schmidt_decomposition(*psi, dim_a, dim_b);
    }

    return result;
}

void ms_register_matrix_call_quantum_schmidt_decomposition() {
    register_matrix_call("quantum_schmidt_decomposition", &handle_quantum_schmidt_decomposition);
}

} // namespace ms::interp
