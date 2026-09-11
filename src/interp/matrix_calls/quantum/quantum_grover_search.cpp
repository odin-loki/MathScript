// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_quantum_grover_search(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "quantum_grover_search" &&
               (assign.args.size() == 2 || assign.args.size() == 3)) {
        auto n_qubits_val = ctx.parse_scalar_arg(assign.args[0], "quantum_grover_search");
        if (!n_qubits_val) {
            return std::unexpected(n_qubits_val.error());
        }
        const int n_qubits = static_cast<int>(*n_qubits_val);
        if (n_qubits < 1 || *n_qubits_val != n_qubits) {
            return std::unexpected(DomainError{
                "quantum_grover_search", "expected positive integer n_qubits"});
        }
        auto marked_m = ctx.resolve_operand(assign.args[1]);
        if (!marked_m) {
            return std::unexpected(marked_m.error());
        }
        auto marked_indices = matrix_to_int_coeff_vector(*marked_m, "quantum_grover_search");
        if (!marked_indices) {
            return std::unexpected(marked_indices.error());
        }
        int n_iterations = quantum::grover_optimal_iterations(
            n_qubits, static_cast<int>(marked_indices->size()));
        if (assign.args.size() == 3) {
            auto iter_val = ctx.parse_scalar_arg(assign.args[2], "quantum_grover_search");
            if (!iter_val) {
                return std::unexpected(iter_val.error());
            }
            n_iterations = static_cast<int>(*iter_val);
            if (n_iterations < 0 || *iter_val != n_iterations) {
                return std::unexpected(DomainError{
                    "quantum_grover_search", "expected non-negative integer n_iterations"});
            }
        }
        result = eval_quantum_grover_search(n_qubits, *marked_m, n_iterations);
    }

    return result;
}

void ms_register_matrix_call_quantum_grover_search() {
    register_matrix_call("quantum_grover_search", &handle_quantum_grover_search);
}

} // namespace ms::interp
