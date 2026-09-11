// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_quantum_fock_state(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "quantum_fock_state" && assign.args.size() == 2) {
        double n_d = 0.0;
        double n_max_d = 0.0;
        if (!parse_number(assign.args[0], n_d)) {
            auto it = ctx.state().scalars.find(assign.args[0]);
            if (it != ctx.state().scalars.end()) {
                n_d = it->second;
            } else {
                return std::unexpected(
                    DomainError{assign.callee, "expected numeric n argument"});
            }
        }
        if (!parse_number(assign.args[1], n_max_d)) {
            auto it = ctx.state().scalars.find(assign.args[1]);
            if (it != ctx.state().scalars.end()) {
                n_max_d = it->second;
            } else {
                return std::unexpected(
                    DomainError{assign.callee, "expected numeric n_max argument"});
            }
        }
        const int n = static_cast<int>(n_d);
        const int n_max = static_cast<int>(n_max_d);
        if (n_max < 0 || n_d != n || n_max_d != n_max) {
            return std::unexpected(DomainError{
                assign.callee, "expected non-negative integer n and n_max"});
        }
        result = eval_quantum_fock_state(n, n_max);
    }

    return result;
}

void ms_register_matrix_call_quantum_fock_state() {
    register_matrix_call("quantum_fock_state", &handle_quantum_fock_state);
}

} // namespace ms::interp
