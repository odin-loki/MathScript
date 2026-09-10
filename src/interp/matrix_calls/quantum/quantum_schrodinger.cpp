// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_quantum_schrodinger(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "quantum_schrodinger" && assign.args.size() == 5) {
        auto H_m = ctx.resolve_operand(assign.args[0]);
        if (!H_m) {
            return std::unexpected(H_m.error());
        }
        auto psi0_m = ctx.resolve_operand(assign.args[1]);
        if (!psi0_m) {
            return std::unexpected(psi0_m.error());
        }
        double t0 = 0.0;
        double t1 = 0.0;
        double n_steps_d = 0.0;
        if (!parse_number(assign.args[2], t0) || !parse_number(assign.args[3], t1) ||
            !parse_number(assign.args[4], n_steps_d)) {
            return std::unexpected(DomainError{
                "quantum_schrodinger",
                "expected quantum_schrodinger(H, psi0, t0, t1, n_steps)"});
        }
        const int n_steps = static_cast<int>(n_steps_d);
        if (n_steps < 0 || n_steps_d != n_steps) {
            return std::unexpected(DomainError{
                "quantum_schrodinger", "expected non-negative integer n_steps"});
        }
        result = eval_quantum_schrodinger_matrix(*H_m, *psi0_m, t0, t1, n_steps);
    }

    return result;
}

void ms_register_matrix_call_quantum_schrodinger() {
    register_matrix_call("quantum_schrodinger", &handle_quantum_schrodinger);
}

} // namespace ms::interp
