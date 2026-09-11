// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_quantum_time_evolve_psi(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "quantum_time_evolve_psi" && assign.args.size() == 3) {
        auto H = ctx.resolve_operand(assign.args[0]);
        if (!H) {
            return std::unexpected(H.error());
        }
        auto psi = ctx.resolve_operand(assign.args[1]);
        if (!psi) {
            return std::unexpected(psi.error());
        }
        auto t = ctx.parse_scalar_arg(assign.args[2], "quantum_time_evolve_psi");
        if (!t) {
            return std::unexpected(t.error());
        }
        result = eval_quantum_time_evolve_psi(*H, *psi, *t);
    }

    return result;
}

void ms_register_matrix_call_quantum_time_evolve_psi() {
    register_matrix_call("quantum_time_evolve_psi", &handle_quantum_time_evolve_psi);
}

} // namespace ms::interp
