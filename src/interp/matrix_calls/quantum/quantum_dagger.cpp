// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_quantum_dagger(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "quantum_dagger" && assign.args.size() == 1) {
        auto op = ctx.resolve_operand(assign.args[0]);
        if (!op) {
            return std::unexpected(op.error());
        }
        result = eval_quantum_dagger(*op);
    }

    return result;
}

void ms_register_matrix_call_quantum_dagger() {
    register_matrix_call("quantum_dagger", &handle_quantum_dagger);
}

} // namespace ms::interp
