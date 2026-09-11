// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_precond_ssor(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "precond_ssor" &&
               (assign.args.size() == 1 || assign.args.size() == 2)) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        double omega = 1.0;
        if (assign.args.size() == 2) {
            if (!parse_number(assign.args[1], omega)) {
                auto it = ctx.state().scalars.find(assign.args[1]);
                if (it != ctx.state().scalars.end()) {
                    omega = it->second;
                } else {
                    return std::unexpected(
                        DomainError{"precond_ssor", "expected precond_ssor(A[, omega])"});
                }
            }
        }
        result = precond_ssor(*matrix, omega);
    }

    return result;
}

void ms_register_matrix_call_precond_ssor() {
    register_matrix_call("precond_ssor", &handle_precond_ssor);
}

} // namespace ms::interp
