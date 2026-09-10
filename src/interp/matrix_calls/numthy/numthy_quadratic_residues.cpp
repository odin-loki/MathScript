// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_numthy_quadratic_residues(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "numthy_quadratic_residues" && assign.args.size() == 1) {
        double n_d = 0.0;
        if (!parse_number(assign.args[0], n_d)) {
            auto n_expr = eval_scalar_expr(ctx.state(), assign.args[0]);
            if (!n_expr) {
                return std::unexpected(
                    DomainError{"numthy_quadratic_residues", "expected numthy_quadratic_residues(p)"});
            }
            n_d = *n_expr;
        }
        const int n = static_cast<int>(n_d);
        if (n < 3 || n_d != n) {
            return std::unexpected(
                DomainError{"numthy_quadratic_residues", "expected odd prime p >= 3"});
        }
        result = eval_numthy_quadratic_residues(n);
    }

    return result;
}

void ms_register_matrix_call_numthy_quadratic_residues() {
    register_matrix_call("numthy_quadratic_residues", &handle_numthy_quadratic_residues);
}

} // namespace ms::interp
