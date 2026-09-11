// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_numthy_pell_solve(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "numthy_pell_solve" && assign.args.size() == 1) {
        double n_d = 0.0;
        if (!parse_number(assign.args[0], n_d)) {
            auto n_expr = eval_scalar_expr(ctx.state(), assign.args[0]);
            if (!n_expr) {
                return std::unexpected(
                    DomainError{"numthy_pell_solve", "expected numthy_pell_solve(D)"});
            }
            n_d = *n_expr;
        }
        const int n = static_cast<int>(n_d);
        if (n < 1 || n_d != n) {
            return std::unexpected(
                DomainError{"numthy_pell_solve", "expected positive integer D"});
        }
        result = eval_numthy_pell_solve(n);
    }

    return result;
}

void ms_register_matrix_call_numthy_pell_solve() {
    register_matrix_call("numthy_pell_solve", &handle_numthy_pell_solve);
}

} // namespace ms::interp
