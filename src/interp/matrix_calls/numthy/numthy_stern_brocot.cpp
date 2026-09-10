// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_numthy_stern_brocot(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "numthy_stern_brocot" && assign.args.size() == 1) {
        double n_d = 0.0;
        if (!parse_number(assign.args[0], n_d)) {
            auto n_expr = eval_scalar_expr(ctx.state(), assign.args[0]);
            if (!n_expr) {
                return std::unexpected(
                    DomainError{"numthy_stern_brocot", "expected numthy_stern_brocot(n)"});
            }
            n_d = *n_expr;
        }
        const int n = static_cast<int>(n_d);
        if (n < 0 || n_d != n) {
            return std::unexpected(
                DomainError{"numthy_stern_brocot", "expected non-negative integer n"});
        }
        result = eval_numthy_stern_brocot(n);
    }

    return result;
}

void ms_register_matrix_call_numthy_stern_brocot() {
    register_matrix_call("numthy_stern_brocot", &handle_numthy_stern_brocot);
}

} // namespace ms::interp
