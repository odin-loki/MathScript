// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include <cmath>
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_numthy_farey(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "numthy_farey" && assign.args.size() == 1) {
        double n_d = 0.0;
        if (!parse_number(assign.args[0], n_d)) {
            auto n_expr = eval_scalar_expr(ctx.state(), assign.args[0]);
            if (!n_expr) {
                return std::unexpected(
                    DomainError{"numthy_farey", "expected numthy_farey(n)"});
            }
            n_d = *n_expr;
        }
        // The order is range-checked on the double: `static_cast<int>` of one outside
        // int's range is undefined behaviour rather than a wrap, so `n < 1` below was
        // testing a value the program was not entitled to have. `eval_numthy_farey`
        // owns the length check, because |F_n| is quadratic in n and only it can count.
        if (!std::isfinite(n_d) || n_d != std::floor(n_d) || n_d < 1.0 ||
            n_d > 2147483647.0) {
            return std::unexpected(
                DomainError{"numthy_farey", "expected positive integer n"});
        }
        result = eval_numthy_farey(static_cast<int>(n_d));
    }

    return result;
}

void ms_register_matrix_call_numthy_farey() {
    register_matrix_call("numthy_farey", &handle_numthy_farey);
}

} // namespace ms::interp
