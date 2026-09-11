// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_combo_gray_code(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "combo_gray_code" && assign.args.size() == 1) {
        double n_d = 0.0;
        if (!parse_number(assign.args[0], n_d)) {
            auto n_expr = eval_scalar_expr(ctx.state(), assign.args[0]);
            if (!n_expr) {
                return std::unexpected(n_expr.error());
            }
            n_d = *n_expr;
        }
        const int n = static_cast<int>(n_d);
        if (n < 0 || n_d != n) {
            return std::unexpected(
                DomainError{"combo_gray_code", "expected non-negative integer n"});
        }
        auto codes = eval_combo_gray_code(n);
        if (!codes) {
            return std::unexpected(codes.error());
        }
        result = *codes;
    }

    return result;
}

void ms_register_matrix_call_combo_gray_code() {
    register_matrix_call("combo_gray_code", &handle_combo_gray_code);
}

} // namespace ms::interp
