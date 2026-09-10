// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_poly_fit(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "poly_fit" && assign.args.size() == 3) {
        auto xs = ctx.resolve_operand(assign.args[0]);
        if (!xs) {
            return std::unexpected(xs.error());
        }
        auto ys = ctx.resolve_operand(assign.args[1]);
        if (!ys) {
            return std::unexpected(ys.error());
        }
        double degree_d = 0.0;
        if (!parse_number(assign.args[2], degree_d)) {
            auto degree_expr = eval_scalar_expr(ctx.state(), assign.args[2]);
            if (!degree_expr) {
                return std::unexpected(
                    DomainError{"poly_fit", "expected poly_fit(xs, ys, degree)"});
            }
            degree_d = *degree_expr;
        }
        const int degree = static_cast<int>(degree_d);
        if (degree_d != degree) {
            return std::unexpected(DomainError{"poly_fit", "expected integer degree"});
        }
        auto coeffs = eval_poly_fit(*xs, *ys, degree);
        if (!coeffs) {
            return std::unexpected(coeffs.error());
        }
        result = *coeffs;
    }

    return result;
}

void ms_register_matrix_call_poly_fit() {
    register_matrix_call("poly_fit", &handle_poly_fit);
}

} // namespace ms::interp
