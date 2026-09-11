// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_geo_catmull_rom(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if ((assign.callee == "geo_bezier_eval" || assign.callee == "geo_bezier_deriv" ||
                assign.callee == "geo_catmull_rom") &&
               assign.args.size() == 2) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        double t = 0.0;
        if (!parse_number(assign.args[1], t)) {
            auto t_expr = eval_scalar_expr(ctx.state(), assign.args[1]);
            if (!t_expr) {
                return std::unexpected(DomainError{
                    assign.callee,
                    assign.callee == "geo_bezier_eval"
                        ? "expected geo_bezier_eval(ctrl, t)"
                        : (assign.callee == "geo_bezier_deriv"
                               ? "expected geo_bezier_deriv(ctrl, t)"
                               : "expected geo_catmull_rom(ctrl, t)")});
            }
            t = *t_expr;
        }
        if (assign.callee == "geo_bezier_eval") {
            result = eval_geo_bezier_eval(*matrix, t);
        } else if (assign.callee == "geo_bezier_deriv") {
            result = eval_geo_bezier_deriv(*matrix, t);
        } else {
            result = eval_geo_catmull_rom(*matrix, t);
        }
    }

    return result;
}

void ms_register_matrix_call_geo_catmull_rom() {
    register_matrix_call("geo_catmull_rom", &handle_geo_catmull_rom);
}

} // namespace ms::interp
