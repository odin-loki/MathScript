// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_diffgeo_surface_normal_sphere(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "diffgeo_surface_normal_sphere" && assign.args.size() == 2) {
        double u = 0.0;
        double v = 0.0;
        if (!parse_number(assign.args[0], u)) {
            auto u_expr = eval_scalar_expr(ctx.state(), assign.args[0]);
            if (!u_expr) {
                return std::unexpected(u_expr.error());
            }
            u = *u_expr;
        }
        if (!parse_number(assign.args[1], v)) {
            auto v_expr = eval_scalar_expr(ctx.state(), assign.args[1]);
            if (!v_expr) {
                return std::unexpected(v_expr.error());
            }
            v = *v_expr;
        }
        auto normal = eval_diffgeo_surface_normal_sphere(u, v);
        if (!normal) {
            return std::unexpected(normal.error());
        }
        result = *normal;
    }

    return result;
}

void ms_register_matrix_call_diffgeo_surface_normal_sphere() {
    register_matrix_call("diffgeo_surface_normal_sphere", &handle_diffgeo_surface_normal_sphere);
}

} // namespace ms::interp
