// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_cfd_upwind_step_1d(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "cfd_upwind_step_1d" &&
               (assign.args.size() == 4 || assign.args.size() == 5)) {
        auto grid = ctx.resolve_operand(assign.args[0]);
        if (!grid) {
            return std::unexpected(grid.error());
        }
        auto u = ctx.resolve_operand(assign.args[1]);
        if (!u) {
            return std::unexpected(u.error());
        }
        auto v = ctx.parse_scalar_arg(assign.args[2], "cfd_upwind_step_1d");
        if (!v) {
            return std::unexpected(v.error());
        }
        auto dt = ctx.parse_scalar_arg(assign.args[3], "cfd_upwind_step_1d");
        if (!dt) {
            return std::unexpected(dt.error());
        }
        cfd::BoundaryCondition bc = cfd::BoundaryCondition::Periodic;
        if (assign.args.size() == 5) {
            auto bc_val = ctx.parse_scalar_arg(assign.args[4], "cfd_upwind_step_1d");
            if (!bc_val) {
                return std::unexpected(bc_val.error());
            }
            auto parsed_bc = parse_cfd_bc(*bc_val, "cfd_upwind_step_1d");
            if (!parsed_bc) {
                return std::unexpected(parsed_bc.error());
            }
            bc = *parsed_bc;
        }
        result = eval_cfd_upwind_step_1d(*grid, *u, *v, *dt, bc);
    }

    return result;
}

void ms_register_matrix_call_cfd_upwind_step_1d() {
    register_matrix_call("cfd_upwind_step_1d", &handle_cfd_upwind_step_1d);
}

} // namespace ms::interp
