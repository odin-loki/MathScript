// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_cfd_upwind_step_3d(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "cfd_upwind_step_3d" &&
               (assign.args.size() == 6 || assign.args.size() == 9)) {
        auto grid = ctx.resolve_operand(assign.args[0]);
        if (!grid) {
            return std::unexpected(grid.error());
        }
        auto u = ctx.resolve_operand(assign.args[1]);
        if (!u) {
            return std::unexpected(u.error());
        }
        auto vx = ctx.parse_scalar_arg(assign.args[2], "cfd_upwind_step_3d");
        if (!vx) {
            return std::unexpected(vx.error());
        }
        auto vy = ctx.parse_scalar_arg(assign.args[3], "cfd_upwind_step_3d");
        if (!vy) {
            return std::unexpected(vy.error());
        }
        auto vz = ctx.parse_scalar_arg(assign.args[4], "cfd_upwind_step_3d");
        if (!vz) {
            return std::unexpected(vz.error());
        }
        auto dt = ctx.parse_scalar_arg(assign.args[5], "cfd_upwind_step_3d");
        if (!dt) {
            return std::unexpected(dt.error());
        }
        cfd::BoundaryCondition bc_x = cfd::BoundaryCondition::Periodic;
        cfd::BoundaryCondition bc_y = cfd::BoundaryCondition::Periodic;
        cfd::BoundaryCondition bc_z = cfd::BoundaryCondition::Periodic;
        if (assign.args.size() == 9) {
            auto bcx = ctx.parse_scalar_arg(assign.args[6], "cfd_upwind_step_3d");
            if (!bcx) {
                return std::unexpected(bcx.error());
            }
            auto bcy = ctx.parse_scalar_arg(assign.args[7], "cfd_upwind_step_3d");
            if (!bcy) {
                return std::unexpected(bcy.error());
            }
            auto bcz = ctx.parse_scalar_arg(assign.args[8], "cfd_upwind_step_3d");
            if (!bcz) {
                return std::unexpected(bcz.error());
            }
            auto parsed_x = parse_cfd_bc(*bcx, "cfd_upwind_step_3d");
            if (!parsed_x) {
                return std::unexpected(parsed_x.error());
            }
            auto parsed_y = parse_cfd_bc(*bcy, "cfd_upwind_step_3d");
            if (!parsed_y) {
                return std::unexpected(parsed_y.error());
            }
            auto parsed_z = parse_cfd_bc(*bcz, "cfd_upwind_step_3d");
            if (!parsed_z) {
                return std::unexpected(parsed_z.error());
            }
            bc_x = *parsed_x;
            bc_y = *parsed_y;
            bc_z = *parsed_z;
        }
        result = eval_cfd_upwind_step_3d(*grid, *u, *vx, *vy, *vz, *dt, bc_x, bc_y, bc_z);
    }

    return result;
}

void ms_register_matrix_call_cfd_upwind_step_3d() {
    register_matrix_call("cfd_upwind_step_3d", &handle_cfd_upwind_step_3d);
}

} // namespace ms::interp
