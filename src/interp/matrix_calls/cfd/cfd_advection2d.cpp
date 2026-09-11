// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_cfd_advection2d(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "cfd_advection2d" && assign.args.size() == 6) {
        auto nx_val = ctx.parse_scalar_arg(assign.args[0], "cfd_advection2d");
        if (!nx_val) {
            return std::unexpected(nx_val.error());
        }
        auto ny_val = ctx.parse_scalar_arg(assign.args[1], "cfd_advection2d");
        if (!ny_val) {
            return std::unexpected(ny_val.error());
        }
        auto vx_val = ctx.parse_scalar_arg(assign.args[2], "cfd_advection2d");
        if (!vx_val) {
            return std::unexpected(vx_val.error());
        }
        auto vy_val = ctx.parse_scalar_arg(assign.args[3], "cfd_advection2d");
        if (!vy_val) {
            return std::unexpected(vy_val.error());
        }
        auto t_end_val = ctx.parse_scalar_arg(assign.args[4], "cfd_advection2d");
        if (!t_end_val) {
            return std::unexpected(t_end_val.error());
        }
        auto dt_val = ctx.parse_scalar_arg(assign.args[5], "cfd_advection2d");
        if (!dt_val) {
            return std::unexpected(dt_val.error());
        }
        ExtentBudget budget("cfd_advection2d");
        auto nx = budget.take("nx", *nx_val);
        if (!nx) {
            return std::unexpected(nx.error());
        }
        auto ny = budget.take("ny", *ny_val);
        if (!ny) {
            return std::unexpected(ny.error());
        }
        result = eval_cfd_advection2d(*nx, *ny, *vx_val, *vy_val, *t_end_val, *dt_val);
    }

    return result;
}

void ms_register_matrix_call_cfd_advection2d() {
    register_matrix_call("cfd_advection2d", &handle_cfd_advection2d);
}

} // namespace ms::interp
