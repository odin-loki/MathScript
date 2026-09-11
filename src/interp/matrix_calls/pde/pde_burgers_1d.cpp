// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_pde_burgers_1d(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "pde_burgers_1d" && assign.args.size() == 5) {
        auto u0_m = ctx.resolve_operand(assign.args[0]);
        if (!u0_m) {
            return std::unexpected(u0_m.error());
        }
        auto nu = ctx.parse_scalar_arg(assign.args[1], "pde_burgers_1d");
        if (!nu) {
            return std::unexpected(nu.error());
        }
        auto dx = ctx.parse_scalar_arg(assign.args[2], "pde_burgers_1d");
        if (!dx) {
            return std::unexpected(dx.error());
        }
        auto dt = ctx.parse_scalar_arg(assign.args[3], "pde_burgers_1d");
        if (!dt) {
            return std::unexpected(dt.error());
        }
        auto steps_val = ctx.parse_scalar_arg(assign.args[4], "pde_burgers_1d");
        if (!steps_val) {
            return std::unexpected(steps_val.error());
        }
        WorkBudget budget(assign.callee, 25.0);
        // Charged before `steps` is read so the bound on it shrinks as the grid grows:
        // the solver keeps one grid per step and the REPL reads only the last.
        budget.charge(u0_m->rows() * u0_m->cols());
        auto steps_arg = budget.take("steps", *steps_val);
        if (!steps_arg) {
            return std::unexpected(steps_arg.error());
        }
        const int steps_i = *steps_arg;
        result = eval_pde_burgers_1d(*u0_m, *nu, *dx, *dt, static_cast<std::size_t>(steps_i));
    }

    return result;
}

void ms_register_matrix_call_pde_burgers_1d() {
    register_matrix_call("pde_burgers_1d", &handle_pde_burgers_1d);
}

} // namespace ms::interp
