// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_pde_advection_1d_lax_wendroff(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "pde_advection_1d_lax_wendroff" && assign.args.size() == 5) {
        auto u0_m = ctx.resolve_operand(assign.args[0]);
        if (!u0_m) {
            return std::unexpected(u0_m.error());
        }
        auto v = ctx.parse_scalar_arg(assign.args[1], "pde_advection_1d_lax_wendroff");
        if (!v) {
            return std::unexpected(v.error());
        }
        auto dx = ctx.parse_scalar_arg(assign.args[2], "pde_advection_1d_lax_wendroff");
        if (!dx) {
            return std::unexpected(dx.error());
        }
        auto dt = ctx.parse_scalar_arg(assign.args[3], "pde_advection_1d_lax_wendroff");
        if (!dt) {
            return std::unexpected(dt.error());
        }
        auto steps_val = ctx.parse_scalar_arg(assign.args[4], "pde_advection_1d_lax_wendroff");
        if (!steps_val) {
            return std::unexpected(steps_val.error());
        }
        WorkBudget budget(assign.callee, 21.0);
        // Charged before `steps` is read so the bound on it shrinks as the grid grows:
        // the solver keeps one grid per step and the REPL reads only the last.
        budget.charge(u0_m->rows() * u0_m->cols());
        auto steps_arg = budget.take("steps", *steps_val);
        if (!steps_arg) {
            return std::unexpected(steps_arg.error());
        }
        const int steps_i = *steps_arg;
        result = eval_pde_advection_1d_lax_wendroff(*u0_m, *v, *dx, *dt,
                                                    static_cast<std::size_t>(steps_i));
    }

    return result;
}

void ms_register_matrix_call_pde_advection_1d_lax_wendroff() {
    register_matrix_call("pde_advection_1d_lax_wendroff", &handle_pde_advection_1d_lax_wendroff);
}

} // namespace ms::interp
