// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_pde_poisson_1d(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "pde_poisson_1d" && assign.args.size() == 4) {
        auto f_m = ctx.resolve_operand(assign.args[0]);
        if (!f_m) {
            return std::unexpected(f_m.error());
        }
        auto dx = ctx.parse_scalar_arg(assign.args[1], "pde_poisson_1d");
        if (!dx) {
            return std::unexpected(dx.error());
        }
        auto ua = ctx.parse_scalar_arg(assign.args[2], "pde_poisson_1d");
        if (!ua) {
            return std::unexpected(ua.error());
        }
        auto ub = ctx.parse_scalar_arg(assign.args[3], "pde_poisson_1d");
        if (!ub) {
            return std::unexpected(ub.error());
        }
        result = eval_pde_poisson_1d(*f_m, *dx, *ua, *ub);
    }

    return result;
}

void ms_register_matrix_call_pde_poisson_1d() {
    register_matrix_call("pde_poisson_1d", &handle_pde_poisson_1d);
}

} // namespace ms::interp
