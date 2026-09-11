// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_pde_poisson_2d(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "pde_poisson_2d" && assign.args.size() == 5) {
        auto f_m = ctx.resolve_operand(assign.args[0]);
        if (!f_m) {
            return std::unexpected(f_m.error());
        }
        auto dx = ctx.parse_scalar_arg(assign.args[1], "pde_poisson_2d");
        if (!dx) {
            return std::unexpected(dx.error());
        }
        auto dy = ctx.parse_scalar_arg(assign.args[2], "pde_poisson_2d");
        if (!dy) {
            return std::unexpected(dy.error());
        }
        auto max_iter_val = ctx.parse_scalar_arg(assign.args[3], "pde_poisson_2d");
        if (!max_iter_val) {
            return std::unexpected(max_iter_val.error());
        }
        auto tolerance = ctx.parse_scalar_arg(assign.args[4], "pde_poisson_2d");
        if (!tolerance) {
            return std::unexpected(tolerance.error());
        }
        const int max_iter_i = static_cast<int>(*max_iter_val);
        if (max_iter_i < 0 || *max_iter_val != max_iter_i) {
            return std::unexpected(
                DomainError{"pde_poisson_2d", "expected non-negative integer max_iterations"});
        }
        result = eval_pde_poisson_2d(*f_m, *dx, *dy, static_cast<std::size_t>(max_iter_i),
                                     *tolerance);
    }

    return result;
}

void ms_register_matrix_call_pde_poisson_2d() {
    register_matrix_call("pde_poisson_2d", &handle_pde_poisson_2d);
}

} // namespace ms::interp
