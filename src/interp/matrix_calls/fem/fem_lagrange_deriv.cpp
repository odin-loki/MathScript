// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_fem_lagrange_deriv(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "fem_lagrange_deriv" && assign.args.size() == 1) {
        auto xi = ctx.parse_scalar_arg(assign.args[0], "fem_lagrange_deriv");
        if (!xi) {
            return std::unexpected(xi.error());
        }
        result = eval_fem_lagrange_deriv(*xi);
    }

    return result;
}

void ms_register_matrix_call_fem_lagrange_deriv() {
    register_matrix_call("fem_lagrange_deriv", &handle_fem_lagrange_deriv);
}

} // namespace ms::interp
