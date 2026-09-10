// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_fem_poisson1d(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "fem_poisson1d" && assign.args.size() == 1) {
        auto n_val = ctx.parse_scalar_arg(assign.args[0], "fem_poisson1d");
        if (!n_val) {
            return std::unexpected(n_val.error());
        }
        const int n_i = static_cast<int>(*n_val);
        if (n_i < 0 || *n_val != n_i) {
            return std::unexpected(
                DomainError{"fem_poisson1d", "expected non-negative integer n"});
        }
        result = eval_fem_poisson1d(static_cast<std::size_t>(n_i));
    }

    return result;
}

void ms_register_matrix_call_fem_poisson1d() {
    register_matrix_call("fem_poisson1d", &handle_fem_poisson1d);
}

} // namespace ms::interp
