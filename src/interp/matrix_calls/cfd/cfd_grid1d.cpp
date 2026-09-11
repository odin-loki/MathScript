// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_cfd_grid1d(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "cfd_grid1d" && assign.args.size() == 3) {
        auto x0 = ctx.parse_scalar_arg(assign.args[0], "cfd_grid1d");
        if (!x0) {
            return std::unexpected(x0.error());
        }
        auto x1 = ctx.parse_scalar_arg(assign.args[1], "cfd_grid1d");
        if (!x1) {
            return std::unexpected(x1.error());
        }
        auto n_val = ctx.parse_scalar_arg(assign.args[2], "cfd_grid1d");
        if (!n_val) {
            return std::unexpected(n_val.error());
        }
        const int n_i = static_cast<int>(*n_val);
        if (n_i < 2 || *n_val != n_i) {
            return std::unexpected(DomainError{"cfd_grid1d", "expected integer n >= 2"});
        }
        result = eval_cfd_grid1d(*x0, *x1, static_cast<std::size_t>(n_i));
    }

    return result;
}

void ms_register_matrix_call_cfd_grid1d() {
    register_matrix_call("cfd_grid1d", &handle_cfd_grid1d);
}

} // namespace ms::interp
