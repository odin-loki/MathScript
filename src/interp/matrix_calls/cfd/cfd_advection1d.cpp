// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_cfd_advection1d(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "cfd_advection1d" && assign.args.size() == 4) {
        auto nx_val = ctx.parse_scalar_arg(assign.args[0], "cfd_advection1d");
        if (!nx_val) {
            return std::unexpected(nx_val.error());
        }
        auto vx_val = ctx.parse_scalar_arg(assign.args[1], "cfd_advection1d");
        if (!vx_val) {
            return std::unexpected(vx_val.error());
        }
        auto t_end_val = ctx.parse_scalar_arg(assign.args[2], "cfd_advection1d");
        if (!t_end_val) {
            return std::unexpected(t_end_val.error());
        }
        auto dt_val = ctx.parse_scalar_arg(assign.args[3], "cfd_advection1d");
        if (!dt_val) {
            return std::unexpected(dt_val.error());
        }
        const int nx_i = static_cast<int>(*nx_val);
        if (nx_i < 0 || *nx_val != nx_i) {
            return std::unexpected(
                DomainError{"cfd_advection1d", "expected non-negative integer nx"});
        }
        result = eval_cfd_advection1d(static_cast<std::size_t>(nx_i), *vx_val, *t_end_val,
                                      *dt_val);
    }

    return result;
}

void ms_register_matrix_call_cfd_advection1d() {
    register_matrix_call("cfd_advection1d", &handle_cfd_advection1d);
}

} // namespace ms::interp
