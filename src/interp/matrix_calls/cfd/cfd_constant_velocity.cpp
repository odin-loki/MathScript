// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_cfd_constant_velocity(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "cfd_constant_velocity" && assign.args.size() == 2) {
        auto n_val = ctx.parse_scalar_arg(assign.args[0], "cfd_constant_velocity");
        if (!n_val) {
            return std::unexpected(n_val.error());
        }
        auto v = ctx.parse_scalar_arg(assign.args[1], "cfd_constant_velocity");
        if (!v) {
            return std::unexpected(v.error());
        }
        auto n_i = ctx.parse_positive_size_arg(*n_val, "cfd_constant_velocity", "expected positive integer n");
        if (!n_i) {
            return std::unexpected(n_i.error());
        }
        result = eval_cfd_constant_velocity(*n_i, *v);
    }

    return result;
}

void ms_register_matrix_call_cfd_constant_velocity() {
    register_matrix_call("cfd_constant_velocity", &handle_cfd_constant_velocity);
}

} // namespace ms::interp
