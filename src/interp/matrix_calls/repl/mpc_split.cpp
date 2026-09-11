// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_mpc_split(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "mpc_split" && assign.args.size() == 3) {
        auto secret_val = ctx.parse_scalar_arg(assign.args[0], "mpc_split");
        if (!secret_val) {
            return std::unexpected(secret_val.error());
        }
        auto secret = ctx.parse_uint64_arg(*secret_val, "mpc_split", "expected unsigned integer secret");
        if (!secret) {
            return std::unexpected(secret.error());
        }
        auto n_val = ctx.parse_scalar_arg(assign.args[1], "mpc_split");
        if (!n_val) {
            return std::unexpected(n_val.error());
        }
        auto k_val = ctx.parse_scalar_arg(assign.args[2], "mpc_split");
        if (!k_val) {
            return std::unexpected(k_val.error());
        }
        const int n = static_cast<int>(*n_val);
        const int k = static_cast<int>(*k_val);
        if (*n_val != n || *k_val != k) {
            return std::unexpected(DomainError{"mpc_split", "expected integer n and k"});
        }
        result = eval_mpc_split(*secret, n, k);
    }

    return result;
}

void ms_register_matrix_call_mpc_split() {
    register_matrix_call("mpc_split", &handle_mpc_split);
}

} // namespace ms::interp
