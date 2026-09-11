// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_topo_alpha_complex(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "topo_alpha_complex" &&
               (assign.args.size() == 2 || assign.args.size() == 3)) {
        auto P = ctx.resolve_operand(assign.args[0]);
        if (!P) {
            return std::unexpected(P.error());
        }
        auto alpha = ctx.parse_scalar_arg(assign.args[1], "topo_alpha_complex");
        if (!alpha) {
            return std::unexpected(alpha.error());
        }
        int max_dim = 2;
        if (assign.args.size() == 3) {
            auto md = ctx.parse_scalar_arg(assign.args[2], "topo_alpha_complex");
            if (!md) {
                return std::unexpected(md.error());
            }
            max_dim = static_cast<int>(*md);
            if (max_dim < 0 || *md != max_dim) {
                return std::unexpected(
                    DomainError{"topo_alpha_complex", "expected non-negative integer max_dim"});
            }
        }
        result = eval_topo_alpha_complex(*P, *alpha, max_dim);
    }

    return result;
}

void ms_register_matrix_call_topo_alpha_complex() {
    register_matrix_call("topo_alpha_complex", &handle_topo_alpha_complex);
}

} // namespace ms::interp
