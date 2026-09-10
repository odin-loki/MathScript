// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_poly_sylvester(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "poly_sylvester" && assign.args.size() == 2) {
        auto p = ctx.resolve_operand(assign.args[0]);
        if (!p) {
            return std::unexpected(p.error());
        }
        auto q = ctx.resolve_operand(assign.args[1]);
        if (!q) {
            return std::unexpected(q.error());
        }
        auto S = eval_poly_sylvester(*p, *q);
        if (!S) {
            return std::unexpected(S.error());
        }
        result = *S;
    }

    return result;
}

void ms_register_matrix_call_poly_sylvester() {
    register_matrix_call("poly_sylvester", &handle_poly_sylvester);
}

} // namespace ms::interp
