// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_poly_gcd(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "poly_gcd" && assign.args.size() == 2) {
        auto a = ctx.resolve_operand(assign.args[0]);
        if (!a) {
            return std::unexpected(a.error());
        }
        auto b = ctx.resolve_operand(assign.args[1]);
        if (!b) {
            return std::unexpected(b.error());
        }
        auto g = eval_poly_gcd(*a, *b);
        if (!g) {
            return std::unexpected(g.error());
        }
        result = *g;
    }

    return result;
}

void ms_register_matrix_call_poly_gcd() {
    register_matrix_call("poly_gcd", &handle_poly_gcd);
}

} // namespace ms::interp
