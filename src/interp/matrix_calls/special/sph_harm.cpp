// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_sph_harm(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "sph_harm" && assign.args.size() == 4) {
        std::array<double, 4> args{};
        for (std::size_t i = 0; i < 4; ++i) {
            if (!parse_number(assign.args[i], args[i])) {
                auto expr = eval_scalar_expr(ctx.state(), assign.args[i]);
                if (!expr) {
                    return std::unexpected(DomainError{
                        "sph_harm", "expected sph_harm(l,m,theta,phi)"});
                }
                args[i] = *expr;
            }
        }
        const int l = static_cast<int>(args[0]);
        const int m = static_cast<int>(args[1]);
        if (args[0] != l || args[1] != m) {
            return std::unexpected(
                DomainError{"sph_harm", "expected integer l and m"});
        }
        result = eval_sph_harm(l, m, args[2], args[3]);
    }

    return result;
}

void ms_register_matrix_call_sph_harm() {
    register_matrix_call("sph_harm", &handle_sph_harm);
}

} // namespace ms::interp
