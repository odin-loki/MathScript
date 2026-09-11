// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_geo_hermite_curve(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "geo_hermite_curve" && assign.args.size() == 9) {
        std::array<double, 9> args{};
        for (std::size_t i = 0; i < 9; ++i) {
            if (!parse_number(assign.args[i], args[i])) {
                auto expr = eval_scalar_expr(ctx.state(), assign.args[i]);
                if (!expr) {
                    return std::unexpected(DomainError{
                        "geo_hermite_curve",
                        "expected geo_hermite_curve(p0x,p0y,m0x,m0y,p1x,p1y,m1x,m1y,t)"});
                }
                args[i] = *expr;
            }
        }
        result = eval_geo_hermite_curve(args[0], args[1], args[2], args[3], args[4], args[5],
                                        args[6], args[7], args[8]);
    }

    return result;
}

void ms_register_matrix_call_geo_hermite_curve() {
    register_matrix_call("geo_hermite_curve", &handle_geo_hermite_curve);
}

} // namespace ms::interp
