// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_ml_elastic_net_fit(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "ml_elastic_net_fit" && assign.args.size() == 4) {
        auto X = ctx.resolve_operand(assign.args[0]);
        if (!X) {
            return std::unexpected(X.error());
        }
        auto y = ctx.resolve_operand(assign.args[1]);
        if (!y) {
            return std::unexpected(y.error());
        }
        auto alpha = ctx.parse_scalar_arg(assign.args[2], "ml_elastic_net_fit");
        if (!alpha) {
            return std::unexpected(alpha.error());
        }
        auto l1_ratio = ctx.parse_scalar_arg(assign.args[3], "ml_elastic_net_fit");
        if (!l1_ratio) {
            return std::unexpected(l1_ratio.error());
        }
        auto fitted = eval_ml_elastic_net_fit(*X, *y, *alpha, *l1_ratio);
        if (!fitted) {
            return std::unexpected(fitted.error());
        }
        result = *fitted;
    }

    return result;
}

void ms_register_matrix_call_ml_elastic_net_fit() {
    register_matrix_call("ml_elastic_net_fit", &handle_ml_elastic_net_fit);
}

} // namespace ms::interp
