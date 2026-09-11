// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_ml_lda_fit(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "ml_lda_fit" &&
               (assign.args.size() == 2 || assign.args.size() == 3)) {
        auto X = ctx.resolve_operand(assign.args[0]);
        if (!X) {
            return std::unexpected(X.error());
        }
        auto y = ctx.resolve_operand(assign.args[1]);
        if (!y) {
            return std::unexpected(y.error());
        }
        int n_components = 0;
        if (assign.args.size() == 3) {
            auto n_comp = ctx.parse_scalar_arg(assign.args[2], "ml_lda_fit");
            if (!n_comp) {
                return std::unexpected(n_comp.error());
            }
            n_components = static_cast<int>(*n_comp);
            if (n_components < 0 || *n_comp != n_components) {
                return std::unexpected(
                    DomainError{"ml_lda_fit", "expected non-negative integer n_components"});
            }
        }
        auto fitted = eval_ml_lda_fit(*X, *y, n_components);
        if (!fitted) {
            return std::unexpected(fitted.error());
        }
        result = *fitted;
    }

    return result;
}

void ms_register_matrix_call_ml_lda_fit() {
    register_matrix_call("ml_lda_fit", &handle_ml_lda_fit);
}

} // namespace ms::interp
