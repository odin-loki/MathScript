// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_ml_pca_fit(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "ml_pca_fit" && assign.args.size() == 2) {
        auto X = ctx.resolve_operand(assign.args[0]);
        if (!X) {
            return std::unexpected(X.error());
        }
        auto n_comp = ctx.parse_scalar_arg(assign.args[1], "ml_pca_fit");
        if (!n_comp) {
            return std::unexpected(n_comp.error());
        }
        const int n_components = static_cast<int>(*n_comp);
        if (*n_comp != n_components) {
            return std::unexpected(
                DomainError{"ml_pca_fit", "expected integer n_components"});
        }
        auto fitted = eval_ml_pca_fit(*X, n_components);
        if (!fitted) {
            return std::unexpected(fitted.error());
        }
        result = *fitted;
    }

    return result;
}

void ms_register_matrix_call_ml_pca_fit() {
    register_matrix_call("ml_pca_fit", &handle_ml_pca_fit);
}

} // namespace ms::interp
