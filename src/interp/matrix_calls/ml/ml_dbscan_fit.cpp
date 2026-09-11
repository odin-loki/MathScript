// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_ml_dbscan_fit(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "ml_dbscan_fit" && assign.args.size() == 3) {
        auto X = ctx.resolve_operand(assign.args[0]);
        if (!X) {
            return std::unexpected(X.error());
        }
        auto eps = ctx.parse_scalar_arg(assign.args[1], "ml_dbscan_fit");
        if (!eps) {
            return std::unexpected(eps.error());
        }
        auto ms = ctx.parse_scalar_arg(assign.args[2], "ml_dbscan_fit");
        if (!ms) {
            return std::unexpected(ms.error());
        }
        const int min_samples = static_cast<int>(*ms);
        if (min_samples < 1 || *ms != min_samples) {
            return std::unexpected(
                DomainError{"ml_dbscan_fit", "expected integer min_samples >= 1"});
        }
        auto labels = eval_ml_dbscan_fit(*X, *eps, min_samples);
        if (!labels) {
            return std::unexpected(labels.error());
        }
        result = *labels;
    }

    return result;
}

void ms_register_matrix_call_ml_dbscan_fit() {
    register_matrix_call("ml_dbscan_fit", &handle_ml_dbscan_fit);
}

} // namespace ms::interp
