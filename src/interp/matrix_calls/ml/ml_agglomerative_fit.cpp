// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_ml_agglomerative_fit(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "ml_agglomerative_fit" &&
               (assign.args.size() >= 1 && assign.args.size() <= 3)) {
        auto X = ctx.resolve_operand(assign.args[0]);
        if (!X) {
            return std::unexpected(X.error());
        }
        int n_clusters = 2;
        std::string linkage = "ward";
        if (assign.args.size() >= 2) {
            auto k_arg = ctx.parse_scalar_arg(assign.args[1], "ml_agglomerative_fit");
            if (!k_arg) {
                return std::unexpected(k_arg.error());
            }
            n_clusters = static_cast<int>(*k_arg);
            if (n_clusters < 1 || *k_arg != n_clusters) {
                return std::unexpected(
                    DomainError{"ml_agglomerative_fit", "expected integer n_clusters >= 1"});
            }
        }
        if (assign.args.size() >= 3) {
            auto linkage_arg = parse_ml_linkage(assign.args[2], "ml_agglomerative_fit");
            if (!linkage_arg) {
                return std::unexpected(linkage_arg.error());
            }
            linkage = *linkage_arg;
        }
        auto labels = eval_ml_agglomerative_fit(*X, n_clusters, linkage);
        if (!labels) {
            return std::unexpected(labels.error());
        }
        result = *labels;
    }

    return result;
}

void ms_register_matrix_call_ml_agglomerative_fit() {
    register_matrix_call("ml_agglomerative_fit", &handle_ml_agglomerative_fit);
}

} // namespace ms::interp
