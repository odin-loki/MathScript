// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_ml_isolation_forest_fit(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "ml_isolation_forest_fit" &&
               (assign.args.size() >= 1 && assign.args.size() <= 4)) {
        auto X = ctx.resolve_operand(assign.args[0]);
        if (!X) {
            return std::unexpected(X.error());
        }
        size_t n_trees = 100;
        size_t sample_size = 256;
        unsigned seed = 42;
        if (assign.args.size() >= 2) {
            auto trees = ctx.parse_scalar_arg(assign.args[1], "ml_isolation_forest_fit");
            if (!trees) {
                return std::unexpected(trees.error());
            }
            n_trees = static_cast<size_t>(*trees);
            if (*trees != static_cast<double>(n_trees) || n_trees < 1) {
                return std::unexpected(
                    DomainError{"ml_isolation_forest_fit", "expected positive integer n_trees"});
            }
        }
        if (assign.args.size() >= 3) {
            auto ss = ctx.parse_scalar_arg(assign.args[2], "ml_isolation_forest_fit");
            if (!ss) {
                return std::unexpected(ss.error());
            }
            sample_size = static_cast<size_t>(*ss);
            if (*ss != static_cast<double>(sample_size) || sample_size < 1) {
                return std::unexpected(DomainError{
                    "ml_isolation_forest_fit", "expected positive integer sample_size"});
            }
        }
        if (assign.args.size() >= 4) {
            auto seed_val = parse_optional_seed(assign.args[3], "ml_isolation_forest_fit");
            if (!seed_val) {
                return std::unexpected(seed_val.error());
            }
            seed = *seed_val;
        }
        auto fitted = eval_ml_isolation_forest_fit(*X, n_trees, sample_size, seed);
        if (!fitted) {
            return std::unexpected(fitted.error());
        }
        result = *fitted;
    }

    return result;
}

void ms_register_matrix_call_ml_isolation_forest_fit() {
    register_matrix_call("ml_isolation_forest_fit", &handle_ml_isolation_forest_fit);
}

} // namespace ms::interp
