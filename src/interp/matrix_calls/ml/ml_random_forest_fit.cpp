#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_ml_random_forest_fit(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "ml_random_forest_fit" &&
               (assign.args.size() >= 2 && assign.args.size() <= 4)) {
        auto X = ctx.resolve_operand(assign.args[0]);
        if (!X) {
            return std::unexpected(X.error());
        }
        auto y = ctx.resolve_operand(assign.args[1]);
        if (!y) {
            return std::unexpected(y.error());
        }
        size_t n_trees = 50;
        size_t max_depth = 5;
        if (assign.args.size() >= 3) {
            auto trees = ctx.parse_scalar_arg(assign.args[2], "ml_random_forest_fit");
            if (!trees) {
                return std::unexpected(trees.error());
            }
            n_trees = static_cast<size_t>(*trees);
            if (*trees != static_cast<double>(n_trees) || n_trees < 1) {
                return std::unexpected(
                    DomainError{"ml_random_forest_fit", "expected positive integer n_trees"});
            }
        }
        if (assign.args.size() == 4) {
            auto depth = ctx.parse_scalar_arg(assign.args[3], "ml_random_forest_fit");
            if (!depth) {
                return std::unexpected(depth.error());
            }
            max_depth = static_cast<size_t>(*depth);
            if (*depth != static_cast<double>(max_depth) || max_depth < 1) {
                return std::unexpected(
                    DomainError{"ml_random_forest_fit", "expected positive integer max_depth"});
            }
        }
        auto fitted = eval_ml_random_forest_fit(*X, *y, n_trees, max_depth);
        if (!fitted) {
            return std::unexpected(fitted.error());
        }
        result = *fitted;
    }

    return result;
}

void ms_register_matrix_call_ml_random_forest_fit() {
    register_matrix_call("ml_random_forest_fit", &handle_ml_random_forest_fit);
}

} // namespace ms::interp
