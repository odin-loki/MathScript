#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_ml_decision_tree_fit(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "ml_decision_tree_fit" &&
        (assign.args.size() == 2 || assign.args.size() == 3)) {
        auto X = ctx.resolve_operand(assign.args[0]);
        if (!X) {
            return std::unexpected(X.error());
        }
        auto y = ctx.resolve_operand(assign.args[1]);
        if (!y) {
            return std::unexpected(y.error());
        }
        int max_depth = 5;
        if (assign.args.size() == 3) {
            auto depth = ctx.parse_scalar_arg(assign.args[2], "ml_decision_tree_fit");
            if (!depth) {
                return std::unexpected(depth.error());
            }
            max_depth = static_cast<int>(*depth);
            if (*depth != max_depth || max_depth < 1) {
                return std::unexpected(
                    DomainError{"ml_decision_tree_fit", "expected positive integer max_depth"});
            }
        }
        auto fitted = eval_ml_decision_tree_fit(*X, *y, max_depth);
        if (!fitted) {
            return std::unexpected(fitted.error());
        }
        result = *fitted;
    }

    return result;
}

void ms_register_matrix_call_ml_decision_tree_fit() {
    register_matrix_call("ml_decision_tree_fit", &handle_ml_decision_tree_fit);
}

} // namespace ms::interp
