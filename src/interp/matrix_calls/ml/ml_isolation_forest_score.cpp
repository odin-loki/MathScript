#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_ml_isolation_forest_score(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "ml_isolation_forest_score" && assign.args.size() == 2) {
        auto X = ctx.resolve_operand(assign.args[0]);
        if (!X) {
            return std::unexpected(X.error());
        }
        auto model = ctx.resolve_operand(assign.args[1]);
        if (!model) {
            return std::unexpected(model.error());
        }
        auto scores = eval_ml_isolation_forest_score(*X, *model);
        if (!scores) {
            return std::unexpected(scores.error());
        }
        result = *scores;
    }

    return result;
}

void ms_register_matrix_call_ml_isolation_forest_score() {
    register_matrix_call("ml_isolation_forest_score", &handle_ml_isolation_forest_score);
}

} // namespace ms::interp
