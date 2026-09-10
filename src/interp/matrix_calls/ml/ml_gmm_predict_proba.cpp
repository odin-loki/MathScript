#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_ml_gmm_predict_proba(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "ml_gmm_predict_proba" && assign.args.size() == 2) {
        auto X = ctx.resolve_operand(assign.args[0]);
        if (!X) {
            return std::unexpected(X.error());
        }
        auto model = ctx.resolve_operand(assign.args[1]);
        if (!model) {
            return std::unexpected(model.error());
        }
        auto proba = eval_ml_gmm_predict_proba(*X, *model);
        if (!proba) {
            return std::unexpected(proba.error());
        }
        result = *proba;
    }

    return result;
}

void ms_register_matrix_call_ml_gmm_predict_proba() {
    register_matrix_call("ml_gmm_predict_proba", &handle_ml_gmm_predict_proba);
}

} // namespace ms::interp
