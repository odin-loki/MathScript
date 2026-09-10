#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_ml_naive_bayes_predict(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "ml_naive_bayes_predict" && assign.args.size() == 2) {
        auto X = ctx.resolve_operand(assign.args[0]);
        if (!X) {
            return std::unexpected(X.error());
        }
        auto model = ctx.resolve_operand(assign.args[1]);
        if (!model) {
            return std::unexpected(model.error());
        }
        auto predicted = eval_ml_naive_bayes_predict(*X, *model);
        if (!predicted) {
            return std::unexpected(predicted.error());
        }
        result = *predicted;
    }

    return result;
}

void ms_register_matrix_call_ml_naive_bayes_predict() {
    register_matrix_call("ml_naive_bayes_predict", &handle_ml_naive_bayes_predict);
}

} // namespace ms::interp
