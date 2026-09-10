#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_ml_gradient_boosting_fit(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "ml_gradient_boosting_fit" &&
               (assign.args.size() >= 2 && assign.args.size() <= 5)) {
        auto X = ctx.resolve_operand(assign.args[0]);
        if (!X) {
            return std::unexpected(X.error());
        }
        auto y = ctx.resolve_operand(assign.args[1]);
        if (!y) {
            return std::unexpected(y.error());
        }
        size_t n_estimators = 50;
        double learning_rate = 0.1;
        size_t max_depth = 3;
        if (assign.args.size() >= 3) {
            auto est = ctx.parse_scalar_arg(assign.args[2], "ml_gradient_boosting_fit");
            if (!est) {
                return std::unexpected(est.error());
            }
            n_estimators = static_cast<size_t>(*est);
            if (*est != static_cast<double>(n_estimators) || n_estimators < 1) {
                return std::unexpected(DomainError{
                    "ml_gradient_boosting_fit", "expected positive integer n_estimators"});
            }
        }
        if (assign.args.size() >= 4) {
            auto lr = ctx.parse_scalar_arg(assign.args[3], "ml_gradient_boosting_fit");
            if (!lr) {
                return std::unexpected(lr.error());
            }
            learning_rate = *lr;
            if (learning_rate <= 0.0) {
                return std::unexpected(DomainError{
                    "ml_gradient_boosting_fit", "expected positive learning_rate"});
            }
        }
        if (assign.args.size() == 5) {
            auto depth = ctx.parse_scalar_arg(assign.args[4], "ml_gradient_boosting_fit");
            if (!depth) {
                return std::unexpected(depth.error());
            }
            max_depth = static_cast<size_t>(*depth);
            if (*depth != static_cast<double>(max_depth) || max_depth < 1) {
                return std::unexpected(DomainError{
                    "ml_gradient_boosting_fit", "expected positive integer max_depth"});
            }
        }
        auto fitted =
            eval_ml_gradient_boosting_fit(*X, *y, n_estimators, learning_rate, max_depth);
        if (!fitted) {
            return std::unexpected(fitted.error());
        }
        result = *fitted;
    }

    return result;
}

void ms_register_matrix_call_ml_gradient_boosting_fit() {
    register_matrix_call("ml_gradient_boosting_fit", &handle_ml_gradient_boosting_fit);
}

} // namespace ms::interp
