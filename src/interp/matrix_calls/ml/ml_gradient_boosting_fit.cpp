#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_ml_gradient_boosting_fit(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);
    auto resolve_operand = [&ctx](const std::string& text) { return ctx.resolve_operand(text); };
    auto parse_scalar_arg = [&ctx](const std::string& arg_text, const char* fn) -> Result<double> {
        double value = 0.0;
        if (parse_number(arg_text, value)) return value;
        auto expr = eval_scalar_expr(ctx.state(), arg_text);
        if (!expr) {
            return std::unexpected(DomainError{fn, "expected numeric scalar argument"});
        }
        return *expr;
    };
    auto parse_positive_size_arg = [](double value, const char* fn, const char* label) -> Result<std::size_t> {
        const int i = static_cast<int>(value);
        if (i < 1 || value != static_cast<double>(i)) {
            return std::unexpected(DomainError{fn, label});
        }
        return static_cast<std::size_t>(i);
    };
    auto parse_uint64_arg = [](double value, const char* fn, const char* label) -> Result<uint64_t> {
        if (value < 0.0 || value != std::floor(value)) {
            return std::unexpected(DomainError{fn, label});
        }
        return static_cast<uint64_t>(value);
    };

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "ml_gradient_boosting_fit" &&
               (assign.args.size() >= 2 && assign.args.size() <= 5)) {
        auto X = resolve_operand(assign.args[0]);
        if (!X) {
            return std::unexpected(X.error());
        }
        auto y = resolve_operand(assign.args[1]);
        if (!y) {
            return std::unexpected(y.error());
        }
        size_t n_estimators = 50;
        double learning_rate = 0.1;
        size_t max_depth = 3;
        if (assign.args.size() >= 3) {
            auto est = parse_scalar_arg(assign.args[2], "ml_gradient_boosting_fit");
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
            auto lr = parse_scalar_arg(assign.args[3], "ml_gradient_boosting_fit");
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
            auto depth = parse_scalar_arg(assign.args[4], "ml_gradient_boosting_fit");
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
