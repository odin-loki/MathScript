#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_ml_confusion_matrix(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "ml_confusion_matrix" &&
        (assign.args.size() == 2 || assign.args.size() == 3)) {
        auto y_pred = resolve_operand(assign.args[0]);
        if (!y_pred) {
            return std::unexpected(y_pred.error());
        }
        auto y_true = resolve_operand(assign.args[1]);
        if (!y_true) {
            return std::unexpected(y_true.error());
        }
        double threshold = 0.5;
        if (assign.args.size() == 3) {
            auto thr = parse_scalar_arg(assign.args[2], "ml_confusion_matrix");
            if (!thr) {
                return std::unexpected(thr.error());
            }
            threshold = *thr;
        }
        result = eval_ml_confusion_matrix(*y_pred, *y_true, threshold);
    }

    return result;
}

void ms_register_matrix_call_ml_confusion_matrix() {
    register_matrix_call("ml_confusion_matrix", &handle_ml_confusion_matrix);
}

} // namespace ms::interp
