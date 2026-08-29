#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_ml_pca_fit_transform(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "ml_pca_fit_transform" && assign.args.size() == 2) {
        auto X = resolve_operand(assign.args[0]);
        if (!X) {
            return std::unexpected(X.error());
        }
        auto n_comp = parse_scalar_arg(assign.args[1], "ml_pca_fit_transform");
        if (!n_comp) {
            return std::unexpected(n_comp.error());
        }
        const int n_components = static_cast<int>(*n_comp);
        if (*n_comp != n_components) {
            return std::unexpected(
                DomainError{"ml_pca_fit_transform", "expected integer n_components"});
        }
        auto transformed = eval_ml_pca_fit_transform(*X, n_components);
        if (!transformed) {
            return std::unexpected(transformed.error());
        }
        result = *transformed;
    }

    return result;
}

void ms_register_matrix_call_ml_pca_fit_transform() {
    register_matrix_call("ml_pca_fit_transform", &handle_ml_pca_fit_transform);
}

} // namespace ms::interp
