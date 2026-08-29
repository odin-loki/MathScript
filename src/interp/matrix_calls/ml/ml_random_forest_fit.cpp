#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_ml_random_forest_fit(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "ml_random_forest_fit" &&
               (assign.args.size() >= 2 && assign.args.size() <= 4)) {
        auto X = resolve_operand(assign.args[0]);
        if (!X) {
            return std::unexpected(X.error());
        }
        auto y = resolve_operand(assign.args[1]);
        if (!y) {
            return std::unexpected(y.error());
        }
        size_t n_trees = 50;
        size_t max_depth = 5;
        if (assign.args.size() >= 3) {
            auto trees = parse_scalar_arg(assign.args[2], "ml_random_forest_fit");
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
            auto depth = parse_scalar_arg(assign.args[3], "ml_random_forest_fit");
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
