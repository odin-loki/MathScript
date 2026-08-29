#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_ml_elastic_net_fit(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "ml_elastic_net_fit" && assign.args.size() == 4) {
        auto X = resolve_operand(assign.args[0]);
        if (!X) {
            return std::unexpected(X.error());
        }
        auto y = resolve_operand(assign.args[1]);
        if (!y) {
            return std::unexpected(y.error());
        }
        auto alpha = parse_scalar_arg(assign.args[2], "ml_elastic_net_fit");
        if (!alpha) {
            return std::unexpected(alpha.error());
        }
        auto l1_ratio = parse_scalar_arg(assign.args[3], "ml_elastic_net_fit");
        if (!l1_ratio) {
            return std::unexpected(l1_ratio.error());
        }
        auto fitted = eval_ml_elastic_net_fit(*X, *y, *alpha, *l1_ratio);
        if (!fitted) {
            return std::unexpected(fitted.error());
        }
        result = *fitted;
    }

    return result;
}

void ms_register_matrix_call_ml_elastic_net_fit() {
    register_matrix_call("ml_elastic_net_fit", &handle_ml_elastic_net_fit);
}

} // namespace ms::interp
