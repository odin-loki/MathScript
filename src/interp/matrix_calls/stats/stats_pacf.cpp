#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_stats_pacf(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "stats_pacf" && assign.args.size() == 2) {
        auto x = resolve_operand(assign.args[0]);
        if (!x) {
            return std::unexpected(x.error());
        }
        double max_lag_d = 0.0;
        if (!parse_number(assign.args[1], max_lag_d)) {
            return std::unexpected(
                DomainError{"stats_pacf", "expected stats_pacf(x, max_lag)"});
        }
        const int max_lag = static_cast<int>(max_lag_d);
        if (max_lag < 0 || max_lag_d != max_lag) {
            return std::unexpected(
                DomainError{"stats_pacf", "expected non-negative integer max_lag"});
        }
        result = eval_stats_pacf(*x, max_lag);
    }

    return result;
}

void ms_register_matrix_call_stats_pacf() {
    register_matrix_call("stats_pacf", &handle_stats_pacf);
}

} // namespace ms::interp
