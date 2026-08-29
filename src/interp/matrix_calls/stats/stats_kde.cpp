#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_stats_kde(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "stats_kde" &&
               (assign.args.size() == 3 || assign.args.size() == 4)) {
        auto samples = resolve_operand(assign.args[0]);
        if (!samples) {
            return std::unexpected(samples.error());
        }
        auto grid = resolve_operand(assign.args[1]);
        if (!grid) {
            return std::unexpected(grid.error());
        }
        double h = 0.0;
        if (!parse_number(assign.args[2], h)) {
            return std::unexpected(
                DomainError{"stats_kde", "expected stats_kde(samples, grid, h[, kernel])"});
        }
        std::string kernel = "gaussian";
        if (assign.args.size() == 4) {
            if (!parse_quoted_string(trim_copy(assign.args[3]), kernel)) {
                kernel = trim_copy(assign.args[3]);
            }
            if (kernel.empty()) {
                return std::unexpected(
                    DomainError{"stats_kde", "expected stats_kde(samples, grid, h[, kernel])"});
            }
        }
        result = eval_stats_kde(*samples, *grid, h, kernel.c_str());
    }

    return result;
}

void ms_register_matrix_call_stats_kde() {
    register_matrix_call("stats_kde", &handle_stats_kde);
}

} // namespace ms::interp
