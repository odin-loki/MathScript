#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_signal_xcov(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if ((assign.callee == "signal_xcorr" || assign.callee == "signal_xcov") &&
               assign.args.size() == 3) {
        auto a = resolve_operand(assign.args[0]);
        if (!a) {
            return std::unexpected(a.error());
        }
        auto b = resolve_operand(assign.args[1]);
        if (!b) {
            return std::unexpected(b.error());
        }
        auto max_lag_val = parse_scalar_arg(assign.args[2], assign.callee.c_str());
        if (!max_lag_val) {
            return std::unexpected(max_lag_val.error());
        }
        const int max_lag = static_cast<int>(*max_lag_val);
        if (max_lag < 0 || *max_lag_val != max_lag) {
            return std::unexpected(
                DomainError{assign.callee, "expected non-negative integer max_lag"});
        }
        if (assign.callee == "signal_xcorr") {
            result = eval_signal_xcorr(*a, *b, max_lag);
        } else {
            result = eval_signal_xcov(*a, *b, max_lag);
        }
    }

    return result;
}

void ms_register_matrix_call_signal_xcov() {
    register_matrix_call("signal_xcov", &handle_signal_xcov);
}

} // namespace ms::interp
