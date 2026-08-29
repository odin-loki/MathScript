#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_signal_coherence(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "signal_coherence" && assign.args.size() == 4) {
        auto x_m = resolve_operand(assign.args[0]);
        if (!x_m) {
            return std::unexpected(x_m.error());
        }
        auto y_m = resolve_operand(assign.args[1]);
        if (!y_m) {
            return std::unexpected(y_m.error());
        }
        auto fs_val = parse_scalar_arg(assign.args[2], "signal_coherence");
        if (!fs_val) {
            return std::unexpected(fs_val.error());
        }
        auto nperseg_val = parse_scalar_arg(assign.args[3], "signal_coherence");
        if (!nperseg_val) {
            return std::unexpected(nperseg_val.error());
        }
        const int nperseg = static_cast<int>(*nperseg_val);
        if (nperseg < 1 || *nperseg_val != nperseg) {
            return std::unexpected(
                DomainError{"signal_coherence", "expected positive integer nperseg"});
        }
        auto coh = eval_signal_coherence(*x_m, *y_m, *fs_val, static_cast<size_t>(nperseg));
        if (!coh) {
            return std::unexpected(coh.error());
        }
        result = *coh;
    }

    return result;
}

void ms_register_matrix_call_signal_coherence() {
    register_matrix_call("signal_coherence", &handle_signal_coherence);
}

} // namespace ms::interp
