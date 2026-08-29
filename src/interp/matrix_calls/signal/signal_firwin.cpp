#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_signal_firwin(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if ((assign.callee == "signal_firwin" || assign.callee == "signal_firwin_highpass") &&
               (assign.args.size() == 2 || assign.args.size() == 3)) {
        auto n_taps_val = parse_scalar_arg(assign.args[0], assign.callee.c_str());
        if (!n_taps_val) {
            return std::unexpected(n_taps_val.error());
        }
        auto cutoff = parse_scalar_arg(assign.args[1], assign.callee.c_str());
        if (!cutoff) {
            return std::unexpected(cutoff.error());
        }
        const int n_taps = static_cast<int>(*n_taps_val);
        if (n_taps < 1 || *n_taps_val != n_taps) {
            return std::unexpected(
                DomainError{assign.callee, "expected integer n_taps >= 1"});
        }
        FirWindow window = FirWindow::Hamming;
        if (assign.args.size() == 3) {
            auto parsed = parse_fir_window(assign.args[2], assign.callee.c_str());
            if (!parsed) {
                return std::unexpected(parsed.error());
            }
            window = *parsed;
        }
        if (assign.callee == "signal_firwin") {
            result = eval_signal_firwin(n_taps, *cutoff, window);
        } else {
            result = eval_signal_firwin_highpass(n_taps, *cutoff, window);
        }
    }

    return result;
}

void ms_register_matrix_call_signal_firwin() {
    register_matrix_call("signal_firwin", &handle_signal_firwin);
}

} // namespace ms::interp
