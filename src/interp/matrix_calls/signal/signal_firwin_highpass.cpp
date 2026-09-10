#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_signal_firwin_highpass(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if ((assign.callee == "signal_firwin" || assign.callee == "signal_firwin_highpass") &&
               (assign.args.size() == 2 || assign.args.size() == 3)) {
        auto n_taps_val = ctx.parse_scalar_arg(assign.args[0], assign.callee.c_str());
        if (!n_taps_val) {
            return std::unexpected(n_taps_val.error());
        }
        auto cutoff = ctx.parse_scalar_arg(assign.args[1], assign.callee.c_str());
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

void ms_register_matrix_call_signal_firwin_highpass() {
    register_matrix_call("signal_firwin_highpass", &handle_signal_firwin_highpass);
}

} // namespace ms::interp
