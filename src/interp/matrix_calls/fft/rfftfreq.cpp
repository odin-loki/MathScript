#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_rfftfreq(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "rfftfreq" &&
               (assign.args.size() == 1 || assign.args.size() == 2)) {
        auto n_val = ctx.parse_scalar_arg(assign.args[0], "rfftfreq");
        if (!n_val) {
            return std::unexpected(n_val.error());
        }
        const int n_i = static_cast<int>(*n_val);
        if (n_i < 0 || *n_val != n_i) {
            return std::unexpected(
                DomainError{"rfftfreq", "expected non-negative integer n"});
        }
        double d = 1.0;
        if (assign.args.size() == 2) {
            auto d_val = ctx.parse_scalar_arg(assign.args[1], "rfftfreq");
            if (!d_val) {
                return std::unexpected(d_val.error());
            }
            d = *d_val;
        }
        auto freqs = eval_rfftfreq(static_cast<size_t>(n_i), d);
        if (!freqs) {
            return std::unexpected(freqs.error());
        }
        result = *freqs;
    }

    return result;
}

void ms_register_matrix_call_rfftfreq() {
    register_matrix_call("rfftfreq", &handle_rfftfreq);
}

} // namespace ms::interp
