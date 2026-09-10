#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_quantum_coherent_state(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "quantum_coherent_state" && assign.args.size() == 3) {
        double alpha_re = 0.0;
        double alpha_im = 0.0;
        double n_max_d = 0.0;
        if (!parse_number(assign.args[0], alpha_re)) {
            auto it = ctx.state().scalars.find(assign.args[0]);
            if (it != ctx.state().scalars.end()) {
                alpha_re = it->second;
            } else {
                return std::unexpected(
                    DomainError{assign.callee, "expected numeric alpha_re argument"});
            }
        }
        if (!parse_number(assign.args[1], alpha_im)) {
            auto it = ctx.state().scalars.find(assign.args[1]);
            if (it != ctx.state().scalars.end()) {
                alpha_im = it->second;
            } else {
                return std::unexpected(
                    DomainError{assign.callee, "expected numeric alpha_im argument"});
            }
        }
        if (!parse_number(assign.args[2], n_max_d)) {
            auto it = ctx.state().scalars.find(assign.args[2]);
            if (it != ctx.state().scalars.end()) {
                n_max_d = it->second;
            } else {
                return std::unexpected(
                    DomainError{assign.callee, "expected numeric n_max argument"});
            }
        }
        const int n_max = static_cast<int>(n_max_d);
        if (n_max < 0 || n_max_d != n_max) {
            return std::unexpected(DomainError{
                assign.callee, "expected non-negative integer n_max"});
        }
        auto state = eval_quantum_coherent_state(alpha_re, alpha_im, n_max);
        if (!state) {
            return std::unexpected(state.error());
        }
        result = *state;
    }

    return result;
}

void ms_register_matrix_call_quantum_coherent_state() {
    register_matrix_call("quantum_coherent_state", &handle_quantum_coherent_state);
}

} // namespace ms::interp
