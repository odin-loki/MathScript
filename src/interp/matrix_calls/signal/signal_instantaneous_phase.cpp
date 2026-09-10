#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_signal_instantaneous_phase(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "signal_instantaneous_phase" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto phase = eval_signal_instantaneous_phase(*matrix);
        if (!phase) {
            return std::unexpected(phase.error());
        }
        result = *phase;
    }

    return result;
}

void ms_register_matrix_call_signal_instantaneous_phase() {
    register_matrix_call("signal_instantaneous_phase", &handle_signal_instantaneous_phase);
}

} // namespace ms::interp
