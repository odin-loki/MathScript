#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_signal_unwrap(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "signal_unwrap" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto unwrapped = eval_signal_unwrap(*matrix);
        if (!unwrapped) {
            return std::unexpected(unwrapped.error());
        }
        result = *unwrapped;
    }

    return result;
}

void ms_register_matrix_call_signal_unwrap() {
    register_matrix_call("signal_unwrap", &handle_signal_unwrap);
}

} // namespace ms::interp
