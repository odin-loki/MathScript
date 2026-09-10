#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_signal_deconv(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "signal_deconv" && assign.args.size() == 2) {
        auto y = ctx.resolve_operand(assign.args[0]);
        if (!y) {
            return std::unexpected(y.error());
        }
        auto b = ctx.resolve_operand(assign.args[1]);
        if (!b) {
            return std::unexpected(b.error());
        }
        result = eval_signal_deconv(*y, *b);
    }

    return result;
}

void ms_register_matrix_call_signal_deconv() {
    register_matrix_call("signal_deconv", &handle_signal_deconv);
}

} // namespace ms::interp
