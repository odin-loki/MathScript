#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_idst2(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "idst2" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto signal = eval_fft_idst2(*matrix);
        if (!signal) {
            return std::unexpected(signal.error());
        }
        result = *signal;
    }

    return result;
}

void ms_register_matrix_call_idst2() {
    register_matrix_call("idst2", &handle_idst2);
}

} // namespace ms::interp
