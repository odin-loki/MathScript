#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_fftshift(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "fftshift" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto shifted = eval_fftshift(*matrix);
        if (!shifted) {
            return std::unexpected(shifted.error());
        }
        result = *shifted;
    }

    return result;
}

void ms_register_matrix_call_fftshift() {
    register_matrix_call("fftshift", &handle_fftshift);
}

} // namespace ms::interp
