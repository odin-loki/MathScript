#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_fft_dst2(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "fft_dst2" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto coeffs = eval_fft_dst2(*matrix);
        if (!coeffs) {
            return std::unexpected(coeffs.error());
        }
        result = *coeffs;
    }

    return result;
}

void ms_register_matrix_call_fft_dst2() {
    register_matrix_call("fft_dst2", &handle_fft_dst2);
}

} // namespace ms::interp
