#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_fft_ifft(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "fft_ifft" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto signal = eval_fft_ifft(*matrix);
        if (!signal) {
            return std::unexpected(signal.error());
        }
        result = *signal;
    }

    return result;
}

void ms_register_matrix_call_fft_ifft() {
    register_matrix_call("fft_ifft", &handle_fft_ifft);
}

} // namespace ms::interp
