#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_fft_rfft(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "fft_rfft" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto spectrum = eval_fft_rfft(*matrix);
        if (!spectrum) {
            return std::unexpected(spectrum.error());
        }
        result = *spectrum;
    }

    return result;
}

void ms_register_matrix_call_fft_rfft() {
    register_matrix_call("fft_rfft", &handle_fft_rfft);
}

} // namespace ms::interp
