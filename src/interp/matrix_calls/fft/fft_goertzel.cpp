#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_fft_goertzel(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "fft_goertzel" && assign.args.size() == 3) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto f_val = ctx.parse_scalar_arg(assign.args[1], "fft_goertzel");
        if (!f_val) {
            return std::unexpected(f_val.error());
        }
        auto fs_val = ctx.parse_scalar_arg(assign.args[2], "fft_goertzel");
        if (!fs_val) {
            return std::unexpected(fs_val.error());
        }
        result = eval_fft_goertzel(*matrix, *f_val, *fs_val);
    }

    return result;
}

void ms_register_matrix_call_fft_goertzel() {
    register_matrix_call("fft_goertzel", &handle_fft_goertzel);
}

} // namespace ms::interp
