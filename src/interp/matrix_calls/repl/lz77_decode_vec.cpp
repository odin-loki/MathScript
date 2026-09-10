#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_lz77_decode_vec(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "lz77_decode_vec" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto decoded = eval_lz77_decode_vec(*matrix);
        if (!decoded) {
            return std::unexpected(decoded.error());
        }
        result = *decoded;
    }

    return result;
}

void ms_register_matrix_call_lz77_decode_vec() {
    register_matrix_call("lz77_decode_vec", &handle_lz77_decode_vec);
}

} // namespace ms::interp
