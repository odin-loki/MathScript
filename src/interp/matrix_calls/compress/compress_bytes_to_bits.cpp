#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_compress_bytes_to_bits(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "compress_bytes_to_bits" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto decoded = eval_compress_bytes_to_bits(*matrix);
        if (!decoded) {
            return std::unexpected(decoded.error());
        }
        result = *decoded;
    }

    return result;
}

void ms_register_matrix_call_compress_bytes_to_bits() {
    register_matrix_call("compress_bytes_to_bits", &handle_compress_bytes_to_bits);
}

} // namespace ms::interp
