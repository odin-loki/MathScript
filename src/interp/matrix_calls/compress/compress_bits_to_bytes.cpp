#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_compress_bits_to_bytes(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "compress_bits_to_bytes" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto encoded = eval_compress_bits_to_bytes(*matrix);
        if (!encoded) {
            return std::unexpected(encoded.error());
        }
        result = *encoded;
    }

    return result;
}

void ms_register_matrix_call_compress_bits_to_bytes() {
    register_matrix_call("compress_bits_to_bytes", &handle_compress_bits_to_bytes);
}

} // namespace ms::interp
