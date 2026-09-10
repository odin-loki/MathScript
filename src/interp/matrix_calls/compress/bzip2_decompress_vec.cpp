#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_bzip2_decompress_vec(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "bzip2_decompress_vec" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto decompressed = eval_bzip2_decompress_vec(*matrix);
        if (!decompressed) {
            return std::unexpected(decompressed.error());
        }
        result = *decompressed;
    }

    return result;
}

void ms_register_matrix_call_bzip2_decompress_vec() {
    register_matrix_call("bzip2_decompress_vec", &handle_bzip2_decompress_vec);
}

} // namespace ms::interp
