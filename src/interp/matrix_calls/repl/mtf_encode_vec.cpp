#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_mtf_encode_vec(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "mtf_encode_vec" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        result = bytes_to_matrix_col(compress::mtf_encode(matrix_to_bytes(*matrix)));
    }

    return result;
}

void ms_register_matrix_call_mtf_encode_vec() {
    register_matrix_call("mtf_encode_vec", &handle_mtf_encode_vec);
}

} // namespace ms::interp
