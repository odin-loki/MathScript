#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_wavelet_decompress_vec(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "wavelet_decompress_vec" && assign.args.size() == 1) {
        auto compressed = ctx.resolve_operand(assign.args[0]);
        if (!compressed) {
            return std::unexpected(compressed.error());
        }
        auto decoded = eval_wavelet_decompress_vec(*compressed);
        if (!decoded) {
            return std::unexpected(decoded.error());
        }
        result = *decoded;
    }

    return result;
}

void ms_register_matrix_call_wavelet_decompress_vec() {
    register_matrix_call("wavelet_decompress_vec", &handle_wavelet_decompress_vec);
}

} // namespace ms::interp
