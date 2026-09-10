#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_wavelet_compress_vec(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "wavelet_compress_vec" &&
               (assign.args.size() == 1 || assign.args.size() == 2)) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        double threshold = 0.0;
        if (assign.args.size() == 2) {
            auto thr = ctx.parse_scalar_arg(assign.args[1], "wavelet_compress_vec");
            if (!thr) {
                return std::unexpected(thr.error());
            }
            threshold = *thr;
        }
        auto compressed = eval_wavelet_compress_vec(*matrix, threshold);
        if (!compressed) {
            return std::unexpected(compressed.error());
        }
        result = *compressed;
    }

    return result;
}

void ms_register_matrix_call_wavelet_compress_vec() {
    register_matrix_call("wavelet_compress_vec", &handle_wavelet_compress_vec);
}

} // namespace ms::interp
