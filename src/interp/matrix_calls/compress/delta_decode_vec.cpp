#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_delta_decode_vec(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "delta_decode_vec" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        if (matrix->cols() != 1) {
            return std::unexpected(
                DomainError{"delta_decode_vec", "expected Nx1 encoded byte vector"});
        }
        compress::Bytes encoded;
        encoded.reserve(matrix->rows());
        for (size_t i = 0; i < matrix->rows(); ++i) {
            const double v = (*matrix)(i, 0);
            if (v < 0.0 || v > 255.0 || std::floor(v) != v) {
                return std::unexpected(
                    DomainError{"delta_decode_vec", "encoded values must be uint8 in [0,255]"});
            }
            encoded.push_back(static_cast<uint8_t>(v));
        }
        result = bytes_to_matrix_col(compress::delta_decode(encoded));
    }

    return result;
}

void ms_register_matrix_call_delta_decode_vec() {
    register_matrix_call("delta_decode_vec", &handle_delta_decode_vec);
}

} // namespace ms::interp
