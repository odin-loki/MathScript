#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_imadjust(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "imadjust" &&
               (assign.args.size() == 3 || assign.args.size() == 5)) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        double in_lo = 0.0;
        double in_hi = 0.0;
        double out_lo = 0.0;
        double out_hi = 1.0;
        if (!parse_number(assign.args[1], in_lo) || !parse_number(assign.args[2], in_hi)) {
            return std::unexpected(
                DomainError{"imadjust", "expected imadjust(M, in_lo, in_hi[, out_lo, out_hi])"});
        }
        if (assign.args.size() == 5) {
            if (!parse_number(assign.args[3], out_lo) || !parse_number(assign.args[4], out_hi)) {
                return std::unexpected(DomainError{
                    "imadjust", "expected imadjust(M, in_lo, in_hi[, out_lo, out_hi])"});
            }
        }
        auto gray = matrix_to_gray_image(*matrix);
        if (!gray) {
            return std::unexpected(gray.error());
        }
        result = gray_image_to_matrix(image::imadjust(
            *gray, static_cast<float>(in_lo), static_cast<float>(in_hi),
            static_cast<float>(out_lo), static_cast<float>(out_hi)));
    }

    return result;
}

void ms_register_matrix_call_imadjust() {
    register_matrix_call("imadjust", &handle_imadjust);
}

} // namespace ms::interp
