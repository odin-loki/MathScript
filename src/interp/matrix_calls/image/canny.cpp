#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_canny(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "canny" && assign.args.size() == 3) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        double low = 0.0;
        double high = 0.0;
        if (!parse_number(assign.args[1], low) || !parse_number(assign.args[2], high)) {
            return std::unexpected(DomainError{"canny", "expected canny(M, low, high)"});
        }
        auto gray = matrix_to_gray_image(*matrix);
        if (!gray) {
            return std::unexpected(gray.error());
        }
        result = gray_image_to_matrix(
            image::canny(*gray, static_cast<float>(low), static_cast<float>(high)));
    }

    return result;
}

void ms_register_matrix_call_canny() {
    register_matrix_call("canny", &handle_canny);
}

} // namespace ms::interp
