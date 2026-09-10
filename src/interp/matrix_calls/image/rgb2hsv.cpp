#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_rgb2hsv(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "rgb2hsv" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto rgb = matrix_to_rgb_image(*matrix);
        if (!rgb) {
            return std::unexpected(rgb.error());
        }
        result = rgb_image_to_matrix(image::rgb2hsv(*rgb));
    }

    return result;
}

void ms_register_matrix_call_rgb2hsv() {
    register_matrix_call("rgb2hsv", &handle_rgb2hsv);
}

} // namespace ms::interp
