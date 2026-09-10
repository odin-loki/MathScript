#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_hsv2rgb(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "hsv2rgb" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto hsv = matrix_to_rgb_image(*matrix);
        if (!hsv) {
            return std::unexpected(hsv.error());
        }
        result = rgb_image_to_matrix(image::hsv2rgb(*hsv));
    }

    return result;
}

void ms_register_matrix_call_hsv2rgb() {
    register_matrix_call("hsv2rgb", &handle_hsv2rgb);
}

} // namespace ms::interp
