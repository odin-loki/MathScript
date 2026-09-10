#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_medfilt2(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "medfilt2" && assign.args.size() == 2) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        double ksize_d = 0.0;
        if (!parse_number(assign.args[1], ksize_d)) {
            return std::unexpected(DomainError{"medfilt2", "expected medfilt2(M, ksize)"});
        }
        const int ksize = static_cast<int>(ksize_d);
        if (ksize < 1 || ksize_d != ksize || (ksize % 2) == 0) {
            return std::unexpected(
                DomainError{"medfilt2", "expected positive odd integer ksize"});
        }
        auto gray = matrix_to_gray_image(*matrix);
        if (!gray) {
            return std::unexpected(gray.error());
        }
        result = gray_image_to_matrix(image::medfilt2(*gray, ksize));
    }

    return result;
}

void ms_register_matrix_call_medfilt2() {
    register_matrix_call("medfilt2", &handle_medfilt2);
}

} // namespace ms::interp
