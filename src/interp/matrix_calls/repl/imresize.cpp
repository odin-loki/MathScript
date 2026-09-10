#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_imresize(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "imresize" && assign.args.size() == 3) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        double rows_d = 0.0;
        double cols_d = 0.0;
        if (!parse_number(assign.args[1], rows_d) || !parse_number(assign.args[2], cols_d)) {
            return std::unexpected(DomainError{"imresize", "expected imresize(M, rows, cols)"});
        }
        auto gray = matrix_to_gray_image(*matrix);
        if (!gray) {
            return std::unexpected(gray.error());
        }
        result = gray_image_to_matrix(
            image::imresize(*gray, static_cast<int>(rows_d), static_cast<int>(cols_d)));
    }

    return result;
}

void ms_register_matrix_call_imresize() {
    register_matrix_call("imresize", &handle_imresize);
}

} // namespace ms::interp
