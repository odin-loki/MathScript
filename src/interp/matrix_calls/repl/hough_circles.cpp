#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_hough_circles(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "hough_circles" &&
               (assign.args.size() == 1 || assign.args.size() == 3)) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        double r_min = 5.0;
        double r_max = 50.0;
        if (assign.args.size() == 3) {
            if (!parse_number(assign.args[1], r_min) || !parse_number(assign.args[2], r_max)) {
                return std::unexpected(DomainError{
                    "hough_circles", "expected hough_circles(M) or hough_circles(M, r_min, r_max)"});
            }
        }
        result = eval_hough_circles(*matrix, 0.5, r_min, r_max, 1, 30);
    }

    return result;
}

void ms_register_matrix_call_hough_circles() {
    register_matrix_call("hough_circles", &handle_hough_circles);
}

} // namespace ms::interp
