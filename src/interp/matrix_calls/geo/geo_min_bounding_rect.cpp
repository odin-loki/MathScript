#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_geo_min_bounding_rect(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "geo_min_bounding_rect" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto rect = eval_geo_min_bounding_rect(*matrix);
        if (!rect) {
            return std::unexpected(rect.error());
        }
        result = *rect;
    }

    return result;
}

void ms_register_matrix_call_geo_min_bounding_rect() {
    register_matrix_call("geo_min_bounding_rect", &handle_geo_min_bounding_rect);
}

} // namespace ms::interp
