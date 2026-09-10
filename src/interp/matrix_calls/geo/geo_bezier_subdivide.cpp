#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_geo_bezier_subdivide(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "geo_bezier_subdivide" && assign.args.size() == 2) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto t = ctx.parse_scalar_arg(assign.args[1], "geo_bezier_subdivide");
        if (!t) {
            return std::unexpected(t.error());
        }
        result = eval_geo_bezier_subdivide(*matrix, *t);
    }

    return result;
}

void ms_register_matrix_call_geo_bezier_subdivide() {
    register_matrix_call("geo_bezier_subdivide", &handle_geo_bezier_subdivide);
}

} // namespace ms::interp
