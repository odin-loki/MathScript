#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_geo_triangulate_polygon(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "geo_triangulate_polygon" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto tris = eval_geo_triangulate_polygon(*matrix);
        if (!tris) {
            return std::unexpected(tris.error());
        }
        result = *tris;
    }

    return result;
}

void ms_register_matrix_call_geo_triangulate_polygon() {
    register_matrix_call("geo_triangulate_polygon", &handle_geo_triangulate_polygon);
}

} // namespace ms::interp
