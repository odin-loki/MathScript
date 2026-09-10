#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_geo_kdtree_range(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if ((assign.callee == "geo_kdtree_knn" || assign.callee == "geo_kdtree_range") &&
               assign.args.size() == 4) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        double qx = 0.0;
        double qy = 0.0;
        double arg3 = 0.0;
        if (!parse_number(assign.args[1], qx) || !parse_number(assign.args[2], qy) ||
            !parse_number(assign.args[3], arg3)) {
            return std::unexpected(DomainError{
                assign.callee,
                assign.callee == "geo_kdtree_knn" ? "expected geo_kdtree_knn(P, x, y, k)"
                                                  : "expected geo_kdtree_range(P, x, y, r)"});
        }
        if (assign.callee == "geo_kdtree_knn") {
            result = eval_geo_kdtree_knn(*matrix, qx, qy, arg3);
        } else {
            result = eval_geo_kdtree_range(*matrix, qx, qy, arg3);
        }
    }

    return result;
}

void ms_register_matrix_call_geo_kdtree_range() {
    register_matrix_call("geo_kdtree_range", &handle_geo_kdtree_range);
}

} // namespace ms::interp
