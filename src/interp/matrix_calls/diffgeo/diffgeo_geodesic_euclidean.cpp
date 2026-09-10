#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_diffgeo_geodesic_euclidean(Interpreter& /*interp*/, const MatrixCallAssign& assign) {
    using namespace detail;

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "diffgeo_geodesic_euclidean" && assign.args.size() == 5) {
        double x0 = 0.0;
        double y0 = 0.0;
        double vx = 0.0;
        double vy = 0.0;
        double s_end = 0.0;
        if (!parse_number(assign.args[0], x0) || !parse_number(assign.args[1], y0) ||
            !parse_number(assign.args[2], vx) || !parse_number(assign.args[3], vy) ||
            !parse_number(assign.args[4], s_end)) {
            return std::unexpected(DomainError{
                "diffgeo_geodesic_euclidean",
                "expected diffgeo_geodesic_euclidean(x0,y0,vx,vy,s_end)"});
        }
        auto traj = eval_diffgeo_geodesic_euclidean(x0, y0, vx, vy, s_end);
        if (!traj) {
            return std::unexpected(traj.error());
        }
        result = *traj;
    }

    return result;
}

void ms_register_matrix_call_diffgeo_geodesic_euclidean() {
    register_matrix_call("diffgeo_geodesic_euclidean", &handle_diffgeo_geodesic_euclidean);
}

} // namespace ms::interp
