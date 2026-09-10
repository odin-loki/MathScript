#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_topo_betti_curve(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "topo_betti_curve" && assign.args.size() == 3) {
        auto dist_m = ctx.resolve_operand(assign.args[0]);
        if (!dist_m) {
            return std::unexpected(dist_m.error());
        }
        auto thresholds_m = ctx.resolve_operand(assign.args[1]);
        if (!thresholds_m) {
            return std::unexpected(thresholds_m.error());
        }
        double max_dim_d = 0.0;
        if (!parse_number(assign.args[2], max_dim_d)) {
            auto md_expr = eval_scalar_expr(ctx.state(), assign.args[2]);
            if (!md_expr) {
                return std::unexpected(md_expr.error());
            }
            max_dim_d = *md_expr;
        }
        const int max_dim = static_cast<int>(max_dim_d);
        if (max_dim < 0 || max_dim_d != max_dim) {
            return std::unexpected(
                DomainError{"topo_betti_curve", "expected non-negative integer max_dim"});
        }
        auto curve = eval_topo_betti_curve(*dist_m, *thresholds_m, max_dim);
        if (!curve) {
            return std::unexpected(curve.error());
        }
        result = *curve;
    }

    return result;
}

void ms_register_matrix_call_topo_betti_curve() {
    register_matrix_call("topo_betti_curve", &handle_topo_betti_curve);
}

} // namespace ms::interp
