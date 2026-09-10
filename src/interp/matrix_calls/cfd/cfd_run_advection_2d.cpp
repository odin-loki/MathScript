#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_cfd_run_advection_2d(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "cfd_run_advection_2d" && assign.args.size() == 6) {
        auto grid = ctx.resolve_operand(assign.args[0]);
        if (!grid) {
            return std::unexpected(grid.error());
        }
        auto u = ctx.resolve_operand(assign.args[1]);
        if (!u) {
            return std::unexpected(u.error());
        }
        auto vx = ctx.parse_scalar_arg(assign.args[2], "cfd_run_advection_2d");
        if (!vx) {
            return std::unexpected(vx.error());
        }
        auto vy = ctx.parse_scalar_arg(assign.args[3], "cfd_run_advection_2d");
        if (!vy) {
            return std::unexpected(vy.error());
        }
        auto t_end = ctx.parse_scalar_arg(assign.args[4], "cfd_run_advection_2d");
        if (!t_end) {
            return std::unexpected(t_end.error());
        }
        auto dt = ctx.parse_scalar_arg(assign.args[5], "cfd_run_advection_2d");
        if (!dt) {
            return std::unexpected(dt.error());
        }
        result = eval_cfd_run_advection_2d(*grid, *u, *vx, *vy, *t_end, *dt);
    }

    return result;
}

void ms_register_matrix_call_cfd_run_advection_2d() {
    register_matrix_call("cfd_run_advection_2d", &handle_cfd_run_advection_2d);
}

} // namespace ms::interp
