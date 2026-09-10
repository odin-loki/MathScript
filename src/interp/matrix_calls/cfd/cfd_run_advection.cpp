#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_cfd_run_advection(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "cfd_run_advection" && assign.args.size() == 5) {
        auto grid = ctx.resolve_operand(assign.args[0]);
        if (!grid) {
            return std::unexpected(grid.error());
        }
        auto u = ctx.resolve_operand(assign.args[1]);
        if (!u) {
            return std::unexpected(u.error());
        }
        auto v = ctx.parse_scalar_arg(assign.args[2], "cfd_run_advection");
        if (!v) {
            return std::unexpected(v.error());
        }
        auto t_end = ctx.parse_scalar_arg(assign.args[3], "cfd_run_advection");
        if (!t_end) {
            return std::unexpected(t_end.error());
        }
        auto dt = ctx.parse_scalar_arg(assign.args[4], "cfd_run_advection");
        if (!dt) {
            return std::unexpected(dt.error());
        }
        result = eval_cfd_run_advection(*grid, *u, *v, *t_end, *dt);
    }

    return result;
}

void ms_register_matrix_call_cfd_run_advection() {
    register_matrix_call("cfd_run_advection", &handle_cfd_run_advection);
}

} // namespace ms::interp
