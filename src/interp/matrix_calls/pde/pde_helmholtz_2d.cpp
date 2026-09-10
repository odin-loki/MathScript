#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_pde_helmholtz_2d(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "pde_helmholtz_2d" &&
               (assign.args.size() == 4 || assign.args.size() == 5)) {
        auto f_m = ctx.resolve_operand(assign.args[0]);
        if (!f_m) {
            return std::unexpected(f_m.error());
        }
        auto k = ctx.parse_scalar_arg(assign.args[1], "pde_helmholtz_2d");
        if (!k) {
            return std::unexpected(k.error());
        }
        auto dx = ctx.parse_scalar_arg(assign.args[2], "pde_helmholtz_2d");
        if (!dx) {
            return std::unexpected(dx.error());
        }
        auto dy = ctx.parse_scalar_arg(assign.args[3], "pde_helmholtz_2d");
        if (!dy) {
            return std::unexpected(dy.error());
        }
        if (assign.args.size() == 4) {
            result = eval_pde_helmholtz_2d(*f_m, *k, *dx, *dy);
        } else {
            auto g_m = ctx.resolve_operand(assign.args[4]);
            if (!g_m) {
                return std::unexpected(g_m.error());
            }
            result = eval_pde_helmholtz_2d(*f_m, *k, *dx, *dy, &(*g_m));
        }
    }

    return result;
}

void ms_register_matrix_call_pde_helmholtz_2d() {
    register_matrix_call("pde_helmholtz_2d", &handle_pde_helmholtz_2d);
}

} // namespace ms::interp
