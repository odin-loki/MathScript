#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_pde_heat_2d_cn_adi(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "pde_heat_2d_cn_adi" && assign.args.size() == 6) {
        auto u0_m = ctx.resolve_operand(assign.args[0]);
        if (!u0_m) {
            return std::unexpected(u0_m.error());
        }
        auto alpha = ctx.parse_scalar_arg(assign.args[1], "pde_heat_2d_cn_adi");
        if (!alpha) {
            return std::unexpected(alpha.error());
        }
        auto dx = ctx.parse_scalar_arg(assign.args[2], "pde_heat_2d_cn_adi");
        if (!dx) {
            return std::unexpected(dx.error());
        }
        auto dy = ctx.parse_scalar_arg(assign.args[3], "pde_heat_2d_cn_adi");
        if (!dy) {
            return std::unexpected(dy.error());
        }
        auto dt = ctx.parse_scalar_arg(assign.args[4], "pde_heat_2d_cn_adi");
        if (!dt) {
            return std::unexpected(dt.error());
        }
        auto steps_val = ctx.parse_scalar_arg(assign.args[5], "pde_heat_2d_cn_adi");
        if (!steps_val) {
            return std::unexpected(steps_val.error());
        }
        const int steps_i = static_cast<int>(*steps_val);
        if (steps_i < 0 || *steps_val != steps_i) {
            return std::unexpected(
                DomainError{"pde_heat_2d_cn_adi", "expected non-negative integer steps"});
        }
        result = eval_pde_heat_2d_cn_adi(*u0_m, *alpha, *dx, *dy, *dt,
                                         static_cast<std::size_t>(steps_i));
    }

    return result;
}

void ms_register_matrix_call_pde_heat_2d_cn_adi() {
    register_matrix_call("pde_heat_2d_cn_adi", &handle_pde_heat_2d_cn_adi);
}

} // namespace ms::interp
