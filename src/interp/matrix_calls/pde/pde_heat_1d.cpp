#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_pde_heat_1d(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "pde_heat_1d" && assign.args.size() == 5) {
        auto x0_m = ctx.resolve_operand(assign.args[0]);
        if (!x0_m) {
            return std::unexpected(x0_m.error());
        }
        auto alpha = ctx.parse_scalar_arg(assign.args[1], "pde_heat_1d");
        if (!alpha) {
            return std::unexpected(alpha.error());
        }
        auto dx = ctx.parse_scalar_arg(assign.args[2], "pde_heat_1d");
        if (!dx) {
            return std::unexpected(dx.error());
        }
        auto dt = ctx.parse_scalar_arg(assign.args[3], "pde_heat_1d");
        if (!dt) {
            return std::unexpected(dt.error());
        }
        auto steps_val = ctx.parse_scalar_arg(assign.args[4], "pde_heat_1d");
        if (!steps_val) {
            return std::unexpected(steps_val.error());
        }
        const int steps_i = static_cast<int>(*steps_val);
        if (steps_i < 0 || *steps_val != steps_i) {
            return std::unexpected(
                DomainError{"pde_heat_1d", "expected non-negative integer steps"});
        }
        result = eval_pde_heat_1d(*x0_m, *alpha, *dx, *dt, static_cast<std::size_t>(steps_i));
    }

    return result;
}

void ms_register_matrix_call_pde_heat_1d() {
    register_matrix_call("pde_heat_1d", &handle_pde_heat_1d);
}

} // namespace ms::interp
