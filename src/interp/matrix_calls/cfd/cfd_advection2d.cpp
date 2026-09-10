#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_cfd_advection2d(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "cfd_advection2d" && assign.args.size() == 6) {
        auto nx_val = ctx.parse_scalar_arg(assign.args[0], "cfd_advection2d");
        if (!nx_val) {
            return std::unexpected(nx_val.error());
        }
        auto ny_val = ctx.parse_scalar_arg(assign.args[1], "cfd_advection2d");
        if (!ny_val) {
            return std::unexpected(ny_val.error());
        }
        auto vx_val = ctx.parse_scalar_arg(assign.args[2], "cfd_advection2d");
        if (!vx_val) {
            return std::unexpected(vx_val.error());
        }
        auto vy_val = ctx.parse_scalar_arg(assign.args[3], "cfd_advection2d");
        if (!vy_val) {
            return std::unexpected(vy_val.error());
        }
        auto t_end_val = ctx.parse_scalar_arg(assign.args[4], "cfd_advection2d");
        if (!t_end_val) {
            return std::unexpected(t_end_val.error());
        }
        auto dt_val = ctx.parse_scalar_arg(assign.args[5], "cfd_advection2d");
        if (!dt_val) {
            return std::unexpected(dt_val.error());
        }
        const int nx_i = static_cast<int>(*nx_val);
        const int ny_i = static_cast<int>(*ny_val);
        if (nx_i < 0 || ny_i < 0 || *nx_val != nx_i || *ny_val != ny_i) {
            return std::unexpected(
                DomainError{"cfd_advection2d", "expected non-negative integer nx and ny"});
        }
        result = eval_cfd_advection2d(static_cast<std::size_t>(nx_i), static_cast<std::size_t>(ny_i),
                                      *vx_val, *vy_val, *t_end_val, *dt_val);
    }

    return result;
}

void ms_register_matrix_call_cfd_advection2d() {
    register_matrix_call("cfd_advection2d", &handle_cfd_advection2d);
}

} // namespace ms::interp
