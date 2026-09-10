#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_cfd_grid2d(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "cfd_grid2d" && assign.args.size() == 6) {
        auto x0 = ctx.parse_scalar_arg(assign.args[0], "cfd_grid2d");
        if (!x0) {
            return std::unexpected(x0.error());
        }
        auto x1 = ctx.parse_scalar_arg(assign.args[1], "cfd_grid2d");
        if (!x1) {
            return std::unexpected(x1.error());
        }
        auto y0 = ctx.parse_scalar_arg(assign.args[2], "cfd_grid2d");
        if (!y0) {
            return std::unexpected(y0.error());
        }
        auto y1 = ctx.parse_scalar_arg(assign.args[3], "cfd_grid2d");
        if (!y1) {
            return std::unexpected(y1.error());
        }
        auto nx_val = ctx.parse_scalar_arg(assign.args[4], "cfd_grid2d");
        if (!nx_val) {
            return std::unexpected(nx_val.error());
        }
        auto ny_val = ctx.parse_scalar_arg(assign.args[5], "cfd_grid2d");
        if (!ny_val) {
            return std::unexpected(ny_val.error());
        }
        const int nx_i = static_cast<int>(*nx_val);
        const int ny_i = static_cast<int>(*ny_val);
        if (nx_i < 0 || ny_i < 0 || *nx_val != nx_i || *ny_val != ny_i) {
            return std::unexpected(
                DomainError{"cfd_grid2d", "expected non-negative integer nx and ny"});
        }
        result = eval_cfd_grid2d(*x0, *x1, *y0, *y1, static_cast<std::size_t>(nx_i),
                                 static_cast<std::size_t>(ny_i));
    }

    return result;
}

void ms_register_matrix_call_cfd_grid2d() {
    register_matrix_call("cfd_grid2d", &handle_cfd_grid2d);
}

} // namespace ms::interp
