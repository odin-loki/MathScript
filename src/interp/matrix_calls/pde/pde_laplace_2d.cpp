#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_pde_laplace_2d(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "pde_laplace_2d" && assign.args.size() == 3) {
        auto nx_val = ctx.parse_scalar_arg(assign.args[0], "pde_laplace_2d");
        if (!nx_val) {
            return std::unexpected(nx_val.error());
        }
        auto ny_val = ctx.parse_scalar_arg(assign.args[1], "pde_laplace_2d");
        if (!ny_val) {
            return std::unexpected(ny_val.error());
        }
        const int nx = static_cast<int>(*nx_val);
        const int ny = static_cast<int>(*ny_val);
        if (nx < 1 || ny < 1 || *nx_val != nx || *ny_val != ny) {
            return std::unexpected(
                DomainError{"pde_laplace_2d", "expected positive integer nx and ny"});
        }
        auto boundary_m = ctx.resolve_operand(assign.args[2]);
        if (!boundary_m) {
            return std::unexpected(boundary_m.error());
        }
        result = eval_pde_laplace_2d(nx, ny, *boundary_m);
    }

    return result;
}

void ms_register_matrix_call_pde_laplace_2d() {
    register_matrix_call("pde_laplace_2d", &handle_pde_laplace_2d);
}

} // namespace ms::interp
