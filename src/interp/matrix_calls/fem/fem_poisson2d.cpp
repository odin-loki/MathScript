#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_fem_poisson2d(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "fem_poisson2d" && assign.args.size() == 2) {
        auto nx_val = ctx.parse_scalar_arg(assign.args[0], "fem_poisson2d");
        if (!nx_val) {
            return std::unexpected(nx_val.error());
        }
        auto ny_val = ctx.parse_scalar_arg(assign.args[1], "fem_poisson2d");
        if (!ny_val) {
            return std::unexpected(ny_val.error());
        }
        const int nx_i = static_cast<int>(*nx_val);
        const int ny_i = static_cast<int>(*ny_val);
        if (nx_i < 0 || ny_i < 0 || *nx_val != nx_i || *ny_val != ny_i) {
            return std::unexpected(
                DomainError{"fem_poisson2d", "expected non-negative integer nx and ny"});
        }
        result = eval_fem_poisson2d(static_cast<std::size_t>(nx_i), static_cast<std::size_t>(ny_i));
    }

    return result;
}

void ms_register_matrix_call_fem_poisson2d() {
    register_matrix_call("fem_poisson2d", &handle_fem_poisson2d);
}

} // namespace ms::interp
