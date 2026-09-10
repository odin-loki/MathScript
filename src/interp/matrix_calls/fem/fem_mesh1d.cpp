#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_fem_mesh1d(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "fem_mesh1d" && assign.args.size() == 3) {
        auto a = ctx.parse_scalar_arg(assign.args[0], "fem_mesh1d");
        if (!a) {
            return std::unexpected(a.error());
        }
        auto b = ctx.parse_scalar_arg(assign.args[1], "fem_mesh1d");
        if (!b) {
            return std::unexpected(b.error());
        }
        auto n_val = ctx.parse_scalar_arg(assign.args[2], "fem_mesh1d");
        if (!n_val) {
            return std::unexpected(n_val.error());
        }
        auto n_i = ctx.parse_positive_size_arg(*n_val, "fem_mesh1d", "expected positive integer n_elements");
        if (!n_i) {
            return std::unexpected(n_i.error());
        }
        result = eval_fem_mesh1d(*a, *b, *n_i);
    }

    return result;
}

void ms_register_matrix_call_fem_mesh1d() {
    register_matrix_call("fem_mesh1d", &handle_fem_mesh1d);
}

} // namespace ms::interp
