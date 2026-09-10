#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_mat_reshape(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    if (assign.callee != "mat_reshape" || assign.args.size() != 3) {
        return std::unexpected(DomainError{"mat_reshape", "expected mat_reshape(A, rows, cols)"});
    }
    auto A = ctx.resolve_operand(assign.args[0]);
    if (!A) {
        return std::unexpected(A.error());
    }
    auto rows = ctx.parse_scalar_arg(assign.args[1], "mat_reshape");
    if (!rows) {
        return std::unexpected(rows.error());
    }
    auto cols = ctx.parse_scalar_arg(assign.args[2], "mat_reshape");
    if (!cols) {
        return std::unexpected(cols.error());
    }
    return eval_mat_reshape(*A, *rows, *cols);
}

void ms_register_matrix_call_mat_reshape() {
    register_matrix_call("mat_reshape", &handle_mat_reshape);
}

} // namespace ms::interp
