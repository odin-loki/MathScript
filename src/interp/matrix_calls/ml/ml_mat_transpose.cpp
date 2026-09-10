#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_ml_mat_transpose(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "ml_mat_transpose" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto At = eval_ml_mat_transpose(*matrix);
        if (!At) {
            return std::unexpected(At.error());
        }
        result = *At;
    }

    return result;
}

void ms_register_matrix_call_ml_mat_transpose() {
    register_matrix_call("ml_mat_transpose", &handle_ml_mat_transpose);
}

} // namespace ms::interp
