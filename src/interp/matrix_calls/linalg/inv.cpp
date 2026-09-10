#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_inv(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "inv" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        // inv via LU solve: A * inv(A) = I
        const size_t n = matrix->rows();
        auto I = eye<double>(n);
        result = solve(*matrix, I);
    }

    return result;
}

void ms_register_matrix_call_inv() {
    register_matrix_call("inv", &handle_inv);
}

} // namespace ms::interp
