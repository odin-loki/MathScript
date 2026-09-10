#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_poly_roots(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "poly_roots" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto roots = eval_poly_roots(*matrix);
        if (!roots) {
            return std::unexpected(roots.error());
        }
        result = *roots;
    }

    return result;
}

void ms_register_matrix_call_poly_roots() {
    register_matrix_call("poly_roots", &handle_poly_roots);
}

} // namespace ms::interp
