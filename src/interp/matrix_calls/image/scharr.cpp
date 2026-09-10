#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_scharr(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "scharr" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto edge = eval_scharr(*matrix);
        if (!edge) {
            return std::unexpected(edge.error());
        }
        result = *edge;
    }

    return result;
}

void ms_register_matrix_call_scharr() {
    register_matrix_call("scharr", &handle_scharr);
}

} // namespace ms::interp
