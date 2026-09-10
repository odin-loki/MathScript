#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_prewitt(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "prewitt" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto edge = eval_prewitt(*matrix);
        if (!edge) {
            return std::unexpected(edge.error());
        }
        result = *edge;
    }

    return result;
}

void ms_register_matrix_call_prewitt() {
    register_matrix_call("prewitt", &handle_prewitt);
}

} // namespace ms::interp
