#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_graph_betweenness(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "graph_betweenness" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto bc = eval_graph_betweenness(*matrix);
        if (!bc) {
            return std::unexpected(bc.error());
        }
        result = *bc;
    }

    return result;
}

void ms_register_matrix_call_graph_betweenness() {
    register_matrix_call("graph_betweenness", &handle_graph_betweenness);
}

} // namespace ms::interp
