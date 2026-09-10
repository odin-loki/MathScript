#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_graph_floyd_warshall(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "graph_floyd_warshall" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto dist = eval_graph_floyd_warshall(*matrix);
        if (!dist) {
            return std::unexpected(dist.error());
        }
        result = *dist;
    }

    return result;
}

void ms_register_matrix_call_graph_floyd_warshall() {
    register_matrix_call("graph_floyd_warshall", &handle_graph_floyd_warshall);
}

} // namespace ms::interp
