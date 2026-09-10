#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_graph_mst_prim(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "graph_mst_prim" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto edges = eval_graph_mst_prim(*matrix);
        if (!edges) {
            return std::unexpected(edges.error());
        }
        result = *edges;
    }

    return result;
}

void ms_register_matrix_call_graph_mst_prim() {
    register_matrix_call("graph_mst_prim", &handle_graph_mst_prim);
}

} // namespace ms::interp
