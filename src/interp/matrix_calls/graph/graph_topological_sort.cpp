#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_graph_topological_sort(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "graph_topological_sort" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto order = eval_graph_topological_sort(*matrix);
        if (!order) {
            return std::unexpected(order.error());
        }
        result = *order;
    }

    return result;
}

void ms_register_matrix_call_graph_topological_sort() {
    register_matrix_call("graph_topological_sort", &handle_graph_topological_sort);
}

} // namespace ms::interp
