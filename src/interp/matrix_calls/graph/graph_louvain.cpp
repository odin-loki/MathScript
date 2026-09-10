#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_graph_louvain(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "graph_louvain" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto comms = eval_graph_louvain(*matrix);
        if (!comms) {
            return std::unexpected(comms.error());
        }
        result = *comms;
    }

    return result;
}

void ms_register_matrix_call_graph_louvain() {
    register_matrix_call("graph_louvain", &handle_graph_louvain);
}

} // namespace ms::interp
