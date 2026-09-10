#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_graph_scc(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "graph_scc" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto scc = eval_graph_scc(*matrix);
        if (!scc) {
            return std::unexpected(scc.error());
        }
        result = *scc;
    }

    return result;
}

void ms_register_matrix_call_graph_scc() {
    register_matrix_call("graph_scc", &handle_graph_scc);
}

} // namespace ms::interp
