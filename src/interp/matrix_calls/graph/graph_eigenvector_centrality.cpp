#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_graph_eigenvector_centrality(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "graph_eigenvector_centrality" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto ec = eval_graph_eigenvector_centrality(*matrix);
        if (!ec) {
            return std::unexpected(ec.error());
        }
        result = *ec;
    }

    return result;
}

void ms_register_matrix_call_graph_eigenvector_centrality() {
    register_matrix_call("graph_eigenvector_centrality", &handle_graph_eigenvector_centrality);
}

} // namespace ms::interp
