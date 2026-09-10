#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_graph_connected_components(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "graph_connected_components" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto cc = eval_graph_connected_components(*matrix);
        if (!cc) {
            return std::unexpected(cc.error());
        }
        result = *cc;
    }

    return result;
}

void ms_register_matrix_call_graph_connected_components() {
    register_matrix_call("graph_connected_components", &handle_graph_connected_components);
}

} // namespace ms::interp
