#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_graph_eulerian_path(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "graph_eulerian_path" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto path = eval_graph_eulerian_path(*matrix);
        if (!path) {
            return std::unexpected(path.error());
        }
        result = *path;
    }

    return result;
}

void ms_register_matrix_call_graph_eulerian_path() {
    register_matrix_call("graph_eulerian_path", &handle_graph_eulerian_path);
}

} // namespace ms::interp
