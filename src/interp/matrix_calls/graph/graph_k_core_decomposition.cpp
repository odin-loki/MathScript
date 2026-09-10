#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_graph_k_core_decomposition(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "graph_k_core_decomposition" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto cores = eval_graph_k_core_decomposition(*matrix);
        if (!cores) {
            return std::unexpected(cores.error());
        }
        result = *cores;
    }

    return result;
}

void ms_register_matrix_call_graph_k_core_decomposition() {
    register_matrix_call("graph_k_core_decomposition", &handle_graph_k_core_decomposition);
}

} // namespace ms::interp
