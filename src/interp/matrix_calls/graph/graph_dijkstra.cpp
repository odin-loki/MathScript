#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_graph_dijkstra(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "graph_dijkstra" && assign.args.size() == 2) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto source_val = ctx.parse_scalar_arg(assign.args[1], "graph_dijkstra");
        if (!source_val) {
            return std::unexpected(source_val.error());
        }
        const int source = static_cast<int>(*source_val);
        if (source < 0 || *source_val != source) {
            return std::unexpected(
                DomainError{"graph_dijkstra", "expected non-negative integer source"});
        }
        auto sp = eval_graph_dijkstra(*matrix, source);
        if (!sp) {
            return std::unexpected(sp.error());
        }
        result = *sp;
    }

    return result;
}

void ms_register_matrix_call_graph_dijkstra() {
    register_matrix_call("graph_dijkstra", &handle_graph_dijkstra);
}

} // namespace ms::interp
