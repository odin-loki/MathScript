#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_graph_euler_circuit(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "graph_euler_circuit" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto circuit = eval_graph_euler_circuit(*matrix);
        if (!circuit) {
            return std::unexpected(circuit.error());
        }
        result = *circuit;
    }

    return result;
}

void ms_register_matrix_call_graph_euler_circuit() {
    register_matrix_call("graph_euler_circuit", &handle_graph_euler_circuit);
}

} // namespace ms::interp
