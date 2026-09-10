#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_topo_pairwise_distances(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "topo_pairwise_distances" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto dist = eval_topo_pairwise_distances(*matrix);
        if (!dist) {
            return std::unexpected(dist.error());
        }
        result = *dist;
    }

    return result;
}

void ms_register_matrix_call_topo_pairwise_distances() {
    register_matrix_call("topo_pairwise_distances", &handle_topo_pairwise_distances);
}

} // namespace ms::interp
