#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_topo_simplicial_counts(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "topo_simplicial_counts" && assign.args.size() == 1) {
        auto sc = ctx.resolve_operand(assign.args[0]);
        if (!sc) {
            return std::unexpected(sc.error());
        }
        result = eval_topo_simplicial_counts(*sc);
    }

    return result;
}

void ms_register_matrix_call_topo_simplicial_counts() {
    register_matrix_call("topo_simplicial_counts", &handle_topo_simplicial_counts);
}

} // namespace ms::interp
