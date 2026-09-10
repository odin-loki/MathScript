#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_topo_simplicial_betti(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "topo_simplicial_betti" && assign.args.size() == 1) {
        auto sc = ctx.resolve_operand(assign.args[0]);
        if (!sc) {
            return std::unexpected(sc.error());
        }
        result = eval_topo_simplicial_betti(*sc);
    }

    return result;
}

void ms_register_matrix_call_topo_simplicial_betti() {
    register_matrix_call("topo_simplicial_betti", &handle_topo_simplicial_betti);
}

} // namespace ms::interp
