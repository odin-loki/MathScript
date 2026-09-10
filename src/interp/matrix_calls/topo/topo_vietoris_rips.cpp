#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_topo_vietoris_rips(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "topo_vietoris_rips" &&
               (assign.args.size() == 2 || assign.args.size() == 3)) {
        auto dist = ctx.resolve_operand(assign.args[0]);
        if (!dist) {
            return std::unexpected(dist.error());
        }
        auto r = ctx.parse_scalar_arg(assign.args[1], "topo_vietoris_rips");
        if (!r) {
            return std::unexpected(r.error());
        }
        int max_dim = 2;
        if (assign.args.size() == 3) {
            auto md = ctx.parse_scalar_arg(assign.args[2], "topo_vietoris_rips");
            if (!md) {
                return std::unexpected(md.error());
            }
            max_dim = static_cast<int>(*md);
            if (*md != max_dim || max_dim < 0) {
                return std::unexpected(
                    DomainError{"topo_vietoris_rips", "expected non-negative integer max_dim"});
            }
        }
        result = eval_topo_vietoris_rips(*dist, *r, max_dim);
    }

    return result;
}

void ms_register_matrix_call_topo_vietoris_rips() {
    register_matrix_call("topo_vietoris_rips", &handle_topo_vietoris_rips);
}

} // namespace ms::interp
