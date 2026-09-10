#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_topo_cech_complex(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "topo_cech_complex" &&
               (assign.args.size() == 2 || assign.args.size() == 3)) {
        auto dist = ctx.resolve_operand(assign.args[0]);
        if (!dist) {
            return std::unexpected(dist.error());
        }
        auto eps = ctx.parse_scalar_arg(assign.args[1], "topo_cech_complex");
        if (!eps) {
            return std::unexpected(eps.error());
        }
        int max_dim = 2;
        if (assign.args.size() == 3) {
            auto md = ctx.parse_scalar_arg(assign.args[2], "topo_cech_complex");
            if (!md) {
                return std::unexpected(md.error());
            }
            max_dim = static_cast<int>(*md);
            if (*md != max_dim || max_dim < 0) {
                return std::unexpected(
                    DomainError{"topo_cech_complex", "expected non-negative integer max_dim"});
            }
        }
        result = eval_topo_cech_complex(*dist, *eps, max_dim);
    }

    return result;
}

void ms_register_matrix_call_topo_cech_complex() {
    register_matrix_call("topo_cech_complex", &handle_topo_cech_complex);
}

} // namespace ms::interp
