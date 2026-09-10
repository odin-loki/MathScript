#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_topo_witness_complex(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "topo_witness_complex" &&
               (assign.args.size() == 3 || assign.args.size() == 4)) {
        auto P = ctx.resolve_operand(assign.args[0]);
        if (!P) {
            return std::unexpected(P.error());
        }
        auto landmarks = ctx.resolve_operand(assign.args[1]);
        if (!landmarks) {
            return std::unexpected(landmarks.error());
        }
        auto eps = ctx.parse_scalar_arg(assign.args[2], "topo_witness_complex");
        if (!eps) {
            return std::unexpected(eps.error());
        }
        int max_dim = 2;
        if (assign.args.size() == 4) {
            auto md = ctx.parse_scalar_arg(assign.args[3], "topo_witness_complex");
            if (!md) {
                return std::unexpected(md.error());
            }
            max_dim = static_cast<int>(*md);
            if (max_dim < 0 || *md != max_dim) {
                return std::unexpected(
                    DomainError{"topo_witness_complex", "expected non-negative integer max_dim"});
            }
        }
        result = eval_topo_witness_complex(*P, *landmarks, *eps, max_dim);
    }

    return result;
}

void ms_register_matrix_call_topo_witness_complex() {
    register_matrix_call("topo_witness_complex", &handle_topo_witness_complex);
}

} // namespace ms::interp
