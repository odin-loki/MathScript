#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_dist_gmres(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "dist_gmres" && assign.args.size() == 2) {
        auto left = ctx.resolve_operand(assign.args[0]);
        if (!left) {
            return std::unexpected(left.error());
        }
        auto right = ctx.resolve_operand(assign.args[1]);
        if (!right) {
            return std::unexpected(right.error());
        }
        result = eval_dist_gmres(*left, *right);
    }

    return result;
}

void ms_register_matrix_call_dist_gmres() {
    register_matrix_call("dist_gmres", &handle_dist_gmres);
}

} // namespace ms::interp
