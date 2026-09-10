#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_combo_next_perm(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "combo_next_perm" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto perm = eval_combo_next_perm(*matrix);
        if (!perm) {
            return std::unexpected(perm.error());
        }
        result = *perm;
    }

    return result;
}

void ms_register_matrix_call_combo_next_perm() {
    register_matrix_call("combo_next_perm", &handle_combo_next_perm);
}

} // namespace ms::interp
