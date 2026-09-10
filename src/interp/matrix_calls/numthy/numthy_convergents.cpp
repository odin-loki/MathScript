#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_numthy_convergents(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "numthy_convergents" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto conv = eval_numthy_convergents(*matrix);
        if (!conv) {
            return std::unexpected(conv.error());
        }
        result = *conv;
    }

    return result;
}

void ms_register_matrix_call_numthy_convergents() {
    register_matrix_call("numthy_convergents", &handle_numthy_convergents);
}

} // namespace ms::interp
