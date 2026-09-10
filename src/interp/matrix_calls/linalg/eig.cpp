#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_eig(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "eig" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto decomp = eig(*matrix);
        if (!decomp) {
            return std::unexpected(decomp.error());
        }
        result = decomp->values;
    }

    return result;
}

void ms_register_matrix_call_eig() {
    register_matrix_call("eig", &handle_eig);
}

} // namespace ms::interp
