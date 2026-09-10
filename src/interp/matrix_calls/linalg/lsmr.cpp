#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_lsmr(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "lsmr" && assign.args.size() == 2) {
        auto left = ctx.resolve_operand(assign.args[0]);
        if (!left) {
            return std::unexpected(left.error());
        }
        auto right = ctx.resolve_operand(assign.args[1]);
        if (!right) {
            return std::unexpected(right.error());
        }
        result = lsmr(*left, *right);
    }

    return result;
}

void ms_register_matrix_call_lsmr() {
    register_matrix_call("lsmr", &handle_lsmr);
}

} // namespace ms::interp
