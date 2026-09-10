#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_solve_sylvester(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "solve_sylvester" && assign.args.size() == 3) {
        auto A = ctx.resolve_operand(assign.args[0]);
        if (!A) {
            return std::unexpected(A.error());
        }
        auto B = ctx.resolve_operand(assign.args[1]);
        if (!B) {
            return std::unexpected(B.error());
        }
        auto C = ctx.resolve_operand(assign.args[2]);
        if (!C) {
            return std::unexpected(C.error());
        }
        result = solve_sylvester(*A, *B, *C);
    }

    return result;
}

void ms_register_matrix_call_solve_sylvester() {
    register_matrix_call("solve_sylvester", &handle_solve_sylvester);
}

} // namespace ms::interp
