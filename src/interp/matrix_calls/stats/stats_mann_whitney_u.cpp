#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_stats_mann_whitney_u(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "stats_mann_whitney_u" && assign.args.size() == 2) {
        auto a = ctx.resolve_operand(assign.args[0]);
        if (!a) {
            return std::unexpected(a.error());
        }
        auto b = ctx.resolve_operand(assign.args[1]);
        if (!b) {
            return std::unexpected(b.error());
        }
        result = eval_stats_mann_whitney_u(*a, *b);
    }

    return result;
}

void ms_register_matrix_call_stats_mann_whitney_u() {
    register_matrix_call("stats_mann_whitney_u", &handle_stats_mann_whitney_u);
}

} // namespace ms::interp
