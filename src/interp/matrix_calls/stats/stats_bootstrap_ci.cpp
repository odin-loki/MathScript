#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_stats_bootstrap_ci(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "stats_bootstrap_ci" && assign.args.size() == 1) {
        auto x = ctx.resolve_operand(assign.args[0]);
        if (!x) {
            return std::unexpected(x.error());
        }
        result = eval_stats_bootstrap_ci(*x);
    }

    return result;
}

void ms_register_matrix_call_stats_bootstrap_ci() {
    register_matrix_call("stats_bootstrap_ci", &handle_stats_bootstrap_ci);
}

} // namespace ms::interp
