#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_stats_wilcoxon_signed_rank(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "stats_wilcoxon_signed_rank" && assign.args.size() == 2) {
        auto x = ctx.resolve_operand(assign.args[0]);
        if (!x) {
            return std::unexpected(x.error());
        }
        auto y = ctx.resolve_operand(assign.args[1]);
        if (!y) {
            return std::unexpected(y.error());
        }
        result = eval_stats_wilcoxon_signed_rank(*x, *y);
    }

    return result;
}

void ms_register_matrix_call_stats_wilcoxon_signed_rank() {
    register_matrix_call("stats_wilcoxon_signed_rank", &handle_stats_wilcoxon_signed_rank);
}

} // namespace ms::interp
