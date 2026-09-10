#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_stats_kde(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "stats_kde" &&
               (assign.args.size() == 3 || assign.args.size() == 4)) {
        auto samples = ctx.resolve_operand(assign.args[0]);
        if (!samples) {
            return std::unexpected(samples.error());
        }
        auto grid = ctx.resolve_operand(assign.args[1]);
        if (!grid) {
            return std::unexpected(grid.error());
        }
        double h = 0.0;
        if (!parse_number(assign.args[2], h)) {
            return std::unexpected(
                DomainError{"stats_kde", "expected stats_kde(samples, grid, h[, kernel])"});
        }
        std::string kernel = "gaussian";
        if (assign.args.size() == 4) {
            if (!parse_quoted_string(trim_copy(assign.args[3]), kernel)) {
                kernel = trim_copy(assign.args[3]);
            }
            if (kernel.empty()) {
                return std::unexpected(
                    DomainError{"stats_kde", "expected stats_kde(samples, grid, h[, kernel])"});
            }
        }
        result = eval_stats_kde(*samples, *grid, h, kernel.c_str());
    }

    return result;
}

void ms_register_matrix_call_stats_kde() {
    register_matrix_call("stats_kde", &handle_stats_kde);
}

} // namespace ms::interp
