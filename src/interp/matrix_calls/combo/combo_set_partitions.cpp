#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_combo_set_partitions(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "combo_set_partitions" && assign.args.size() == 1) {
        double n_d = 0.0;
        if (!parse_number(assign.args[0], n_d)) {
            auto n_expr = eval_scalar_expr(ctx.state(), assign.args[0]);
            if (!n_expr) {
                return std::unexpected(n_expr.error());
            }
            n_d = *n_expr;
        }
        const int n = static_cast<int>(n_d);
        if (n < 0 || n_d != n) {
            return std::unexpected(
                DomainError{"combo_set_partitions", "expected non-negative integer n"});
        }
        auto parts = eval_combo_set_partitions(n);
        if (!parts) {
            return std::unexpected(parts.error());
        }
        result = *parts;
    }

    return result;
}

void ms_register_matrix_call_combo_set_partitions() {
    register_matrix_call("combo_set_partitions", &handle_combo_set_partitions);
}

} // namespace ms::interp
