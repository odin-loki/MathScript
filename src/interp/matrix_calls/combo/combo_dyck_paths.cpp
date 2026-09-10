#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_combo_dyck_paths(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "combo_dyck_paths" && assign.args.size() == 1) {
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
                DomainError{"combo_dyck_paths", "expected non-negative integer n"});
        }
        auto paths = eval_combo_dyck_paths(n);
        if (!paths) {
            return std::unexpected(paths.error());
        }
        result = *paths;
    }

    return result;
}

void ms_register_matrix_call_combo_dyck_paths() {
    register_matrix_call("combo_dyck_paths", &handle_combo_dyck_paths);
}

} // namespace ms::interp
