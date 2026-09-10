#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_combo_necklaces(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "combo_necklaces" && assign.args.size() == 2) {
        double n_d = 0.0;
        if (!parse_number(assign.args[0], n_d)) {
            auto n_expr = eval_scalar_expr(ctx.state(), assign.args[0]);
            if (!n_expr) {
                return std::unexpected(
                    DomainError{"combo_necklaces", "expected combo_necklaces(n,k)"});
            }
            n_d = *n_expr;
        }
        double k_d = 0.0;
        if (!parse_number(assign.args[1], k_d)) {
            auto k_expr = eval_scalar_expr(ctx.state(), assign.args[1]);
            if (!k_expr) {
                return std::unexpected(
                    DomainError{"combo_necklaces", "expected combo_necklaces(n,k)"});
            }
            k_d = *k_expr;
        }
        const int n = static_cast<int>(n_d);
        const int k = static_cast<int>(k_d);
        if (n < 0 || n_d != n) {
            return std::unexpected(
                DomainError{"combo_necklaces", "expected non-negative integer n"});
        }
        if (k <= 0 || k_d != k) {
            return std::unexpected(
                DomainError{"combo_necklaces", "expected positive integer k"});
        }
        auto necks = eval_combo_necklaces(n, k);
        if (!necks) {
            return std::unexpected(necks.error());
        }
        result = *necks;
    }

    return result;
}

void ms_register_matrix_call_combo_necklaces() {
    register_matrix_call("combo_necklaces", &handle_combo_necklaces);
}

} // namespace ms::interp
