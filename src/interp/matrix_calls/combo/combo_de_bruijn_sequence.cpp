#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_combo_de_bruijn_sequence(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "combo_de_bruijn_sequence" && assign.args.size() == 2) {
        double k_d = 0.0;
        if (!parse_number(assign.args[0], k_d)) {
            auto k_expr = eval_scalar_expr(ctx.state(), assign.args[0]);
            if (!k_expr) {
                return std::unexpected(DomainError{
                    "combo_de_bruijn_sequence", "expected combo_de_bruijn_sequence(k,n)"});
            }
            k_d = *k_expr;
        }
        double n_d = 0.0;
        if (!parse_number(assign.args[1], n_d)) {
            auto n_expr = eval_scalar_expr(ctx.state(), assign.args[1]);
            if (!n_expr) {
                return std::unexpected(DomainError{
                    "combo_de_bruijn_sequence", "expected combo_de_bruijn_sequence(k,n)"});
            }
            n_d = *n_expr;
        }
        const int k = static_cast<int>(k_d);
        const int n = static_cast<int>(n_d);
        if (k <= 0 || k_d != k) {
            return std::unexpected(DomainError{
                "combo_de_bruijn_sequence", "expected positive integer k"});
        }
        if (n <= 0 || n_d != n) {
            return std::unexpected(DomainError{
                "combo_de_bruijn_sequence", "expected positive integer n"});
        }
        auto seq = eval_combo_de_bruijn_sequence(k, n);
        if (!seq) {
            return std::unexpected(seq.error());
        }
        result = *seq;
    }

    return result;
}

void ms_register_matrix_call_combo_de_bruijn_sequence() {
    register_matrix_call("combo_de_bruijn_sequence", &handle_combo_de_bruijn_sequence);
}

} // namespace ms::interp
