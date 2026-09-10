#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_gria_divergence_trajectory(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "gria_divergence_trajectory" && assign.args.size() == 4) {
        auto a = ctx.resolve_operand(assign.args[0]);
        if (!a) {
            return std::unexpected(a.error());
        }
        auto b = ctx.resolve_operand(assign.args[1]);
        if (!b) {
            return std::unexpected(b.error());
        }
        auto rule_val = ctx.parse_scalar_arg(assign.args[2], "gria_divergence_trajectory");
        if (!rule_val) {
            return std::unexpected(rule_val.error());
        }
        auto n_steps_val = ctx.parse_scalar_arg(assign.args[3], "gria_divergence_trajectory");
        if (!n_steps_val) {
            return std::unexpected(n_steps_val.error());
        }
        const int rule = static_cast<int>(*rule_val);
        if (*rule_val != rule || rule < 0 || rule > 255) {
            return std::unexpected(
                DomainError{"gria_divergence_trajectory", "expected integer rule in [0,255]"});
        }
        const int n_steps = static_cast<int>(*n_steps_val);
        if (*n_steps_val != n_steps) {
            return std::unexpected(DomainError{
                "gria_divergence_trajectory", "expected integer n_steps"});
        }
        result = eval_gria_divergence_trajectory(*a, *b, rule, n_steps);
    }

    return result;
}

void ms_register_matrix_call_gria_divergence_trajectory() {
    register_matrix_call("gria_divergence_trajectory", &handle_gria_divergence_trajectory);
}

} // namespace ms::interp
