#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_gria_ca_step(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "gria_ca_step" && assign.args.size() == 2) {
        auto state = ctx.resolve_operand(assign.args[0]);
        if (!state) {
            return std::unexpected(state.error());
        }
        auto rule_val = ctx.parse_scalar_arg(assign.args[1], "gria_ca_step");
        if (!rule_val) {
            return std::unexpected(rule_val.error());
        }
        const int rule = static_cast<int>(*rule_val);
        if (*rule_val != rule || rule < 0 || rule > 255) {
            return std::unexpected(
                DomainError{"gria_ca_step", "expected integer rule in [0,255]"});
        }
        result = eval_gria_ca_step(*state, rule);
    }

    return result;
}

void ms_register_matrix_call_gria_ca_step() {
    register_matrix_call("gria_ca_step", &handle_gria_ca_step);
}

} // namespace ms::interp
