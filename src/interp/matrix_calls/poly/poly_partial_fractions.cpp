#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_poly_partial_fractions(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "poly_partial_fractions" && assign.args.size() == 2) {
        auto num = ctx.resolve_operand(assign.args[0]);
        if (!num) {
            return std::unexpected(num.error());
        }
        auto den = ctx.resolve_operand(assign.args[1]);
        if (!den) {
            return std::unexpected(den.error());
        }
        auto pf = eval_poly_partial_fractions(*num, *den);
        if (!pf) {
            return std::unexpected(pf.error());
        }
        result = *pf;
    }

    return result;
}

void ms_register_matrix_call_poly_partial_fractions() {
    register_matrix_call("poly_partial_fractions", &handle_poly_partial_fractions);
}

} // namespace ms::interp
