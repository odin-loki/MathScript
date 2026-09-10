#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_poly_eval_at(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "poly_eval_at" && assign.args.size() == 2) {
        auto coeffs = ctx.resolve_operand(assign.args[0]);
        if (!coeffs) {
            return std::unexpected(coeffs.error());
        }
        auto xs = ctx.resolve_operand(assign.args[1]);
        if (!xs) {
            return std::unexpected(xs.error());
        }
        auto values = eval_poly_eval_at(*coeffs, *xs);
        if (!values) {
            return std::unexpected(values.error());
        }
        result = *values;
    }

    return result;
}

void ms_register_matrix_call_poly_eval_at() {
    register_matrix_call("poly_eval_at", &handle_poly_eval_at);
}

} // namespace ms::interp
