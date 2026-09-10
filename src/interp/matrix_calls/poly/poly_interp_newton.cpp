#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_poly_interp_newton(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "poly_interp_newton" && assign.args.size() == 2) {
        auto xs = ctx.resolve_operand(assign.args[0]);
        if (!xs) {
            return std::unexpected(xs.error());
        }
        auto ys = ctx.resolve_operand(assign.args[1]);
        if (!ys) {
            return std::unexpected(ys.error());
        }
        auto coeffs = eval_poly_interp_newton(*xs, *ys);
        if (!coeffs) {
            return std::unexpected(coeffs.error());
        }
        result = *coeffs;
    }

    return result;
}

void ms_register_matrix_call_poly_interp_newton() {
    register_matrix_call("poly_interp_newton", &handle_poly_interp_newton);
}

} // namespace ms::interp
