#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_poly_lagrange(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "poly_lagrange" && assign.args.size() == 2) {
        auto xs = ctx.resolve_operand(assign.args[0]);
        if (!xs) {
            return std::unexpected(xs.error());
        }
        auto ys = ctx.resolve_operand(assign.args[1]);
        if (!ys) {
            return std::unexpected(ys.error());
        }
        auto coeffs = eval_poly_lagrange(*xs, *ys);
        if (!coeffs) {
            return std::unexpected(coeffs.error());
        }
        result = *coeffs;
    }

    return result;
}

void ms_register_matrix_call_poly_lagrange() {
    register_matrix_call("poly_lagrange", &handle_poly_lagrange);
}

} // namespace ms::interp
