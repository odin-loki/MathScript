#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_poly_interp_hermite(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "poly_interp_hermite" && assign.args.size() == 3) {
        auto xs = ctx.resolve_operand(assign.args[0]);
        if (!xs) {
            return std::unexpected(xs.error());
        }
        auto ys = ctx.resolve_operand(assign.args[1]);
        if (!ys) {
            return std::unexpected(ys.error());
        }
        auto dys = ctx.resolve_operand(assign.args[2]);
        if (!dys) {
            return std::unexpected(dys.error());
        }
        auto coeffs = eval_poly_interp_hermite(*xs, *ys, *dys);
        if (!coeffs) {
            return std::unexpected(coeffs.error());
        }
        result = *coeffs;
    }

    return result;
}

void ms_register_matrix_call_poly_interp_hermite() {
    register_matrix_call("poly_interp_hermite", &handle_poly_interp_hermite);
}

} // namespace ms::interp
