#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_poly_mod(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "poly_mod" && assign.args.size() == 2) {
        auto a = ctx.resolve_operand(assign.args[0]);
        if (!a) {
            return std::unexpected(a.error());
        }
        auto b = ctx.resolve_operand(assign.args[1]);
        if (!b) {
            return std::unexpected(b.error());
        }
        auto rem = eval_poly_mod(*a, *b);
        if (!rem) {
            return std::unexpected(rem.error());
        }
        result = *rem;
    }

    return result;
}

void ms_register_matrix_call_poly_mod() {
    register_matrix_call("poly_mod", &handle_poly_mod);
}

} // namespace ms::interp
