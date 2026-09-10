#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_poly_lcm(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "poly_lcm" && assign.args.size() == 2) {
        auto a = ctx.resolve_operand(assign.args[0]);
        if (!a) {
            return std::unexpected(a.error());
        }
        auto b = ctx.resolve_operand(assign.args[1]);
        if (!b) {
            return std::unexpected(b.error());
        }
        auto l = eval_poly_lcm(*a, *b);
        if (!l) {
            return std::unexpected(l.error());
        }
        result = *l;
    }

    return result;
}

void ms_register_matrix_call_poly_lcm() {
    register_matrix_call("poly_lcm", &handle_poly_lcm);
}

} // namespace ms::interp
