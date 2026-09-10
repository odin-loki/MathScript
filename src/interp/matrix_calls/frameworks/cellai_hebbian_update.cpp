#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_cellai_hebbian_update(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "cellai_hebbian_update" && assign.args.size() == 4) {
        auto w = ctx.resolve_operand(assign.args[0]);
        if (!w) {
            return std::unexpected(w.error());
        }
        auto x = ctx.resolve_operand(assign.args[1]);
        if (!x) {
            return std::unexpected(x.error());
        }
        auto y = ctx.resolve_operand(assign.args[2]);
        if (!y) {
            return std::unexpected(y.error());
        }
        auto lr = ctx.parse_scalar_arg(assign.args[3], "cellai_hebbian_update");
        if (!lr) {
            return std::unexpected(lr.error());
        }
        result = eval_cellai_hebbian_update(*w, *x, *y, *lr);
    }

    return result;
}

void ms_register_matrix_call_cellai_hebbian_update() {
    register_matrix_call("cellai_hebbian_update", &handle_cellai_hebbian_update);
}

} // namespace ms::interp
