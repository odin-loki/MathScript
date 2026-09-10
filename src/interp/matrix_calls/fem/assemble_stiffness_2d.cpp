#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_assemble_stiffness_2d(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if ((assign.callee == "fem_stiffness_2d" || assign.callee == "assemble_stiffness_2d") &&
               assign.args.size() == 1) {
        auto mesh = ctx.resolve_operand(assign.args[0]);
        if (!mesh) {
            return std::unexpected(mesh.error());
        }
        result = eval_fem_stiffness_2d(*mesh);
    }

    return result;
}

void ms_register_matrix_call_assemble_stiffness_2d() {
    register_matrix_call("assemble_stiffness_2d", &handle_assemble_stiffness_2d);
}

} // namespace ms::interp
