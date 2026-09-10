#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_kron(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "kron" && assign.args.size() == 2) {
        auto left = ctx.resolve_operand(assign.args[0]);
        if (!left) {
            return std::unexpected(left.error());
        }
        auto right = ctx.resolve_operand(assign.args[1]);
        if (!right) {
            return std::unexpected(right.error());
        }
        const size_t out_r = left->rows();
        const size_t out_c = left->cols();
        const size_t rr = right->rows();
        const size_t rc = right->cols();
        if ((rr != 0 && out_r > kMaxReplMatrixElems / rr) ||
            (rc != 0 && out_c > kMaxReplMatrixElems / rc) ||
            !repl_elems_allowed(out_r * rr, out_c * rc)) {
            return std::unexpected(DomainError{"kron", kReplMatrixTooLarge});
        }
        result = kron(*left, *right);
    }

    return result;
}

void ms_register_matrix_call_kron() {
    register_matrix_call("kron", &handle_kron);
}

} // namespace ms::interp
