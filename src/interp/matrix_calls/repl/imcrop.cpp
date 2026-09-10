#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_imcrop(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "imcrop" && assign.args.size() == 5) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        double r0_d = 0.0;
        double c0_d = 0.0;
        double r1_d = 0.0;
        double c1_d = 0.0;
        if (!parse_number(assign.args[1], r0_d) || !parse_number(assign.args[2], c0_d) ||
            !parse_number(assign.args[3], r1_d) || !parse_number(assign.args[4], c1_d)) {
            return std::unexpected(
                DomainError{"imcrop", "expected imcrop(M, r0, c0, r1, c1)"});
        }
        const int r0 = static_cast<int>(r0_d);
        const int c0 = static_cast<int>(c0_d);
        const int r1 = static_cast<int>(r1_d);
        const int c1 = static_cast<int>(c1_d);
        if (r0_d != r0 || c0_d != c0 || r1_d != r1 || c1_d != c1) {
            return std::unexpected(DomainError{"imcrop", "expected integer crop bounds"});
        }
        auto cropped = eval_imcrop(*matrix, r0, c0, r1, c1);
        if (!cropped) {
            return std::unexpected(cropped.error());
        }
        result = *cropped;
    }

    return result;
}

void ms_register_matrix_call_imcrop() {
    register_matrix_call("imcrop", &handle_imcrop);
}

} // namespace ms::interp
