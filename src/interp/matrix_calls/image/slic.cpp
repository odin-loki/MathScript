#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_slic(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "slic" &&
               (assign.args.size() == 2 || assign.args.size() == 3)) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        double k_d = 0.0;
        if (!parse_number(assign.args[1], k_d)) {
            return std::unexpected(DomainError{"slic", "expected slic(M, K[, compactness])"});
        }
        double compactness = 10.0;
        if (assign.args.size() == 3) {
            if (!parse_number(assign.args[2], compactness)) {
                return std::unexpected(
                    DomainError{"slic", "expected slic(M, K[, compactness])"});
            }
        }
        auto rgb = matrix_to_gray_image(*matrix);
        if (!rgb) {
            return std::unexpected(rgb.error());
        }
        result = gray_image_to_matrix(
            image::slic(*rgb, static_cast<int>(k_d), compactness));
    }

    return result;
}

void ms_register_matrix_call_slic() {
    register_matrix_call("slic", &handle_slic);
}

} // namespace ms::interp
