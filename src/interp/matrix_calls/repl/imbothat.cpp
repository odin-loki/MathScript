#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_imbothat(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if ((assign.callee == "imtophat" || assign.callee == "imbothat" ||
                assign.callee == "imgradient_morph") &&
               (assign.args.size() == 1 || assign.args.size() == 2)) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        int ksize = 3;
        if (assign.args.size() == 2) {
            double ksize_d = 0.0;
            if (!parse_number(assign.args[1], ksize_d)) {
                return std::unexpected(
                    DomainError{assign.callee, "expected morphology(M, ksize)"});
            }
            auto parsed = parse_morph_ksize(ksize_d, assign.callee.c_str());
            if (!parsed) {
                return std::unexpected(parsed.error());
            }
            ksize = *parsed;
        }
        auto gray = matrix_to_gray_image(*matrix);
        if (!gray) {
            return std::unexpected(gray.error());
        }
        if (assign.callee == "imtophat") {
            result = gray_image_to_matrix(image::imtophat(*gray, ksize));
        } else if (assign.callee == "imbothat") {
            result = gray_image_to_matrix(image::imbothat(*gray, ksize));
        } else {
            result = gray_image_to_matrix(image::imgradient_morph(*gray, ksize));
        }
    }

    return result;
}

void ms_register_matrix_call_imbothat() {
    register_matrix_call("imbothat", &handle_imbothat);
}

} // namespace ms::interp
