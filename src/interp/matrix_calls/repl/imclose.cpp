// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_imclose(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if ((assign.callee == "imdilate" || assign.callee == "imerode" ||
                assign.callee == "imopen" || assign.callee == "imclose") &&
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
        if (assign.callee == "imdilate") {
            result = gray_image_to_matrix(image::imdilate(*gray, ksize));
        } else if (assign.callee == "imerode") {
            result = gray_image_to_matrix(image::imerode(*gray, ksize));
        } else if (assign.callee == "imopen") {
            result = gray_image_to_matrix(image::imopen(*gray, ksize));
        } else {
            result = gray_image_to_matrix(image::imclose(*gray, ksize));
        }
    }

    return result;
}

void ms_register_matrix_call_imclose() {
    register_matrix_call("imclose", &handle_imclose);
}

} // namespace ms::interp
