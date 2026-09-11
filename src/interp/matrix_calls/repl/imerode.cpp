// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_imerode(Interpreter& interp, const MatrixCallAssign& assign) {
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
        // The filter visits every pixel once per kernel cell, so the cost is the image
        // times the kernel AREA. Measured at 20 ns per pixel-cell on a 256x256:
        // imerode(ones(512,512), 4999) is 6.5e12 of them, about a day.
        // A kernel wider than the image is also meaningless -- every window is then
        // the whole image -- but that shape bound alone would still leave
        // 512 x 512 x 512^2 to do.
        WorkBudget budget(assign.callee, 20.0, kMaxReplSimulationWorkNanos);
        budget.charge(matrix->rows() * matrix->cols());
        auto kernel = budget.take_square("ksize", static_cast<double>(ksize));
        if (!kernel) {
            return std::unexpected(kernel.error());
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

void ms_register_matrix_call_imerode() {
    register_matrix_call("imerode", &handle_imerode);
}

} // namespace ms::interp
