// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_imtophat(Interpreter& interp, const MatrixCallAssign& assign) {
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
        // The filter visits every pixel once per kernel cell, so the cost is the image
        // times the kernel. Measured at 2 ns per pixel-cell on a 256x256; the audit's
        // probe asks for imtophat(ones(512,512), 4999), which is a day. A kernel wider than the image is also
        // meaningless -- every window is then the whole image -- but that shape bound
        // alone would still leave 512 x 512 x 512^2 to do.
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

void ms_register_matrix_call_imtophat() {
    register_matrix_call("imtophat", &handle_imtophat);
}

} // namespace ms::interp
