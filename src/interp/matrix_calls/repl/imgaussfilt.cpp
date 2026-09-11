// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_imgaussfilt(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "imgaussfilt" &&
               (assign.args.size() == 1 || assign.args.size() == 2)) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        // `is_valid_matrix_call_arity` lists this callee at arity 1 and this handler
        // took only 2, so `B = imgaussfilt(A)` reported "assign: unsupported matrix
        // call" -- which names neither the callee nor what is missing. The arity is
        // answered here rather than removed from the table, because the table is what
        // decides whether the line is a matrix call at all: drop it and the same input
        // stops being recognised as a call and reports something further still from
        // the truth.
        //
        // No default is supplied. `sigma` is not a tuning knob on a Gaussian blur, it
        // IS the blur -- there is no width that a caller who did not name one meant.
        // Its neighbours in this group default a kernel SIZE, which is a different
        // kind of argument, and `image::imgaussfilt` declares no default either.
        if (assign.args.size() == 1) {
            return std::unexpected(DomainError{
                "imgaussfilt", "sigma has no default: it is the width of the blur, not a "
                               "setting on it. Write imgaussfilt(M, sigma)"});
        }
        double sigma = 0.0;
        if (!parse_number(assign.args[1], sigma)) {
            return std::unexpected(DomainError{"imgaussfilt", "expected imgaussfilt(M, sigma)"});
        }
        auto gray = matrix_to_gray_image(*matrix);
        if (!gray) {
            return std::unexpected(gray.error());
        }
        // `sigma` is not a tuning knob on the cost: the kernel half-width is 3*sigma,
        // so what sigma really names is the kernel width, and the filter is separable, so the cost is the image times the width.
        // Measured at 35 ns per pixel-cell on a 256x256; imgaussfilt(ones(512,512), 100000)
        // is past anything that finishes. The bound is stated on the KERNEL rather than
        // on sigma, because the kernel is the thing that has to fit.
        if (!std::isfinite(sigma) || sigma < 0.0) {
            return std::unexpected(
                DomainError{"imgaussfilt", "expected a finite non-negative sigma"});
        }
        const double kernel_width = 6 * sigma + 1.0;
        WorkBudget budget(assign.callee, 35.0, kMaxReplSimulationWorkNanos);
        budget.charge(matrix->rows() * matrix->cols());
        auto kernel = budget.take("the kernel sigma implies", kernel_width);
        if (!kernel) {
            return std::unexpected(kernel.error());
        }
        result = gray_image_to_matrix(image::imgaussfilt(*gray, static_cast<float>(sigma)));
    }

    return result;
}

void ms_register_matrix_call_imgaussfilt() {
    register_matrix_call("imgaussfilt", &handle_imgaussfilt);
}

} // namespace ms::interp
