// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_laplacian_of_gaussian(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "laplacian_of_gaussian" &&
               (assign.args.size() == 1 || assign.args.size() == 2)) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        // Same as imgaussfilt, and for the same reason: a Laplacian of Gaussian is
        // built by blurring at `sigma` and then differencing, so the scale is the
        // operation. See that handler for why the arity is answered rather than
        // removed from the table.
        if (assign.args.size() == 1) {
            return std::unexpected(DomainError{
                "laplacian_of_gaussian",
                "sigma has no default: it is the scale the operator detects at, not a "
                "setting on it. Write laplacian_of_gaussian(M, sigma)"});
        }
        auto gray = matrix_to_gray_image(*matrix);
        if (!gray) {
            return std::unexpected(gray.error());
        }
        auto sigma = ctx.parse_scalar_arg(assign.args[1], "laplacian_of_gaussian");
        if (!sigma) {
            return std::unexpected(sigma.error());
        }
        // `sigma` is not a tuning knob on the cost: the kernel half-width is 3*sigma,
        // so what sigma really names is the kernel width, and the filter is separable, so the cost is the image times the width.
        // Measured at 35 ns per pixel-cell on a 256x256; laplacian_of_gaussian(ones(512,512), 100000)
        // is past anything that finishes. The bound is stated on the KERNEL rather than
        // on sigma, because the kernel is the thing that has to fit.
        if (!std::isfinite(*sigma) || *sigma < 0.0) {
            return std::unexpected(
                DomainError{"laplacian_of_gaussian", "expected a finite non-negative sigma"});
        }
        const double kernel_width = 6 * *sigma + 1.0;
        WorkBudget budget(assign.callee, 35.0, kMaxReplSimulationWorkNanos);
        budget.charge(matrix->rows() * matrix->cols());
        auto kernel = budget.take("the kernel sigma implies", kernel_width);
        if (!kernel) {
            return std::unexpected(kernel.error());
        }
        result = gray_image_to_matrix(
            image::laplacian_of_gaussian(*gray, static_cast<float>(*sigma)));
    }

    return result;
}

void ms_register_matrix_call_laplacian_of_gaussian() {
    register_matrix_call("laplacian_of_gaussian", &handle_laplacian_of_gaussian);
}

} // namespace ms::interp
