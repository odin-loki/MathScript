// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_imfilter(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "imfilter" && assign.args.size() == 2) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto kernel_m = ctx.resolve_operand(assign.args[1]);
        if (!kernel_m) {
            return std::unexpected(kernel_m.error());
        }
        auto gray = matrix_to_gray_image(*matrix);
        if (!gray) {
            return std::unexpected(gray.error());
        }
        auto kernel = matrix_to_filter_kernel(*kernel_m, "imfilter");
        if (!kernel) {
            return std::unexpected(kernel.error());
        }
        result = gray_image_to_matrix(image::imfilter(*gray, *kernel));
    }

    return result;
}

void ms_register_matrix_call_imfilter() {
    register_matrix_call("imfilter", &handle_imfilter);
}

} // namespace ms::interp
