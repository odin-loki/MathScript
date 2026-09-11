// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_rgb2gray(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "rgb2gray" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto rgb = matrix_to_rgb_image(*matrix);
        if (!rgb) {
            return std::unexpected(rgb.error());
        }
        result = gray_image_to_column(image::rgb2gray(*rgb));
    }

    return result;
}

void ms_register_matrix_call_rgb2gray() {
    register_matrix_call("rgb2gray", &handle_rgb2gray);
}

} // namespace ms::interp
