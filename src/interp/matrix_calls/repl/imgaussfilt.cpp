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
    if (assign.callee == "imgaussfilt" && assign.args.size() == 2) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        double sigma = 0.0;
        if (!parse_number(assign.args[1], sigma)) {
            return std::unexpected(DomainError{"imgaussfilt", "expected imgaussfilt(M, sigma)"});
        }
        auto gray = matrix_to_gray_image(*matrix);
        if (!gray) {
            return std::unexpected(gray.error());
        }
        result = gray_image_to_matrix(image::imgaussfilt(*gray, static_cast<float>(sigma)));
    }

    return result;
}

void ms_register_matrix_call_imgaussfilt() {
    register_matrix_call("imgaussfilt", &handle_imgaussfilt);
}

} // namespace ms::interp
