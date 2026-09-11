// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_imflip(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "imflip" && assign.args.size() == 2) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        double horizontal = 0.0;
        if (!parse_number(assign.args[1], horizontal)) {
            return std::unexpected(DomainError{"imflip", "expected imflip(M, horizontal)"});
        }
        auto gray = matrix_to_gray_image(*matrix);
        if (!gray) {
            return std::unexpected(gray.error());
        }
        result = gray_image_to_matrix(image::imflip(*gray, horizontal != 0.0));
    }

    return result;
}

void ms_register_matrix_call_imflip() {
    register_matrix_call("imflip", &handle_imflip);
}

} // namespace ms::interp
