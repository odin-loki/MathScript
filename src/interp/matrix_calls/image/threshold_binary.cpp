// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_threshold_binary(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "threshold_binary" && assign.args.size() == 2) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        double t = 0.0;
        if (!parse_number(assign.args[1], t)) {
            return std::unexpected(
                DomainError{"threshold_binary", "expected threshold_binary(M, t)"});
        }
        auto gray = matrix_to_gray_image(*matrix);
        if (!gray) {
            return std::unexpected(gray.error());
        }
        result = gray_image_to_matrix(image::threshold_binary(*gray, static_cast<float>(t)));
    }

    return result;
}

void ms_register_matrix_call_threshold_binary() {
    register_matrix_call("threshold_binary", &handle_threshold_binary);
}

} // namespace ms::interp
