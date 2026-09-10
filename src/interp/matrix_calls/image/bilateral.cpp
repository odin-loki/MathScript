// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_bilateral(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "bilateral" && assign.args.size() == 3) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        double sigma_s = 0.0;
        double sigma_r = 0.0;
        if (!parse_number(assign.args[1], sigma_s) || !parse_number(assign.args[2], sigma_r)) {
            return std::unexpected(
                DomainError{"bilateral", "expected bilateral(M, sigma_s, sigma_r)"});
        }
        auto gray = matrix_to_gray_image(*matrix);
        if (!gray) {
            return std::unexpected(gray.error());
        }
        result = gray_image_to_matrix(
            image::bilateral(*gray, static_cast<float>(sigma_s), static_cast<float>(sigma_r)));
    }

    return result;
}

void ms_register_matrix_call_bilateral() {
    register_matrix_call("bilateral", &handle_bilateral);
}

} // namespace ms::interp
