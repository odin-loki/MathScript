// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include <cmath>
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_boxfilter(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "boxfilter" &&
               (assign.args.size() == 1 || assign.args.size() == 2)) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        // Arity 1 is what `is_valid_matrix_call_arity` has always said this callee
        // takes, and the eight neighbours it is grouped with implement it. 3 is the
        // same default they use, and the smallest kernel a box filter can have that
        // is odd and does something.
        int ksize = 3;
        if (assign.args.size() == 2) {
            double ksize_d = 0.0;
            if (!parse_number(assign.args[1], ksize_d)) {
                return std::unexpected(DomainError{"boxfilter", "expected boxfilter(M[, ksize])"});
            }
            if (!std::isfinite(ksize_d) || ksize_d != std::floor(ksize_d) || ksize_d < 1.0 ||
                ksize_d > 2147483647.0 || (static_cast<long long>(ksize_d) % 2) == 0) {
                return std::unexpected(
                    DomainError{"boxfilter", "expected positive odd integer ksize"});
            }
            ksize = static_cast<int>(ksize_d);
        }
        auto gray = matrix_to_gray_image(*matrix);
        if (!gray) {
            return std::unexpected(gray.error());
        }
        result = gray_image_to_matrix(image::boxfilter(*gray, ksize));
    }

    return result;
}

void ms_register_matrix_call_boxfilter() {
    register_matrix_call("boxfilter", &handle_boxfilter);
}

} // namespace ms::interp
