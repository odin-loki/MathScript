// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include <cmath>
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_medfilt2(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "medfilt2" &&
               (assign.args.size() == 1 || assign.args.size() == 2)) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        // `is_valid_matrix_call_arity` has always listed medfilt2 at arity 1, alongside
        // the eight neighbours in its group that implement it; this handler did not, so
        // `B = medfilt2(A)` came back "assign: unsupported matrix call", which names
        // neither the callee nor what was missing. The default is not invented here --
        // `image::medfilt2` declares `int ksize = 3` in its own signature.
        int ksize = 3;
        if (assign.args.size() == 2) {
            double ksize_d = 0.0;
            if (!parse_number(assign.args[1], ksize_d)) {
                return std::unexpected(DomainError{"medfilt2", "expected medfilt2(M[, ksize])"});
            }
            // The range is settled on the double: `static_cast<int>` of one outside int's
            // range is undefined behaviour rather than a wrap.
            if (!std::isfinite(ksize_d) || ksize_d != std::floor(ksize_d) || ksize_d < 1.0 ||
                ksize_d > 2147483647.0 || (static_cast<long long>(ksize_d) % 2) == 0) {
                return std::unexpected(
                    DomainError{"medfilt2", "expected positive odd integer ksize"});
            }
            ksize = static_cast<int>(ksize_d);
        }
        auto gray = matrix_to_gray_image(*matrix);
        if (!gray) {
            return std::unexpected(gray.error());
        }
        result = gray_image_to_matrix(image::medfilt2(*gray, ksize));
    }

    return result;
}

void ms_register_matrix_call_medfilt2() {
    register_matrix_call("medfilt2", &handle_medfilt2);
}

} // namespace ms::interp
