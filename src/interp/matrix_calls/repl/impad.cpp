// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_impad(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "impad" &&
               (assign.args.size() == 2 || assign.args.size() == 3)) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        double pad_d = 0.0;
        if (!parse_number(assign.args[1], pad_d)) {
            return std::unexpected(DomainError{"impad", "expected impad(M, pad[, val])"});
        }
        ExtentBudget budget("impad");
        auto pad_extent = budget.take("pad", pad_d);
        if (!pad_extent) {
            return std::unexpected(pad_extent.error());
        }
        const std::size_t pad = *pad_extent;
        // A bound on `pad` alone is not a bound on what gets allocated. The result is
        // the source grown by `pad` on all FOUR sides, so the element count goes as the
        // square: `impad(A, 1000000)` is a padding well inside any per-extent cap and
        // four trillion elements of result, and the allocation ended the process.
        if (!repl_elems_allowed(matrix->rows() + 2 * pad, matrix->cols() + 2 * pad)) {
            return std::unexpected(DomainError{"impad", kReplMatrixTooLarge});
        }
        double val_d = 0.0;
        if (assign.args.size() == 3 && !parse_number(assign.args[2], val_d)) {
            return std::unexpected(DomainError{"impad", "expected numeric pad value"});
        }
        auto gray = matrix_to_gray_image(*matrix);
        if (!gray) {
            return std::unexpected(gray.error());
        }
        result = gray_image_to_matrix(
            image::impad(*gray, static_cast<int>(pad), static_cast<float>(val_d)));
    }

    return result;
}

void ms_register_matrix_call_impad() {
    register_matrix_call("impad", &handle_impad);
}

} // namespace ms::interp
