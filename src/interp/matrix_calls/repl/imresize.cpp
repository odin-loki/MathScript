// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_imresize(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "imresize" && assign.args.size() == 3) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        double rows_d = 0.0;
        double cols_d = 0.0;
        if (!parse_number(assign.args[1], rows_d) || !parse_number(assign.args[2], cols_d)) {
            return std::unexpected(DomainError{"imresize", "expected imresize(M, rows, cols)"});
        }
        // Both extents at once, so that the cap is on the product: 100000 rows is not a
        // large number and 100000 columns is not either, and together they are ten
        // billion elements. Neither was checked for integrality or for range before --
        // `static_cast<int>` of a double outside int's range is undefined behaviour, not
        // a wrap, so the check has to happen on the double.
        ExtentBudget budget("imresize");
        auto rows = budget.take("rows", rows_d);
        if (!rows) {
            return std::unexpected(rows.error());
        }
        auto cols = budget.take("cols", cols_d);
        if (!cols) {
            return std::unexpected(cols.error());
        }
        auto gray = matrix_to_gray_image(*matrix);
        if (!gray) {
            return std::unexpected(gray.error());
        }
        result = gray_image_to_matrix(
            image::imresize(*gray, static_cast<int>(*rows), static_cast<int>(*cols)));
    }

    return result;
}

void ms_register_matrix_call_imresize() {
    register_matrix_call("imresize", &handle_imresize);
}

} // namespace ms::interp
