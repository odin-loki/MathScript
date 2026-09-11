// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_izaac_randn_matrix(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "izaac_randn_matrix" && assign.args.size() == 2) {
        auto rows_val = ctx.parse_scalar_arg(assign.args[0], "izaac_randn_matrix");
        if (!rows_val) {
            return std::unexpected(rows_val.error());
        }
        auto cols_val = ctx.parse_scalar_arg(assign.args[1], "izaac_randn_matrix");
        if (!cols_val) {
            return std::unexpected(cols_val.error());
        }
        auto rows_i = ctx.parse_positive_size_arg(*rows_val, "izaac_randn_matrix", "expected positive integer rows");
        if (!rows_i) {
            return std::unexpected(rows_i.error());
        }
        auto cols_i = ctx.parse_positive_size_arg(*cols_val, "izaac_randn_matrix", "expected positive integer cols");
        if (!cols_i) {
            return std::unexpected(cols_i.error());
        }
        result = eval_izaac_randn_matrix(*rows_i, *cols_i);
    }

    return result;
}

void ms_register_matrix_call_izaac_randn_matrix() {
    register_matrix_call("izaac_randn_matrix", &handle_izaac_randn_matrix);
}

} // namespace ms::interp
