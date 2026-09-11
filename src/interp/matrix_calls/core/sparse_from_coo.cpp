// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_sparse_from_coo(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "sparse_from_coo" && assign.args.size() == 5) {
        auto rows_val = ctx.parse_scalar_arg(assign.args[0], "sparse_from_coo");
        if (!rows_val) {
            return std::unexpected(rows_val.error());
        }
        auto cols_val = ctx.parse_scalar_arg(assign.args[1], "sparse_from_coo");
        if (!cols_val) {
            return std::unexpected(cols_val.error());
        }
        const int rows_i = static_cast<int>(*rows_val);
        const int cols_i = static_cast<int>(*cols_val);
        if (rows_i < 0 || cols_i < 0 || *rows_val != rows_i || *cols_val != cols_i) {
            return std::unexpected(
                DomainError{"sparse_from_coo", "expected non-negative integer rows and cols"});
        }
        auto row_idx = ctx.resolve_operand(assign.args[2]);
        if (!row_idx) {
            return std::unexpected(row_idx.error());
        }
        auto col_idx = ctx.resolve_operand(assign.args[3]);
        if (!col_idx) {
            return std::unexpected(col_idx.error());
        }
        auto values = ctx.resolve_operand(assign.args[4]);
        if (!values) {
            return std::unexpected(values.error());
        }
        result = eval_sparse_from_coo(static_cast<std::size_t>(rows_i),
                                      static_cast<std::size_t>(cols_i), *row_idx, *col_idx,
                                      *values);
    }

    return result;
}

void ms_register_matrix_call_sparse_from_coo() {
    register_matrix_call("sparse_from_coo", &handle_sparse_from_coo);
}

} // namespace ms::interp
