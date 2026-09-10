// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_rand(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if ((assign.callee == "rand" || assign.callee == "randn") && assign.args.size() == 2) {
        double m_d = 0.0, n_d = 0.0;
        if (!parse_number(assign.args[0], m_d)) {
            auto it = ctx.state().scalars.find(assign.args[0]);
            if (it != ctx.state().scalars.end()) m_d = it->second;
            else return std::unexpected(DomainError{assign.callee, "expected numeric size"});
        }
        if (!parse_number(assign.args[1], n_d)) {
            auto it = ctx.state().scalars.find(assign.args[1]);
            if (it != ctx.state().scalars.end()) n_d = it->second;
            else return std::unexpected(DomainError{assign.callee, "expected numeric size"});
        }
        size_t rows = 0;
        size_t cols = 0;
        if (!repl_dims_allowed(m_d, n_d, rows, cols)) {
            return std::unexpected(
                DomainError{assign.callee, kReplMatrixTooLarge});
        }
        if (assign.callee == "rand") {
            auto R = rand<double>(rows, cols, 0u);
            Matrix<double> stored(rows, cols);
            for (size_t i = 0; i < rows; ++i)
                for (size_t j = 0; j < cols; ++j)
                    stored(i, j) = R(i, j);
            result = stored;
        } else {
            auto R = randn<double>(rows, cols, 0u);
            Matrix<double> stored(rows, cols);
            for (size_t i = 0; i < rows; ++i)
                for (size_t j = 0; j < cols; ++j)
                    stored(i, j) = R(i, j);
            result = stored;
        }
    }

    return result;
}

void ms_register_matrix_call_rand() {
    register_matrix_call("rand", &handle_rand);
}

} // namespace ms::interp
