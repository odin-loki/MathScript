// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_diag(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "diag" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto v = matrix_to_coeff_vector(*matrix, "diag");
        if (!v) {
            return std::unexpected(v.error());
        }
        if (v->empty()) {
            return std::unexpected(DomainError{"diag", "expected non-empty vector"});
        }
        result = diag(*v);
    }

    return result;
}

void ms_register_matrix_call_diag() {
    register_matrix_call("diag", &handle_diag);
}

} // namespace ms::interp
