// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_repmat(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "repmat" && assign.args.size() == 3) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        double p_d = 0.0, q_d = 0.0;
        if (!parse_number(assign.args[1], p_d)) {
            return std::unexpected(DomainError{"repmat", "expected numeric row repeat count"});
        }
        if (!parse_number(assign.args[2], q_d)) {
            return std::unexpected(DomainError{"repmat", "expected numeric col repeat count"});
        }
        size_t p = 0;
        size_t q = 0;
        if (!repl_dims_allowed(p_d, q_d, p, q)) {
            return std::unexpected(DomainError{"repmat", kReplMatrixTooLarge});
        }
        const size_t in_r = matrix->rows();
        const size_t in_c = matrix->cols();
        if ((p != 0 && in_r > kMaxReplMatrixElems / p) ||
            (q != 0 && in_c > kMaxReplMatrixElems / q) ||
            !repl_elems_allowed(in_r * p, in_c * q)) {
            return std::unexpected(DomainError{"repmat", kReplMatrixTooLarge});
        }
        result = repmat(*matrix, p, q);
    }

    return result;
}

void ms_register_matrix_call_repmat() {
    register_matrix_call("repmat", &handle_repmat);
}

} // namespace ms::interp
