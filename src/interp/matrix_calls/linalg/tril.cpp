// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_tril(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "tril" &&
               (assign.args.size() == 1 || assign.args.size() == 2)) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        int k = 0;
        if (assign.args.size() == 2) {
            double k_d = 0.0;
            if (!parse_number(assign.args[1], k_d)) {
                auto it = ctx.state().scalars.find(assign.args[1]);
                if (it != ctx.state().scalars.end()) {
                    k_d = it->second;
                } else {
                    return std::unexpected(
                        DomainError{"tril", "expected tril(A) or tril(A, k)"});
                }
            }
            if (k_d != static_cast<int>(k_d)) {
                return std::unexpected(DomainError{"tril", "expected integer k"});
            }
            k = static_cast<int>(k_d);
        }
        result = tril(*matrix, k);
    }

    return result;
}

void ms_register_matrix_call_tril() {
    register_matrix_call("tril", &handle_tril);
}

} // namespace ms::interp
