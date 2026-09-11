// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_shi_tomasi(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "shi_tomasi" &&
               (assign.args.size() == 2 || assign.args.size() == 3)) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        double n_d = 0.0;
        double quality = 0.01;
        if (!parse_number(assign.args[1], n_d)) {
            return std::unexpected(
                DomainError{"shi_tomasi", "expected shi_tomasi(M, n) or shi_tomasi(M, n, quality)"});
        }
        if (assign.args.size() == 3 && !parse_number(assign.args[2], quality)) {
            return std::unexpected(
                DomainError{"shi_tomasi", "expected shi_tomasi(M, n) or shi_tomasi(M, n, quality)"});
        }
        const int n = static_cast<int>(n_d);
        if (n < 1 || n_d != n) {
            return std::unexpected(DomainError{"shi_tomasi", "expected positive integer n"});
        }
        result = eval_shi_tomasi(*matrix, n, static_cast<float>(quality));
    }

    return result;
}

void ms_register_matrix_call_shi_tomasi() {
    register_matrix_call("shi_tomasi", &handle_shi_tomasi);
}

} // namespace ms::interp
