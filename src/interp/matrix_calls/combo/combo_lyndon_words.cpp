// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_combo_lyndon_words(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "combo_lyndon_words" && assign.args.size() == 2) {
        double n_d = 0.0;
        if (!parse_number(assign.args[0], n_d)) {
            auto n_expr = eval_scalar_expr(ctx.state(), assign.args[0]);
            if (!n_expr) {
                return std::unexpected(
                    DomainError{"combo_lyndon_words", "expected combo_lyndon_words(n,k)"});
            }
            n_d = *n_expr;
        }
        double k_d = 0.0;
        if (!parse_number(assign.args[1], k_d)) {
            auto k_expr = eval_scalar_expr(ctx.state(), assign.args[1]);
            if (!k_expr) {
                return std::unexpected(
                    DomainError{"combo_lyndon_words", "expected combo_lyndon_words(n,k)"});
            }
            k_d = *k_expr;
        }
        const int n = static_cast<int>(n_d);
        const int k = static_cast<int>(k_d);
        if (n < 0 || n_d != n) {
            return std::unexpected(
                DomainError{"combo_lyndon_words", "expected non-negative integer n"});
        }
        if (k <= 0 || k_d != k) {
            return std::unexpected(
                DomainError{"combo_lyndon_words", "expected positive integer k"});
        }
        auto words = eval_combo_lyndon_words(n, k);
        if (!words) {
            return std::unexpected(words.error());
        }
        result = *words;
    }

    return result;
}

void ms_register_matrix_call_combo_lyndon_words() {
    register_matrix_call("combo_lyndon_words", &handle_combo_lyndon_words);
}

} // namespace ms::interp
