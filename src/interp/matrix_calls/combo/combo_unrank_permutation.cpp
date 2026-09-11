// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_combo_unrank_permutation(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "combo_unrank_permutation" && assign.args.size() == 2) {
        double n_d = 0.0;
        double rank_d = 0.0;
        if (!parse_number(assign.args[0], n_d)) {
            auto it = ctx.state().scalars.find(assign.args[0]);
            if (it != ctx.state().scalars.end()) {
                n_d = it->second;
            } else {
                return std::unexpected(
                    DomainError{assign.callee, "expected numeric n argument"});
            }
        }
        if (!parse_number(assign.args[1], rank_d)) {
            auto it = ctx.state().scalars.find(assign.args[1]);
            if (it != ctx.state().scalars.end()) {
                rank_d = it->second;
            } else {
                return std::unexpected(
                    DomainError{assign.callee, "expected numeric rank argument"});
            }
        }
        const int n = static_cast<int>(n_d);
        if (n < 0 || n_d != n || rank_d < 0.0 || std::floor(rank_d) != rank_d) {
            return std::unexpected(DomainError{
                assign.callee, "expected non-negative integer n and rank"});
        }
        auto perm = eval_combo_unrank_permutation(n, static_cast<uint64_t>(rank_d));
        if (!perm) {
            return std::unexpected(perm.error());
        }
        result = *perm;
    }

    return result;
}

void ms_register_matrix_call_combo_unrank_permutation() {
    register_matrix_call("combo_unrank_permutation", &handle_combo_unrank_permutation);
}

} // namespace ms::interp
