// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_fem_solve(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "fem_solve" && assign.args.size() == 2) {
        auto K = ctx.resolve_operand(assign.args[0]);
        if (!K) {
            return std::unexpected(K.error());
        }
        auto f = ctx.resolve_operand(assign.args[1]);
        if (!f) {
            return std::unexpected(f.error());
        }
        result = eval_fem_solve(*K, *f);
    }
    else if (assign.callee == "fem_solve" && assign.args.size() == 1) {
        auto sys = ctx.resolve_operand(assign.args[0]);
        if (!sys) {
            return std::unexpected(sys.error());
        }
        result = eval_fem_solve_packed(*sys);
    }

    return result;
}

void ms_register_matrix_call_fem_solve() {
    register_matrix_call("fem_solve", &handle_fem_solve);
}

} // namespace ms::interp
