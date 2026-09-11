// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_fem_stiffness_1d(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "fem_stiffness_1d" && assign.args.size() == 1) {
        auto mesh = ctx.resolve_operand(assign.args[0]);
        if (!mesh) {
            return std::unexpected(mesh.error());
        }
        result = eval_fem_stiffness_1d(*mesh);
    }

    return result;
}

void ms_register_matrix_call_fem_stiffness_1d() {
    register_matrix_call("fem_stiffness_1d", &handle_fem_stiffness_1d);
}

} // namespace ms::interp
