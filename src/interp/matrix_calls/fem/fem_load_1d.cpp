// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_fem_load_1d(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "fem_load_1d" && assign.args.size() == 2) {
        auto mesh = ctx.resolve_operand(assign.args[0]);
        if (!mesh) {
            return std::unexpected(mesh.error());
        }
        auto f_val = ctx.parse_scalar_arg(assign.args[1], "fem_load_1d");
        if (!f_val) {
            return std::unexpected(f_val.error());
        }
        result = eval_fem_load_1d(*mesh, *f_val);
    }

    return result;
}

void ms_register_matrix_call_fem_load_1d() {
    register_matrix_call("fem_load_1d", &handle_fem_load_1d);
}

} // namespace ms::interp
