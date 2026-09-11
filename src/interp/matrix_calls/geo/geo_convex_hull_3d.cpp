// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_geo_convex_hull_3d(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "geo_convex_hull_3d" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto faces = eval_geo_convex_hull_3d(*matrix);
        if (!faces) {
            return std::unexpected(faces.error());
        }
        result = *faces;
    }

    return result;
}

void ms_register_matrix_call_geo_convex_hull_3d() {
    register_matrix_call("geo_convex_hull_3d", &handle_geo_convex_hull_3d);
}

} // namespace ms::interp
