// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_geo_voronoi(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "geo_voronoi" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto verts = eval_geo_voronoi(*matrix);
        if (!verts) {
            return std::unexpected(verts.error());
        }
        result = *verts;
    }

    return result;
}

void ms_register_matrix_call_geo_voronoi() {
    register_matrix_call("geo_voronoi", &handle_geo_voronoi);
}

} // namespace ms::interp
