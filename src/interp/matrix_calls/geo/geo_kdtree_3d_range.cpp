// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_geo_kdtree_3d_range(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if ((assign.callee == "geo_kdtree_3d_knn" || assign.callee == "geo_kdtree_3d_range") &&
               assign.args.size() == 5) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto qx = ctx.parse_scalar_arg(assign.args[1], assign.callee.c_str());
        if (!qx) {
            return std::unexpected(qx.error());
        }
        auto qy = ctx.parse_scalar_arg(assign.args[2], assign.callee.c_str());
        if (!qy) {
            return std::unexpected(qy.error());
        }
        auto qz = ctx.parse_scalar_arg(assign.args[3], assign.callee.c_str());
        if (!qz) {
            return std::unexpected(qz.error());
        }
        auto arg4 = ctx.parse_scalar_arg(assign.args[4], assign.callee.c_str());
        if (!arg4) {
            return std::unexpected(arg4.error());
        }
        if (assign.callee == "geo_kdtree_3d_knn") {
            result = eval_geo_kdtree_3d_knn(*matrix, *qx, *qy, *qz, *arg4);
        } else {
            result = eval_geo_kdtree_3d_range(*matrix, *qx, *qy, *qz, *arg4);
        }
    }

    return result;
}

void ms_register_matrix_call_geo_kdtree_3d_range() {
    register_matrix_call("geo_kdtree_3d_range", &handle_geo_kdtree_3d_range);
}

} // namespace ms::interp
