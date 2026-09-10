// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_fem_mesh2d_rectangular(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if ((assign.callee == "fem_mesh2d_rectangular" || assign.callee == "fem_mesh2d") &&
               assign.args.size() == 6) {
        const char* fn = assign.callee.c_str();
        auto x0 = ctx.parse_scalar_arg(assign.args[0], fn);
        if (!x0) {
            return std::unexpected(x0.error());
        }
        auto y0 = ctx.parse_scalar_arg(assign.args[1], fn);
        if (!y0) {
            return std::unexpected(y0.error());
        }
        auto x1 = ctx.parse_scalar_arg(assign.args[2], fn);
        if (!x1) {
            return std::unexpected(x1.error());
        }
        auto y1 = ctx.parse_scalar_arg(assign.args[3], fn);
        if (!y1) {
            return std::unexpected(y1.error());
        }
        auto nx = ctx.parse_scalar_arg(assign.args[4], fn);
        if (!nx) {
            return std::unexpected(nx.error());
        }
        auto ny = ctx.parse_scalar_arg(assign.args[5], fn);
        if (!ny) {
            return std::unexpected(ny.error());
        }
        auto nx_i = ctx.parse_positive_size_arg(*nx, fn, "expected positive integer nx");
        if (!nx_i) {
            return std::unexpected(nx_i.error());
        }
        auto ny_i = ctx.parse_positive_size_arg(*ny, fn, "expected positive integer ny");
        if (!ny_i) {
            return std::unexpected(ny_i.error());
        }
        result = eval_fem_mesh2d_rectangular(*x0, *y0, *x1, *y1, *nx_i, *ny_i);
    }

    return result;
}

void ms_register_matrix_call_fem_mesh2d_rectangular() {
    register_matrix_call("fem_mesh2d_rectangular", &handle_fem_mesh2d_rectangular);
}

} // namespace ms::interp
