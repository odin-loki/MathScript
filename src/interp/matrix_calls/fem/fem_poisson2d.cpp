// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_fem_poisson2d(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "fem_poisson2d" && assign.args.size() == 2) {
        auto nx_val = ctx.parse_scalar_arg(assign.args[0], "fem_poisson2d");
        if (!nx_val) {
            return std::unexpected(nx_val.error());
        }
        auto ny_val = ctx.parse_scalar_arg(assign.args[1], "fem_poisson2d");
        if (!ny_val) {
            return std::unexpected(ny_val.error());
        }
        ExtentBudget budget("fem_poisson2d");
        auto nx = budget.take("nx", *nx_val);
        if (!nx) {
            return std::unexpected(nx.error());
        }
        auto ny = budget.take("ny", *ny_val);
        if (!ny) {
            return std::unexpected(ny.error());
        }
        // The result is a vector of node values, which is what the extent budget above
        // charged. The solve goes through a DENSE nodes x nodes stiffness matrix, and
        // that is what actually gets allocated, so the order is charged a second time.
        auto order = budget.charge_dense_order("the mesh", (*nx + 1) * (*ny + 1));
        if (!order) {
            return std::unexpected(order.error());
        }
        result = eval_fem_poisson2d(*nx, *ny);
    }

    return result;
}

void ms_register_matrix_call_fem_poisson2d() {
    register_matrix_call("fem_poisson2d", &handle_fem_poisson2d);
}

} // namespace ms::interp
