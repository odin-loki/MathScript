// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_fem_poisson1d(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "fem_poisson1d" && assign.args.size() == 1) {
        auto n_val = ctx.parse_scalar_arg(assign.args[0], "fem_poisson1d");
        if (!n_val) {
            return std::unexpected(n_val.error());
        }
        ExtentBudget budget("fem_poisson1d");
        auto n = budget.take("n", *n_val);
        if (!n) {
            return std::unexpected(n.error());
        }
        // The result is a vector of node values, which is what the extent budget above
        // charged. The solve goes through a DENSE nodes x nodes stiffness matrix, and
        // that is what actually gets allocated, so the order is charged a second time.
        auto order = budget.charge_dense_order("the mesh", *n + 1);
        if (!order) {
            return std::unexpected(order.error());
        }
        result = eval_fem_poisson1d(*n);
    }

    return result;
}

void ms_register_matrix_call_fem_poisson1d() {
    register_matrix_call("fem_poisson1d", &handle_fem_poisson1d);
}

} // namespace ms::interp
