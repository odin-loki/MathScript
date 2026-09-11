// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_fem_poisson3d(Interpreter& /*interp*/, const MatrixCallAssign& assign) {
    using namespace detail;

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "fem_poisson3d" && assign.args.size() == 3) {
        double nx_d = 0.0;
        double ny_d = 0.0;
        double nz_d = 0.0;
        if (!parse_number(assign.args[0], nx_d) || !parse_number(assign.args[1], ny_d) ||
            !parse_number(assign.args[2], nz_d)) {
            return std::unexpected(
                DomainError{"fem_poisson3d", "expected fem_poisson3d(nx, ny, nz)"});
        }
        ExtentBudget budget("fem_poisson3d");
        auto nx = budget.take("nx", nx_d);
        if (!nx) {
            return std::unexpected(nx.error());
        }
        auto ny = budget.take("ny", ny_d);
        if (!ny) {
            return std::unexpected(ny.error());
        }
        auto nz = budget.take("nz", nz_d);
        if (!nz) {
            return std::unexpected(nz.error());
        }
        // The result is a vector of node values, which is what the extent budget above
        // charged. The solve goes through a DENSE nodes x nodes stiffness matrix, and
        // that is what actually gets allocated, so the order is charged a second time.
        auto order = budget.charge_dense_order("the mesh", (*nx + 1) * (*ny + 1) * (*nz + 1));
        if (!order) {
            return std::unexpected(order.error());
        }
        result = eval_fem_poisson3d(*nx, *ny, *nz);
    }

    return result;
}

void ms_register_matrix_call_fem_poisson3d() {
    register_matrix_call("fem_poisson3d", &handle_fem_poisson3d);
}

} // namespace ms::interp
