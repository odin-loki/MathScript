// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_cfd_advection3d(Interpreter& /*interp*/, const MatrixCallAssign& assign) {
    using namespace detail;

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "cfd_advection3d" && assign.args.size() == 8) {
        double nx_d = 0.0;
        double ny_d = 0.0;
        double nz_d = 0.0;
        double vx = 0.0;
        double vy = 0.0;
        double vz = 0.0;
        double t_end = 0.0;
        double dt = 0.0;
        if (!parse_number(assign.args[0], nx_d) || !parse_number(assign.args[1], ny_d) ||
            !parse_number(assign.args[2], nz_d) || !parse_number(assign.args[3], vx) ||
            !parse_number(assign.args[4], vy) || !parse_number(assign.args[5], vz) ||
            !parse_number(assign.args[6], t_end) || !parse_number(assign.args[7], dt)) {
            return std::unexpected(DomainError{
                "cfd_advection3d",
                "expected cfd_advection3d(nx, ny, nz, vx, vy, vz, t_end, dt)"});
        }
        ExtentBudget budget("cfd_advection3d");
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
        result = eval_cfd_advection3d(*nx, *ny, *nz, vx, vy, vz, t_end, dt);
    }

    return result;
}

void ms_register_matrix_call_cfd_advection3d() {
    register_matrix_call("cfd_advection3d", &handle_cfd_advection3d);
}

} // namespace ms::interp
