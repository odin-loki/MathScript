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
        const int nx_i = static_cast<int>(nx_d);
        const int ny_i = static_cast<int>(ny_d);
        const int nz_i = static_cast<int>(nz_d);
        if (nx_i < 0 || ny_i < 0 || nz_i < 0 || nx_d != nx_i || ny_d != ny_i || nz_d != nz_i) {
            return std::unexpected(
                DomainError{"fem_poisson3d", "expected non-negative integer nx, ny, and nz"});
        }
        result = eval_fem_poisson3d(static_cast<std::size_t>(nx_i), static_cast<std::size_t>(ny_i),
                                    static_cast<std::size_t>(nz_i));
    }

    return result;
}

void ms_register_matrix_call_fem_poisson3d() {
    register_matrix_call("fem_poisson3d", &handle_fem_poisson3d);
}

} // namespace ms::interp
