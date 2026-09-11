// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_cfd_square_pulse_2d(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "cfd_square_pulse_2d" &&
               (assign.args.size() == 5 || assign.args.size() == 6)) {
        auto grid = ctx.resolve_operand(assign.args[0]);
        if (!grid) {
            return std::unexpected(grid.error());
        }
        auto xc = ctx.parse_scalar_arg(assign.args[1], "cfd_square_pulse_2d");
        if (!xc) {
            return std::unexpected(xc.error());
        }
        auto yc = ctx.parse_scalar_arg(assign.args[2], "cfd_square_pulse_2d");
        if (!yc) {
            return std::unexpected(yc.error());
        }
        auto width_x = ctx.parse_scalar_arg(assign.args[3], "cfd_square_pulse_2d");
        if (!width_x) {
            return std::unexpected(width_x.error());
        }
        auto width_y = ctx.parse_scalar_arg(assign.args[4], "cfd_square_pulse_2d");
        if (!width_y) {
            return std::unexpected(width_y.error());
        }
        double amplitude = 1.0;
        if (assign.args.size() == 6) {
            auto amp = ctx.parse_scalar_arg(assign.args[5], "cfd_square_pulse_2d");
            if (!amp) {
                return std::unexpected(amp.error());
            }
            amplitude = *amp;
        }
        result = eval_cfd_square_pulse_2d(*grid, *xc, *yc, *width_x, *width_y, amplitude);
    }

    return result;
}

void ms_register_matrix_call_cfd_square_pulse_2d() {
    register_matrix_call("cfd_square_pulse_2d", &handle_cfd_square_pulse_2d);
}

} // namespace ms::interp
