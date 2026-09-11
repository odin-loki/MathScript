// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_ode_adams_bashforth2(Interpreter& /*interp*/, const MatrixCallAssign& assign) {
    using namespace detail;

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if ((assign.callee == "ode_adams_bashforth2" ||
                assign.callee == "ode_backward_euler" || assign.callee == "ode_bdf2") &&
               assign.args.size() == 5) {
        OdeResult (*solver)(OdeFunc, double, double, double, size_t) = nullptr;
        if (assign.callee == "ode_adams_bashforth2") {
            solver = ode_adams_bashforth2;
        } else if (assign.callee == "ode_backward_euler") {
            solver = ode_backward_euler;
        } else {
            solver = ode_bdf2;
        }
        result = eval_ode_fixed_step_matrix(
            assign.callee, trim_copy(assign.args[0]), trim_copy(assign.args[1]),
            trim_copy(assign.args[2]), trim_copy(assign.args[3]), trim_copy(assign.args[4]),
            solver);
    }

    return result;
}

void ms_register_matrix_call_ode_adams_bashforth2() {
    register_matrix_call("ode_adams_bashforth2", &handle_ode_adams_bashforth2);
}

} // namespace ms::interp
