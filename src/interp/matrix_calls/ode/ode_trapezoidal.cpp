#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_ode_trapezoidal(Interpreter& /*interp*/, const MatrixCallAssign& assign) {
    using namespace detail;

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "ode_trapezoidal" && assign.args.size() == 5) {
        result = eval_ode_fixed_step_matrix(
            assign.callee, trim_copy(assign.args[0]), trim_copy(assign.args[1]),
            trim_copy(assign.args[2]), trim_copy(assign.args[3]), trim_copy(assign.args[4]),
            ode_trapezoidal_wrapped);
    }

    return result;
}

void ms_register_matrix_call_ode_trapezoidal() {
    register_matrix_call("ode_trapezoidal", &handle_ode_trapezoidal);
}

} // namespace ms::interp
