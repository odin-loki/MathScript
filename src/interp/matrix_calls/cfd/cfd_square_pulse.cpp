#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_cfd_square_pulse(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "cfd_square_pulse" &&
               (assign.args.size() == 3 || assign.args.size() == 4)) {
        auto grid = ctx.resolve_operand(assign.args[0]);
        if (!grid) {
            return std::unexpected(grid.error());
        }
        auto xc = ctx.parse_scalar_arg(assign.args[1], "cfd_square_pulse");
        if (!xc) {
            return std::unexpected(xc.error());
        }
        auto width = ctx.parse_scalar_arg(assign.args[2], "cfd_square_pulse");
        if (!width) {
            return std::unexpected(width.error());
        }
        double amplitude = 1.0;
        if (assign.args.size() == 4) {
            auto amp = ctx.parse_scalar_arg(assign.args[3], "cfd_square_pulse");
            if (!amp) {
                return std::unexpected(amp.error());
            }
            amplitude = *amp;
        }
        result = eval_cfd_square_pulse(*grid, *xc, *width, amplitude);
    }

    return result;
}

void ms_register_matrix_call_cfd_square_pulse() {
    register_matrix_call("cfd_square_pulse", &handle_cfd_square_pulse);
}

} // namespace ms::interp
