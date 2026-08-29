#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_cfd_square_pulse_3d(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);
    auto resolve_operand = [&ctx](const std::string& text) { return ctx.resolve_operand(text); };
    auto parse_scalar_arg = [&ctx](const std::string& arg_text, const char* fn) -> Result<double> {
        double value = 0.0;
        if (parse_number(arg_text, value)) return value;
        auto expr = eval_scalar_expr(ctx.state(), arg_text);
        if (!expr) {
            return std::unexpected(DomainError{fn, "expected numeric scalar argument"});
        }
        return *expr;
    };
    auto parse_positive_size_arg = [](double value, const char* fn, const char* label) -> Result<std::size_t> {
        const int i = static_cast<int>(value);
        if (i < 1 || value != static_cast<double>(i)) {
            return std::unexpected(DomainError{fn, label});
        }
        return static_cast<std::size_t>(i);
    };
    auto parse_uint64_arg = [](double value, const char* fn, const char* label) -> Result<uint64_t> {
        if (value < 0.0 || value != std::floor(value)) {
            return std::unexpected(DomainError{fn, label});
        }
        return static_cast<uint64_t>(value);
    };

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "cfd_square_pulse_3d" &&
               (assign.args.size() == 7 || assign.args.size() == 8)) {
        auto grid = resolve_operand(assign.args[0]);
        if (!grid) {
            return std::unexpected(grid.error());
        }
        auto xc = parse_scalar_arg(assign.args[1], "cfd_square_pulse_3d");
        if (!xc) {
            return std::unexpected(xc.error());
        }
        auto yc = parse_scalar_arg(assign.args[2], "cfd_square_pulse_3d");
        if (!yc) {
            return std::unexpected(yc.error());
        }
        auto zc = parse_scalar_arg(assign.args[3], "cfd_square_pulse_3d");
        if (!zc) {
            return std::unexpected(zc.error());
        }
        auto width_x = parse_scalar_arg(assign.args[4], "cfd_square_pulse_3d");
        if (!width_x) {
            return std::unexpected(width_x.error());
        }
        auto width_y = parse_scalar_arg(assign.args[5], "cfd_square_pulse_3d");
        if (!width_y) {
            return std::unexpected(width_y.error());
        }
        auto width_z = parse_scalar_arg(assign.args[6], "cfd_square_pulse_3d");
        if (!width_z) {
            return std::unexpected(width_z.error());
        }
        double amplitude = 1.0;
        if (assign.args.size() == 8) {
            auto amp = parse_scalar_arg(assign.args[7], "cfd_square_pulse_3d");
            if (!amp) {
                return std::unexpected(amp.error());
            }
            amplitude = *amp;
        }
        result = eval_cfd_square_pulse_3d(*grid, *xc, *yc, *zc, *width_x, *width_y, *width_z,
                                          amplitude);
    }

    return result;
}

void ms_register_matrix_call_cfd_square_pulse_3d() {
    register_matrix_call("cfd_square_pulse_3d", &handle_cfd_square_pulse_3d);
}

} // namespace ms::interp
