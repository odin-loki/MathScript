#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_pde_heat_2d(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "pde_heat_2d" && assign.args.size() == 6) {
        auto u0_m = resolve_operand(assign.args[0]);
        if (!u0_m) {
            return std::unexpected(u0_m.error());
        }
        auto alpha = parse_scalar_arg(assign.args[1], "pde_heat_2d");
        if (!alpha) {
            return std::unexpected(alpha.error());
        }
        auto dx = parse_scalar_arg(assign.args[2], "pde_heat_2d");
        if (!dx) {
            return std::unexpected(dx.error());
        }
        auto dy = parse_scalar_arg(assign.args[3], "pde_heat_2d");
        if (!dy) {
            return std::unexpected(dy.error());
        }
        auto dt = parse_scalar_arg(assign.args[4], "pde_heat_2d");
        if (!dt) {
            return std::unexpected(dt.error());
        }
        auto steps_val = parse_scalar_arg(assign.args[5], "pde_heat_2d");
        if (!steps_val) {
            return std::unexpected(steps_val.error());
        }
        const int steps_i = static_cast<int>(*steps_val);
        if (steps_i < 0 || *steps_val != steps_i) {
            return std::unexpected(
                DomainError{"pde_heat_2d", "expected non-negative integer steps"});
        }
        result = eval_pde_heat_2d(*u0_m, *alpha, *dx, *dy, *dt, static_cast<std::size_t>(steps_i));
    }

    return result;
}

void ms_register_matrix_call_pde_heat_2d() {
    register_matrix_call("pde_heat_2d", &handle_pde_heat_2d);
}

} // namespace ms::interp
