#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_cfd_run_advection_2d(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "cfd_run_advection_2d" && assign.args.size() == 6) {
        auto grid = resolve_operand(assign.args[0]);
        if (!grid) {
            return std::unexpected(grid.error());
        }
        auto u = resolve_operand(assign.args[1]);
        if (!u) {
            return std::unexpected(u.error());
        }
        auto vx = parse_scalar_arg(assign.args[2], "cfd_run_advection_2d");
        if (!vx) {
            return std::unexpected(vx.error());
        }
        auto vy = parse_scalar_arg(assign.args[3], "cfd_run_advection_2d");
        if (!vy) {
            return std::unexpected(vy.error());
        }
        auto t_end = parse_scalar_arg(assign.args[4], "cfd_run_advection_2d");
        if (!t_end) {
            return std::unexpected(t_end.error());
        }
        auto dt = parse_scalar_arg(assign.args[5], "cfd_run_advection_2d");
        if (!dt) {
            return std::unexpected(dt.error());
        }
        result = eval_cfd_run_advection_2d(*grid, *u, *vx, *vy, *t_end, *dt);
    }

    return result;
}

void ms_register_matrix_call_cfd_run_advection_2d() {
    register_matrix_call("cfd_run_advection_2d", &handle_cfd_run_advection_2d);
}

} // namespace ms::interp
