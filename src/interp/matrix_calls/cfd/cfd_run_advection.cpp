#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_cfd_run_advection(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "cfd_run_advection" && assign.args.size() == 5) {
        auto grid = resolve_operand(assign.args[0]);
        if (!grid) {
            return std::unexpected(grid.error());
        }
        auto u = resolve_operand(assign.args[1]);
        if (!u) {
            return std::unexpected(u.error());
        }
        auto v = parse_scalar_arg(assign.args[2], "cfd_run_advection");
        if (!v) {
            return std::unexpected(v.error());
        }
        auto t_end = parse_scalar_arg(assign.args[3], "cfd_run_advection");
        if (!t_end) {
            return std::unexpected(t_end.error());
        }
        auto dt = parse_scalar_arg(assign.args[4], "cfd_run_advection");
        if (!dt) {
            return std::unexpected(dt.error());
        }
        result = eval_cfd_run_advection(*grid, *u, *v, *t_end, *dt);
    }

    return result;
}

void ms_register_matrix_call_cfd_run_advection() {
    register_matrix_call("cfd_run_advection", &handle_cfd_run_advection);
}

} // namespace ms::interp
