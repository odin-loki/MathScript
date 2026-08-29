#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_diffgeo_surface_normal_sphere(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "diffgeo_surface_normal_sphere" && assign.args.size() == 2) {
        double u = 0.0;
        double v = 0.0;
        if (!parse_number(assign.args[0], u)) {
            auto u_expr = eval_scalar_expr(ctx.state(), assign.args[0]);
            if (!u_expr) {
                return std::unexpected(u_expr.error());
            }
            u = *u_expr;
        }
        if (!parse_number(assign.args[1], v)) {
            auto v_expr = eval_scalar_expr(ctx.state(), assign.args[1]);
            if (!v_expr) {
                return std::unexpected(v_expr.error());
            }
            v = *v_expr;
        }
        auto normal = eval_diffgeo_surface_normal_sphere(u, v);
        if (!normal) {
            return std::unexpected(normal.error());
        }
        result = *normal;
    }

    return result;
}

void ms_register_matrix_call_diffgeo_surface_normal_sphere() {
    register_matrix_call("diffgeo_surface_normal_sphere", &handle_diffgeo_surface_normal_sphere);
}

} // namespace ms::interp
