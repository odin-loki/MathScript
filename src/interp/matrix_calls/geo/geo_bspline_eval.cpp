#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_geo_bspline_eval(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "geo_bspline_eval" && assign.args.size() == 4) {
        auto ctrl = resolve_operand(assign.args[0]);
        if (!ctrl) {
            return std::unexpected(ctrl.error());
        }
        auto knots = resolve_operand(assign.args[1]);
        if (!knots) {
            return std::unexpected(knots.error());
        }
        double degree = 0.0;
        double t = 0.0;
        if (!parse_number(assign.args[2], degree)) {
            auto deg_expr = eval_scalar_expr(ctx.state(), assign.args[2]);
            if (!deg_expr) {
                return std::unexpected(DomainError{
                    "geo_bspline_eval", "expected geo_bspline_eval(ctrl, knots, degree, t)"});
            }
            degree = *deg_expr;
        }
        if (!parse_number(assign.args[3], t)) {
            auto t_expr = eval_scalar_expr(ctx.state(), assign.args[3]);
            if (!t_expr) {
                return std::unexpected(DomainError{
                    "geo_bspline_eval", "expected geo_bspline_eval(ctrl, knots, degree, t)"});
            }
            t = *t_expr;
        }
        result = eval_geo_bspline_eval(*ctrl, *knots, degree, t);
    }

    return result;
}

void ms_register_matrix_call_geo_bspline_eval() {
    register_matrix_call("geo_bspline_eval", &handle_geo_bspline_eval);
}

} // namespace ms::interp
