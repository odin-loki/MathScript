#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_geo_bezier_deriv(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if ((assign.callee == "geo_bezier_eval" || assign.callee == "geo_bezier_deriv" ||
                assign.callee == "geo_catmull_rom") &&
               assign.args.size() == 2) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        double t = 0.0;
        if (!parse_number(assign.args[1], t)) {
            auto t_expr = eval_scalar_expr(ctx.state(), assign.args[1]);
            if (!t_expr) {
                return std::unexpected(DomainError{
                    assign.callee,
                    assign.callee == "geo_bezier_eval"
                        ? "expected geo_bezier_eval(ctrl, t)"
                        : (assign.callee == "geo_bezier_deriv"
                               ? "expected geo_bezier_deriv(ctrl, t)"
                               : "expected geo_catmull_rom(ctrl, t)")});
            }
            t = *t_expr;
        }
        if (assign.callee == "geo_bezier_eval") {
            result = eval_geo_bezier_eval(*matrix, t);
        } else if (assign.callee == "geo_bezier_deriv") {
            result = eval_geo_bezier_deriv(*matrix, t);
        } else {
            result = eval_geo_catmull_rom(*matrix, t);
        }
    }

    return result;
}

void ms_register_matrix_call_geo_bezier_deriv() {
    register_matrix_call("geo_bezier_deriv", &handle_geo_bezier_deriv);
}

} // namespace ms::interp
