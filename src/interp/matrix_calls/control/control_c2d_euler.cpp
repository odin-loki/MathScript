#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_control_c2d_euler(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if ((assign.callee == "control_c2d_tustin" || assign.callee == "control_c2d_euler" ||
                assign.callee == "control_d2c_tustin" || assign.callee == "control_d2c_euler") &&
               assign.args.size() == 5) {
        auto A_m = resolve_operand(assign.args[0]);
        if (!A_m) {
            return std::unexpected(A_m.error());
        }
        auto B_m = resolve_operand(assign.args[1]);
        if (!B_m) {
            return std::unexpected(B_m.error());
        }
        auto C_m = resolve_operand(assign.args[2]);
        if (!C_m) {
            return std::unexpected(C_m.error());
        }
        auto D_m = resolve_operand(assign.args[3]);
        if (!D_m) {
            return std::unexpected(D_m.error());
        }
        double Ts = 0.0;
        if (!parse_number(assign.args[4], Ts)) {
            auto ts_expr = eval_scalar_expr(ctx.state(), assign.args[4]);
            if (!ts_expr) {
                return std::unexpected(DomainError{assign.callee, "expected positive Ts"});
            }
            Ts = *ts_expr;
        }
        const char* fn = assign.callee.c_str();
        const control::DiscretizationMethod method =
            (assign.callee == "control_c2d_tustin" || assign.callee == "control_d2c_tustin")
                ? control::DiscretizationMethod::Tustin
                : control::DiscretizationMethod::Euler;
        if (assign.callee == "control_c2d_tustin" || assign.callee == "control_c2d_euler") {
            result = eval_control_c2d(*A_m, *B_m, *C_m, *D_m, Ts, method, fn);
        } else {
            result = eval_control_d2c(*A_m, *B_m, *C_m, *D_m, Ts, method, fn);
        }
    }

    return result;
}

void ms_register_matrix_call_control_c2d_euler() {
    register_matrix_call("control_c2d_euler", &handle_control_c2d_euler);
}

} // namespace ms::interp
