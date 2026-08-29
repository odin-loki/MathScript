#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_control_feedback(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "control_feedback" &&
               (assign.args.size() == 4 || assign.args.size() == 5)) {
        auto numG = resolve_operand(assign.args[0]);
        if (!numG) {
            return std::unexpected(numG.error());
        }
        auto denG = resolve_operand(assign.args[1]);
        if (!denG) {
            return std::unexpected(denG.error());
        }
        auto numH = resolve_operand(assign.args[2]);
        if (!numH) {
            return std::unexpected(numH.error());
        }
        auto denH = resolve_operand(assign.args[3]);
        if (!denH) {
            return std::unexpected(denH.error());
        }
        int sign = -1;
        if (assign.args.size() == 5) {
            double sign_d = 0.0;
            if (!parse_number(assign.args[4], sign_d)) {
                auto sign_expr = eval_scalar_expr(ctx.state(), assign.args[4]);
                if (!sign_expr) {
                    return std::unexpected(
                        DomainError{"control_feedback", "expected feedback sign -1 or +1"});
                }
                sign_d = *sign_expr;
            }
            if (sign_d != -1.0 && sign_d != 1.0) {
                return std::unexpected(
                    DomainError{"control_feedback", "expected feedback sign -1 or +1"});
            }
            sign = static_cast<int>(sign_d);
        }
        result = eval_control_feedback(*numG, *denG, *numH, *denH, sign);
    }

    return result;
}

void ms_register_matrix_call_control_feedback() {
    register_matrix_call("control_feedback", &handle_control_feedback);
}

} // namespace ms::interp
