#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_control_kalman_update_cov(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "control_kalman_update_cov" && assign.args.size() == 5) {
        auto x_m = resolve_operand(assign.args[0]);
        if (!x_m) {
            return std::unexpected(x_m.error());
        }
        auto P_m = resolve_operand(assign.args[1]);
        if (!P_m) {
            return std::unexpected(P_m.error());
        }
        auto z_m = resolve_operand(assign.args[2]);
        if (!z_m) {
            return std::unexpected(z_m.error());
        }
        auto H_m = resolve_operand(assign.args[3]);
        if (!H_m) {
            return std::unexpected(H_m.error());
        }
        auto R_m = resolve_operand(assign.args[4]);
        if (!R_m) {
            return std::unexpected(R_m.error());
        }
        auto updated = eval_control_kalman_update_cov(*x_m, *P_m, *z_m, *H_m, *R_m);
        if (!updated) {
            return std::unexpected(updated.error());
        }
        result = *updated;
    }

    return result;
}

void ms_register_matrix_call_control_kalman_update_cov() {
    register_matrix_call("control_kalman_update_cov", &handle_control_kalman_update_cov);
}

} // namespace ms::interp
