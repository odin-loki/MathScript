#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_gria_divergence_trajectory(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "gria_divergence_trajectory" && assign.args.size() == 4) {
        auto a = resolve_operand(assign.args[0]);
        if (!a) {
            return std::unexpected(a.error());
        }
        auto b = resolve_operand(assign.args[1]);
        if (!b) {
            return std::unexpected(b.error());
        }
        auto rule_val = parse_scalar_arg(assign.args[2], "gria_divergence_trajectory");
        if (!rule_val) {
            return std::unexpected(rule_val.error());
        }
        auto n_steps_val = parse_scalar_arg(assign.args[3], "gria_divergence_trajectory");
        if (!n_steps_val) {
            return std::unexpected(n_steps_val.error());
        }
        const int rule = static_cast<int>(*rule_val);
        if (*rule_val != rule || rule < 0 || rule > 255) {
            return std::unexpected(
                DomainError{"gria_divergence_trajectory", "expected integer rule in [0,255]"});
        }
        const int n_steps = static_cast<int>(*n_steps_val);
        if (*n_steps_val != n_steps) {
            return std::unexpected(DomainError{
                "gria_divergence_trajectory", "expected integer n_steps"});
        }
        result = eval_gria_divergence_trajectory(*a, *b, rule, n_steps);
    }

    return result;
}

void ms_register_matrix_call_gria_divergence_trajectory() {
    register_matrix_call("gria_divergence_trajectory", &handle_gria_divergence_trajectory);
}

} // namespace ms::interp
