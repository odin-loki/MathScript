#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_gria_ca_step(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "gria_ca_step" && assign.args.size() == 2) {
        auto state = resolve_operand(assign.args[0]);
        if (!state) {
            return std::unexpected(state.error());
        }
        auto rule_val = parse_scalar_arg(assign.args[1], "gria_ca_step");
        if (!rule_val) {
            return std::unexpected(rule_val.error());
        }
        const int rule = static_cast<int>(*rule_val);
        if (*rule_val != rule || rule < 0 || rule > 255) {
            return std::unexpected(
                DomainError{"gria_ca_step", "expected integer rule in [0,255]"});
        }
        result = eval_gria_ca_step(*state, rule);
    }

    return result;
}

void ms_register_matrix_call_gria_ca_step() {
    register_matrix_call("gria_ca_step", &handle_gria_ca_step);
}

} // namespace ms::interp
