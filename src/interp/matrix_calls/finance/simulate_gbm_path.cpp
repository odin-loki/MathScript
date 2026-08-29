#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_simulate_gbm_path(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "simulate_gbm_path" && assign.args.size() == 5) {
        const char* fn = "simulate_gbm_path";
        std::array<Result<double>, 5> scalars{};
        for (size_t i = 0; i < 5; ++i) {
            scalars[i] = parse_scalar_arg(assign.args[i], fn);
            if (!scalars[i]) {
                return std::unexpected(scalars[i].error());
            }
        }
        auto steps_i = parse_positive_size_arg(*scalars[4], fn, "expected positive integer steps");
        if (!steps_i) {
            return std::unexpected(steps_i.error());
        }
        result = eval_simulate_gbm_path(*scalars[0], *scalars[1], *scalars[2], *scalars[3], *steps_i);
    }

    return result;
}

void ms_register_matrix_call_simulate_gbm_path() {
    register_matrix_call("simulate_gbm_path", &handle_simulate_gbm_path);
}

} // namespace ms::interp
