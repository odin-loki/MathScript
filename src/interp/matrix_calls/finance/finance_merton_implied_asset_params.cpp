#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_finance_merton_implied_asset_params(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "finance_merton_implied_asset_params" && assign.args.size() == 5) {
        double E = 0.0;
        double sigma_E = 0.0;
        double D = 0.0;
        double r = 0.0;
        double T = 0.0;
        if (auto v = parse_scalar_arg(assign.args[0], "finance_merton_implied_asset_params")) {
            E = *v;
        } else {
            return std::unexpected(v.error());
        }
        if (auto v = parse_scalar_arg(assign.args[1], "finance_merton_implied_asset_params")) {
            sigma_E = *v;
        } else {
            return std::unexpected(v.error());
        }
        if (auto v = parse_scalar_arg(assign.args[2], "finance_merton_implied_asset_params")) {
            D = *v;
        } else {
            return std::unexpected(v.error());
        }
        if (auto v = parse_scalar_arg(assign.args[3], "finance_merton_implied_asset_params")) {
            r = *v;
        } else {
            return std::unexpected(v.error());
        }
        if (auto v = parse_scalar_arg(assign.args[4], "finance_merton_implied_asset_params")) {
            T = *v;
        } else {
            return std::unexpected(v.error());
        }
        auto value = eval_finance_merton_implied_asset_params(E, sigma_E, D, r, T);
        if (!value) {
            return std::unexpected(value.error());
        }
        result = *value;
    }

    return result;
}

void ms_register_matrix_call_finance_merton_implied_asset_params() {
    register_matrix_call("finance_merton_implied_asset_params", &handle_finance_merton_implied_asset_params);
}

} // namespace ms::interp
