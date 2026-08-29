#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_poly_partial_fractions(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "poly_partial_fractions" && assign.args.size() == 2) {
        auto num = resolve_operand(assign.args[0]);
        if (!num) {
            return std::unexpected(num.error());
        }
        auto den = resolve_operand(assign.args[1]);
        if (!den) {
            return std::unexpected(den.error());
        }
        auto pf = eval_poly_partial_fractions(*num, *den);
        if (!pf) {
            return std::unexpected(pf.error());
        }
        result = *pf;
    }

    return result;
}

void ms_register_matrix_call_poly_partial_fractions() {
    register_matrix_call("poly_partial_fractions", &handle_poly_partial_fractions);
}

} // namespace ms::interp
