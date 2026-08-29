#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_numthy_primes(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "numthy_primes" && assign.args.size() == 2) {
        double lo_d = 0.0;
        double hi_d = 0.0;
        if (!parse_number(assign.args[0], lo_d)) {
            auto it = ctx.state().scalars.find(assign.args[0]);
            if (it != ctx.state().scalars.end()) {
                lo_d = it->second;
            } else {
                return std::unexpected(
                    DomainError{assign.callee, "expected numeric lo argument"});
            }
        }
        if (!parse_number(assign.args[1], hi_d)) {
            auto it = ctx.state().scalars.find(assign.args[1]);
            if (it != ctx.state().scalars.end()) {
                hi_d = it->second;
            } else {
                return std::unexpected(
                    DomainError{assign.callee, "expected numeric hi argument"});
            }
        }
        if (lo_d < 0.0 || hi_d < 0.0 || std::floor(lo_d) != lo_d || std::floor(hi_d) != hi_d) {
            return std::unexpected(
                DomainError{assign.callee, "expected non-negative integer bounds"});
        }
        auto primes = eval_numthy_primes(static_cast<uint64_t>(lo_d), static_cast<uint64_t>(hi_d));
        if (!primes) {
            return std::unexpected(primes.error());
        }
        result = *primes;
    }

    return result;
}

void ms_register_matrix_call_numthy_primes() {
    register_matrix_call("numthy_primes", &handle_numthy_primes);
}

} // namespace ms::interp
