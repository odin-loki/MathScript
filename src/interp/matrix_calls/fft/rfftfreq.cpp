#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_rfftfreq(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "rfftfreq" &&
               (assign.args.size() == 1 || assign.args.size() == 2)) {
        auto n_val = parse_scalar_arg(assign.args[0], "rfftfreq");
        if (!n_val) {
            return std::unexpected(n_val.error());
        }
        const int n_i = static_cast<int>(*n_val);
        if (n_i < 0 || *n_val != n_i) {
            return std::unexpected(
                DomainError{"rfftfreq", "expected non-negative integer n"});
        }
        double d = 1.0;
        if (assign.args.size() == 2) {
            auto d_val = parse_scalar_arg(assign.args[1], "rfftfreq");
            if (!d_val) {
                return std::unexpected(d_val.error());
            }
            d = *d_val;
        }
        auto freqs = eval_rfftfreq(static_cast<size_t>(n_i), d);
        if (!freqs) {
            return std::unexpected(freqs.error());
        }
        result = *freqs;
    }

    return result;
}

void ms_register_matrix_call_rfftfreq() {
    register_matrix_call("rfftfreq", &handle_rfftfreq);
}

} // namespace ms::interp
