#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_signal_lms(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if ((assign.callee == "signal_lms" || assign.callee == "signal_lms_weights") &&
               assign.args.size() == 4) {
        auto x = resolve_operand(assign.args[0]);
        if (!x) {
            return std::unexpected(x.error());
        }
        auto d = resolve_operand(assign.args[1]);
        if (!d) {
            return std::unexpected(d.error());
        }
        auto filter_length_val = parse_scalar_arg(assign.args[2], assign.callee.c_str());
        if (!filter_length_val) {
            return std::unexpected(filter_length_val.error());
        }
        auto mu_val = parse_scalar_arg(assign.args[3], assign.callee.c_str());
        if (!mu_val) {
            return std::unexpected(mu_val.error());
        }
        if (assign.callee == "signal_lms") {
            result = eval_signal_lms(*x, *d, *filter_length_val, *mu_val);
        } else {
            result = eval_signal_lms_weights(*x, *d, *filter_length_val, *mu_val);
        }
    }

    return result;
}

void ms_register_matrix_call_signal_lms() {
    register_matrix_call("signal_lms", &handle_signal_lms);
}

} // namespace ms::interp
