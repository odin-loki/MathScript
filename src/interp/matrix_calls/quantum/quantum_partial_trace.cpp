#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_quantum_partial_trace(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "quantum_partial_trace" && assign.args.size() == 4) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto d1_val = parse_scalar_arg(assign.args[1], "quantum_partial_trace");
        if (!d1_val) {
            return std::unexpected(d1_val.error());
        }
        auto d2_val = parse_scalar_arg(assign.args[2], "quantum_partial_trace");
        if (!d2_val) {
            return std::unexpected(d2_val.error());
        }
        auto sub_val = parse_scalar_arg(assign.args[3], "quantum_partial_trace");
        if (!sub_val) {
            return std::unexpected(sub_val.error());
        }
        const int d1 = static_cast<int>(*d1_val);
        const int d2 = static_cast<int>(*d2_val);
        const int subsystem = static_cast<int>(*sub_val);
        if (d1 < 1 || d2 < 1 || *d1_val != d1 || *d2_val != d2 ||
            (subsystem != 0 && subsystem != 1) || *sub_val != subsystem) {
            return std::unexpected(DomainError{
                "quantum_partial_trace",
                "expected positive integer d1, d2 and subsystem 0 or 1"});
        }
        result = eval_quantum_partial_trace_matrix(*matrix, d1, d2, subsystem);
    }

    return result;
}

void ms_register_matrix_call_quantum_partial_trace() {
    register_matrix_call("quantum_partial_trace", &handle_quantum_partial_trace);
}

} // namespace ms::interp
