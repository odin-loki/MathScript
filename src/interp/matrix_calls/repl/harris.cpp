#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_harris(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "harris" && assign.args.size() >= 1 && assign.args.size() <= 3) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        double k = 0.04;
        double threshold = 0.01;
        if (assign.args.size() >= 2 && !parse_number(assign.args[1], k)) {
            return std::unexpected(
                DomainError{"harris", "expected harris(M), harris(M, k), or harris(M, k, threshold)"});
        }
        if (assign.args.size() == 3 && !parse_number(assign.args[2], threshold)) {
            return std::unexpected(
                DomainError{"harris", "expected harris(M), harris(M, k), or harris(M, k, threshold)"});
        }
        result = eval_harris(*matrix, static_cast<float>(k), static_cast<float>(threshold));
    }

    return result;
}

void ms_register_matrix_call_harris() {
    register_matrix_call("harris", &handle_harris);
}

} // namespace ms::interp
