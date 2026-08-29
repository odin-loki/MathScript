#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_hough_circles(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "hough_circles" &&
               (assign.args.size() == 1 || assign.args.size() == 3)) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        double r_min = 5.0;
        double r_max = 50.0;
        if (assign.args.size() == 3) {
            if (!parse_number(assign.args[1], r_min) || !parse_number(assign.args[2], r_max)) {
                return std::unexpected(DomainError{
                    "hough_circles", "expected hough_circles(M) or hough_circles(M, r_min, r_max)"});
            }
        }
        result = eval_hough_circles(*matrix, 0.5, r_min, r_max, 1, 30);
    }

    return result;
}

void ms_register_matrix_call_hough_circles() {
    register_matrix_call("hough_circles", &handle_hough_circles);
}

} // namespace ms::interp
