#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_imadjust(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "imadjust" &&
               (assign.args.size() == 3 || assign.args.size() == 5)) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        double in_lo = 0.0;
        double in_hi = 0.0;
        double out_lo = 0.0;
        double out_hi = 1.0;
        if (!parse_number(assign.args[1], in_lo) || !parse_number(assign.args[2], in_hi)) {
            return std::unexpected(
                DomainError{"imadjust", "expected imadjust(M, in_lo, in_hi[, out_lo, out_hi])"});
        }
        if (assign.args.size() == 5) {
            if (!parse_number(assign.args[3], out_lo) || !parse_number(assign.args[4], out_hi)) {
                return std::unexpected(DomainError{
                    "imadjust", "expected imadjust(M, in_lo, in_hi[, out_lo, out_hi])"});
            }
        }
        auto gray = matrix_to_gray_image(*matrix);
        if (!gray) {
            return std::unexpected(gray.error());
        }
        result = gray_image_to_matrix(image::imadjust(
            *gray, static_cast<float>(in_lo), static_cast<float>(in_hi),
            static_cast<float>(out_lo), static_cast<float>(out_hi)));
    }

    return result;
}

void ms_register_matrix_call_imadjust() {
    register_matrix_call("imadjust", &handle_imadjust);
}

} // namespace ms::interp
