#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_imhist(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "imhist" &&
               (assign.args.size() == 1 || assign.args.size() == 2)) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        int nbins = 256;
        if (assign.args.size() == 2) {
            double nbins_d = 0.0;
            if (!parse_number(assign.args[1], nbins_d)) {
                return std::unexpected(DomainError{"imhist", "expected imhist(M[, nbins])"});
            }
            nbins = static_cast<int>(nbins_d);
            if (nbins < 1 || nbins_d != nbins) {
                return std::unexpected(
                    DomainError{"imhist", "expected positive integer nbins"});
            }
        }
        auto gray = matrix_to_gray_image(*matrix);
        if (!gray) {
            return std::unexpected(gray.error());
        }
        const auto hist = image::imhist(*gray, nbins);
        Matrix<double> out(hist.size(), 1);
        for (size_t i = 0; i < hist.size(); ++i) {
            out(i, 0) = static_cast<double>(hist[i]);
        }
        result = out;
    }

    return result;
}

void ms_register_matrix_call_imhist() {
    register_matrix_call("imhist", &handle_imhist);
}

} // namespace ms::interp
