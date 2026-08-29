#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_mtf_decode_vec(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "mtf_decode_vec" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        if (matrix->cols() != 1) {
            return std::unexpected(
                DomainError{"mtf_decode_vec", "expected Nx1 encoded byte vector"});
        }
        compress::Bytes encoded;
        encoded.reserve(matrix->rows());
        for (size_t i = 0; i < matrix->rows(); ++i) {
            const double v = (*matrix)(i, 0);
            if (v < 0.0 || v > 255.0 || std::floor(v) != v) {
                return std::unexpected(
                    DomainError{"mtf_decode_vec", "encoded values must be uint8 in [0,255]"});
            }
            encoded.push_back(static_cast<uint8_t>(v));
        }
        result = bytes_to_matrix_col(compress::mtf_decode(encoded));
    }

    return result;
}

void ms_register_matrix_call_mtf_decode_vec() {
    register_matrix_call("mtf_decode_vec", &handle_mtf_decode_vec);
}

} // namespace ms::interp
