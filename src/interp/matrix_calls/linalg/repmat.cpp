#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_repmat(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "repmat" && assign.args.size() == 3) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        double p_d = 0.0, q_d = 0.0;
        if (!parse_number(assign.args[1], p_d)) {
            return std::unexpected(DomainError{"repmat", "expected numeric row repeat count"});
        }
        if (!parse_number(assign.args[2], q_d)) {
            return std::unexpected(DomainError{"repmat", "expected numeric col repeat count"});
        }
        size_t p = 0;
        size_t q = 0;
        if (!repl_dims_allowed(p_d, q_d, p, q)) {
            return std::unexpected(DomainError{"repmat", kReplMatrixTooLarge});
        }
        const size_t in_r = matrix->rows();
        const size_t in_c = matrix->cols();
        if ((p != 0 && in_r > kMaxReplMatrixElems / p) ||
            (q != 0 && in_c > kMaxReplMatrixElems / q) ||
            !repl_elems_allowed(in_r * p, in_c * q)) {
            return std::unexpected(DomainError{"repmat", kReplMatrixTooLarge});
        }
        result = repmat(*matrix, p, q);
    }

    return result;
}

void ms_register_matrix_call_repmat() {
    register_matrix_call("repmat", &handle_repmat);
}

} // namespace ms::interp
