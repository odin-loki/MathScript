#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_izaac_rand_matrix(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "izaac_rand_matrix" && assign.args.size() == 2) {
        auto rows_val = parse_scalar_arg(assign.args[0], "izaac_rand_matrix");
        if (!rows_val) {
            return std::unexpected(rows_val.error());
        }
        auto cols_val = parse_scalar_arg(assign.args[1], "izaac_rand_matrix");
        if (!cols_val) {
            return std::unexpected(cols_val.error());
        }
        auto rows_i = parse_positive_size_arg(*rows_val, "izaac_rand_matrix", "expected positive integer rows");
        if (!rows_i) {
            return std::unexpected(rows_i.error());
        }
        auto cols_i = parse_positive_size_arg(*cols_val, "izaac_rand_matrix", "expected positive integer cols");
        if (!cols_i) {
            return std::unexpected(cols_i.error());
        }
        result = eval_izaac_rand_matrix(*rows_i, *cols_i);
    }

    return result;
}

void ms_register_matrix_call_izaac_rand_matrix() {
    register_matrix_call("izaac_rand_matrix", &handle_izaac_rand_matrix);
}

} // namespace ms::interp
