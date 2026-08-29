#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_sparse_from_coo(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "sparse_from_coo" && assign.args.size() == 5) {
        auto rows_val = parse_scalar_arg(assign.args[0], "sparse_from_coo");
        if (!rows_val) {
            return std::unexpected(rows_val.error());
        }
        auto cols_val = parse_scalar_arg(assign.args[1], "sparse_from_coo");
        if (!cols_val) {
            return std::unexpected(cols_val.error());
        }
        const int rows_i = static_cast<int>(*rows_val);
        const int cols_i = static_cast<int>(*cols_val);
        if (rows_i < 0 || cols_i < 0 || *rows_val != rows_i || *cols_val != cols_i) {
            return std::unexpected(
                DomainError{"sparse_from_coo", "expected non-negative integer rows and cols"});
        }
        auto row_idx = resolve_operand(assign.args[2]);
        if (!row_idx) {
            return std::unexpected(row_idx.error());
        }
        auto col_idx = resolve_operand(assign.args[3]);
        if (!col_idx) {
            return std::unexpected(col_idx.error());
        }
        auto values = resolve_operand(assign.args[4]);
        if (!values) {
            return std::unexpected(values.error());
        }
        result = eval_sparse_from_coo(static_cast<std::size_t>(rows_i),
                                      static_cast<std::size_t>(cols_i), *row_idx, *col_idx,
                                      *values);
    }

    return result;
}

void ms_register_matrix_call_sparse_from_coo() {
    register_matrix_call("sparse_from_coo", &handle_sparse_from_coo);
}

} // namespace ms::interp
