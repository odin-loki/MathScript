#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_shi_tomasi(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "shi_tomasi" &&
               (assign.args.size() == 2 || assign.args.size() == 3)) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        double n_d = 0.0;
        double quality = 0.01;
        if (!parse_number(assign.args[1], n_d)) {
            return std::unexpected(
                DomainError{"shi_tomasi", "expected shi_tomasi(M, n) or shi_tomasi(M, n, quality)"});
        }
        if (assign.args.size() == 3 && !parse_number(assign.args[2], quality)) {
            return std::unexpected(
                DomainError{"shi_tomasi", "expected shi_tomasi(M, n) or shi_tomasi(M, n, quality)"});
        }
        const int n = static_cast<int>(n_d);
        if (n < 1 || n_d != n) {
            return std::unexpected(DomainError{"shi_tomasi", "expected positive integer n"});
        }
        result = eval_shi_tomasi(*matrix, n, static_cast<float>(quality));
    }

    return result;
}

void ms_register_matrix_call_shi_tomasi() {
    register_matrix_call("shi_tomasi", &handle_shi_tomasi);
}

} // namespace ms::interp
