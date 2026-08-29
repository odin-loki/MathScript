#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_poly_cheb_expand(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "poly_cheb_expand" && assign.args.size() == 4) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        double n_d = 0.0;
        if (!parse_number(assign.args[1], n_d)) {
            auto n_expr = eval_scalar_expr(ctx.state(), assign.args[1]);
            if (!n_expr) {
                return std::unexpected(
                    DomainError{"poly_cheb_expand", "expected non-negative integer n"});
            }
            n_d = *n_expr;
        }
        const int n = static_cast<int>(n_d);
        if (n < 0 || n_d != n) {
            return std::unexpected(
                DomainError{"poly_cheb_expand", "expected non-negative integer n"});
        }
        auto a = parse_scalar_arg(assign.args[2], "poly_cheb_expand");
        if (!a) {
            return std::unexpected(a.error());
        }
        auto b = parse_scalar_arg(assign.args[3], "poly_cheb_expand");
        if (!b) {
            return std::unexpected(b.error());
        }
        auto cheb = eval_poly_cheb_expand(*matrix, n, *a, *b);
        if (!cheb) {
            return std::unexpected(cheb.error());
        }
        result = *cheb;
    }

    return result;
}

void ms_register_matrix_call_poly_cheb_expand() {
    register_matrix_call("poly_cheb_expand", &handle_poly_cheb_expand);
}

} // namespace ms::interp
