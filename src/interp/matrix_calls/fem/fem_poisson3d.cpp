#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_fem_poisson3d(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "fem_poisson3d" && assign.args.size() == 3) {
        double nx_d = 0.0;
        double ny_d = 0.0;
        double nz_d = 0.0;
        if (!parse_number(assign.args[0], nx_d) || !parse_number(assign.args[1], ny_d) ||
            !parse_number(assign.args[2], nz_d)) {
            return std::unexpected(
                DomainError{"fem_poisson3d", "expected fem_poisson3d(nx, ny, nz)"});
        }
        const int nx_i = static_cast<int>(nx_d);
        const int ny_i = static_cast<int>(ny_d);
        const int nz_i = static_cast<int>(nz_d);
        if (nx_i < 0 || ny_i < 0 || nz_i < 0 || nx_d != nx_i || ny_d != ny_i || nz_d != nz_i) {
            return std::unexpected(
                DomainError{"fem_poisson3d", "expected non-negative integer nx, ny, and nz"});
        }
        result = eval_fem_poisson3d(static_cast<std::size_t>(nx_i), static_cast<std::size_t>(ny_i),
                                    static_cast<std::size_t>(nz_i));
    }

    return result;
}

void ms_register_matrix_call_fem_poisson3d() {
    register_matrix_call("fem_poisson3d", &handle_fem_poisson3d);
}

} // namespace ms::interp
