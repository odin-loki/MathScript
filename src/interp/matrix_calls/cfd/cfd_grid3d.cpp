#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_cfd_grid3d(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "cfd_grid3d" && assign.args.size() == 9) {
        const char* fn = "cfd_grid3d";
        std::array<Result<double>, 9> scalars{};
        for (size_t i = 0; i < 9; ++i) {
            scalars[i] = parse_scalar_arg(assign.args[i], fn);
            if (!scalars[i]) {
                return std::unexpected(scalars[i].error());
            }
        }
        const int nx_i = static_cast<int>(*scalars[6]);
        const int ny_i = static_cast<int>(*scalars[7]);
        const int nz_i = static_cast<int>(*scalars[8]);
        if (nx_i < 2 || ny_i < 2 || nz_i < 2 || *scalars[6] != nx_i || *scalars[7] != ny_i ||
            *scalars[8] != nz_i) {
            return std::unexpected(
                DomainError{fn, "expected non-negative integer nx, ny, nz >= 2"});
        }
        result = eval_cfd_grid3d(*scalars[0], *scalars[1], *scalars[2], *scalars[3], *scalars[4],
                                 *scalars[5], static_cast<std::size_t>(nx_i),
                                 static_cast<std::size_t>(ny_i), static_cast<std::size_t>(nz_i));
    }

    return result;
}

void ms_register_matrix_call_cfd_grid3d() {
    register_matrix_call("cfd_grid3d", &handle_cfd_grid3d);
}

} // namespace ms::interp
