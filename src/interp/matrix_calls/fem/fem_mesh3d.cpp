#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_fem_mesh3d(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if ((assign.callee == "fem_mesh3d_box" || assign.callee == "fem_mesh3d") &&
               assign.args.size() == 9) {
        const char* fn = assign.callee.c_str();
        std::array<Result<double>, 9> scalars{};
        for (size_t i = 0; i < 9; ++i) {
            scalars[i] = parse_scalar_arg(assign.args[i], fn);
            if (!scalars[i]) {
                return std::unexpected(scalars[i].error());
            }
        }
        auto nx_i = parse_positive_size_arg(*scalars[6], fn, "expected positive integer nx");
        if (!nx_i) {
            return std::unexpected(nx_i.error());
        }
        auto ny_i = parse_positive_size_arg(*scalars[7], fn, "expected positive integer ny");
        if (!ny_i) {
            return std::unexpected(ny_i.error());
        }
        auto nz_i = parse_positive_size_arg(*scalars[8], fn, "expected positive integer nz");
        if (!nz_i) {
            return std::unexpected(nz_i.error());
        }
        result = eval_fem_mesh3d_box(*scalars[0], *scalars[1], *scalars[2], *scalars[3],
                                     *scalars[4], *scalars[5], *nx_i, *ny_i, *nz_i);
    }

    return result;
}

void ms_register_matrix_call_fem_mesh3d() {
    register_matrix_call("fem_mesh3d", &handle_fem_mesh3d);
}

} // namespace ms::interp
