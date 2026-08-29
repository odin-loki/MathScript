#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_fem_mesh2d(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if ((assign.callee == "fem_mesh2d_rectangular" || assign.callee == "fem_mesh2d") &&
               assign.args.size() == 6) {
        const char* fn = assign.callee.c_str();
        auto x0 = parse_scalar_arg(assign.args[0], fn);
        if (!x0) {
            return std::unexpected(x0.error());
        }
        auto y0 = parse_scalar_arg(assign.args[1], fn);
        if (!y0) {
            return std::unexpected(y0.error());
        }
        auto x1 = parse_scalar_arg(assign.args[2], fn);
        if (!x1) {
            return std::unexpected(x1.error());
        }
        auto y1 = parse_scalar_arg(assign.args[3], fn);
        if (!y1) {
            return std::unexpected(y1.error());
        }
        auto nx = parse_scalar_arg(assign.args[4], fn);
        if (!nx) {
            return std::unexpected(nx.error());
        }
        auto ny = parse_scalar_arg(assign.args[5], fn);
        if (!ny) {
            return std::unexpected(ny.error());
        }
        auto nx_i = parse_positive_size_arg(*nx, fn, "expected positive integer nx");
        if (!nx_i) {
            return std::unexpected(nx_i.error());
        }
        auto ny_i = parse_positive_size_arg(*ny, fn, "expected positive integer ny");
        if (!ny_i) {
            return std::unexpected(ny_i.error());
        }
        result = eval_fem_mesh2d_rectangular(*x0, *y0, *x1, *y1, *nx_i, *ny_i);
    }

    return result;
}

void ms_register_matrix_call_fem_mesh2d() {
    register_matrix_call("fem_mesh2d", &handle_fem_mesh2d);
}

} // namespace ms::interp
