#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_fem_mesh1d(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "fem_mesh1d" && assign.args.size() == 3) {
        auto a = parse_scalar_arg(assign.args[0], "fem_mesh1d");
        if (!a) {
            return std::unexpected(a.error());
        }
        auto b = parse_scalar_arg(assign.args[1], "fem_mesh1d");
        if (!b) {
            return std::unexpected(b.error());
        }
        auto n_val = parse_scalar_arg(assign.args[2], "fem_mesh1d");
        if (!n_val) {
            return std::unexpected(n_val.error());
        }
        auto n_i = parse_positive_size_arg(*n_val, "fem_mesh1d", "expected positive integer n_elements");
        if (!n_i) {
            return std::unexpected(n_i.error());
        }
        result = eval_fem_mesh1d(*a, *b, *n_i);
    }

    return result;
}

void ms_register_matrix_call_fem_mesh1d() {
    register_matrix_call("fem_mesh1d", &handle_fem_mesh1d);
}

} // namespace ms::interp
