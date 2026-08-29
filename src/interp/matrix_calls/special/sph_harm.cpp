#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_sph_harm(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "sph_harm" && assign.args.size() == 4) {
        std::array<double, 4> args{};
        for (std::size_t i = 0; i < 4; ++i) {
            if (!parse_number(assign.args[i], args[i])) {
                auto expr = eval_scalar_expr(ctx.state(), assign.args[i]);
                if (!expr) {
                    return std::unexpected(DomainError{
                        "sph_harm", "expected sph_harm(l,m,theta,phi)"});
                }
                args[i] = *expr;
            }
        }
        const int l = static_cast<int>(args[0]);
        const int m = static_cast<int>(args[1]);
        if (args[0] != l || args[1] != m) {
            return std::unexpected(
                DomainError{"sph_harm", "expected integer l and m"});
        }
        result = eval_sph_harm(l, m, args[2], args[3]);
    }

    return result;
}

void ms_register_matrix_call_sph_harm() {
    register_matrix_call("sph_harm", &handle_sph_harm);
}

} // namespace ms::interp
