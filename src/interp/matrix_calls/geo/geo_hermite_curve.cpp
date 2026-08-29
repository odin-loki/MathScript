#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_geo_hermite_curve(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "geo_hermite_curve" && assign.args.size() == 9) {
        std::array<double, 9> args{};
        for (std::size_t i = 0; i < 9; ++i) {
            if (!parse_number(assign.args[i], args[i])) {
                auto expr = eval_scalar_expr(ctx.state(), assign.args[i]);
                if (!expr) {
                    return std::unexpected(DomainError{
                        "geo_hermite_curve",
                        "expected geo_hermite_curve(p0x,p0y,m0x,m0y,p1x,p1y,m1x,m1y,t)"});
                }
                args[i] = *expr;
            }
        }
        result = eval_geo_hermite_curve(args[0], args[1], args[2], args[3], args[4], args[5],
                                        args[6], args[7], args[8]);
    }

    return result;
}

void ms_register_matrix_call_geo_hermite_curve() {
    register_matrix_call("geo_hermite_curve", &handle_geo_hermite_curve);
}

} // namespace ms::interp
