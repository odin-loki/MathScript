#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_numthy_lucas_sequence(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "numthy_lucas_sequence" && assign.args.size() == 3) {
        double k_d = 0.0;
        double P_d = 0.0;
        double Q_d = 0.0;
        if (!parse_number(assign.args[0], k_d)) {
            auto k_expr = eval_scalar_expr(ctx.state(), assign.args[0]);
            if (!k_expr) {
                return std::unexpected(DomainError{
                    "numthy_lucas_sequence", "expected numthy_lucas_sequence(k,P,Q)"});
            }
            k_d = *k_expr;
        }
        if (!parse_number(assign.args[1], P_d)) {
            auto P_expr = eval_scalar_expr(ctx.state(), assign.args[1]);
            if (!P_expr) {
                return std::unexpected(DomainError{
                    "numthy_lucas_sequence", "expected numthy_lucas_sequence(k,P,Q)"});
            }
            P_d = *P_expr;
        }
        if (!parse_number(assign.args[2], Q_d)) {
            auto Q_expr = eval_scalar_expr(ctx.state(), assign.args[2]);
            if (!Q_expr) {
                return std::unexpected(DomainError{
                    "numthy_lucas_sequence", "expected numthy_lucas_sequence(k,P,Q)"});
            }
            Q_d = *Q_expr;
        }
        if (std::floor(k_d) != k_d || std::floor(P_d) != P_d || std::floor(Q_d) != Q_d) {
            return std::unexpected(
                DomainError{"numthy_lucas_sequence", "expected integer arguments"});
        }
        result = eval_numthy_lucas_sequence(static_cast<int64_t>(k_d),
                                           static_cast<int64_t>(P_d),
                                           static_cast<int64_t>(Q_d));
    }

    return result;
}

void ms_register_matrix_call_numthy_lucas_sequence() {
    register_matrix_call("numthy_lucas_sequence", &handle_numthy_lucas_sequence);
}

} // namespace ms::interp
