#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_combo_unrank_combination(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "combo_unrank_combination" && assign.args.size() == 3) {
        double n_d = 0.0;
        double k_d = 0.0;
        double rank_d = 0.0;
        if (!parse_number(assign.args[0], n_d)) {
            auto it = ctx.state().scalars.find(assign.args[0]);
            if (it != ctx.state().scalars.end()) {
                n_d = it->second;
            } else {
                return std::unexpected(
                    DomainError{assign.callee, "expected numeric n argument"});
            }
        }
        if (!parse_number(assign.args[1], k_d)) {
            auto it = ctx.state().scalars.find(assign.args[1]);
            if (it != ctx.state().scalars.end()) {
                k_d = it->second;
            } else {
                return std::unexpected(
                    DomainError{assign.callee, "expected numeric k argument"});
            }
        }
        if (!parse_number(assign.args[2], rank_d)) {
            auto it = ctx.state().scalars.find(assign.args[2]);
            if (it != ctx.state().scalars.end()) {
                rank_d = it->second;
            } else {
                return std::unexpected(
                    DomainError{assign.callee, "expected numeric rank argument"});
            }
        }
        const int n = static_cast<int>(n_d);
        const int k = static_cast<int>(k_d);
        if (n < 0 || k < 0 || n_d != n || k_d != k || rank_d < 0.0 ||
            std::floor(rank_d) != rank_d) {
            return std::unexpected(DomainError{
                assign.callee, "expected non-negative integer n, k and rank"});
        }
        auto comb = eval_combo_unrank_combination(n, k, static_cast<uint64_t>(rank_d));
        if (!comb) {
            return std::unexpected(comb.error());
        }
        result = *comb;
    }

    return result;
}

void ms_register_matrix_call_combo_unrank_combination() {
    register_matrix_call("combo_unrank_combination", &handle_combo_unrank_combination);
}

} // namespace ms::interp
