#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_ml_spectral_clustering(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "ml_spectral_clustering" &&
               (assign.args.size() >= 2 && assign.args.size() <= 4)) {
        auto X = resolve_operand(assign.args[0]);
        if (!X) {
            return std::unexpected(X.error());
        }
        auto k_arg = parse_scalar_arg(assign.args[1], "ml_spectral_clustering");
        if (!k_arg) {
            return std::unexpected(k_arg.error());
        }
        const int k = static_cast<int>(*k_arg);
        if (k < 1 || *k_arg != k) {
            return std::unexpected(
                DomainError{"ml_spectral_clustering", "expected integer k >= 1"});
        }
        double sigma = 1.0;
        int n_neighbors = 0;
        if (assign.args.size() >= 3) {
            auto sigma_arg = parse_scalar_arg(assign.args[2], "ml_spectral_clustering");
            if (!sigma_arg) {
                return std::unexpected(sigma_arg.error());
            }
            sigma = *sigma_arg;
        }
        if (assign.args.size() >= 4) {
            auto nn_arg = parse_scalar_arg(assign.args[3], "ml_spectral_clustering");
            if (!nn_arg) {
                return std::unexpected(nn_arg.error());
            }
            n_neighbors = static_cast<int>(*nn_arg);
            if (*nn_arg != n_neighbors) {
                return std::unexpected(
                    DomainError{"ml_spectral_clustering", "expected integer n_neighbors"});
            }
        }
        auto labels = eval_ml_spectral_clustering(*X, k, sigma, n_neighbors);
        if (!labels) {
            return std::unexpected(labels.error());
        }
        result = *labels;
    }

    return result;
}

void ms_register_matrix_call_ml_spectral_clustering() {
    register_matrix_call("ml_spectral_clustering", &handle_ml_spectral_clustering);
}

} // namespace ms::interp
