#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_ml_spectral_clustering(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "ml_spectral_clustering" &&
               (assign.args.size() >= 2 && assign.args.size() <= 4)) {
        auto X = ctx.resolve_operand(assign.args[0]);
        if (!X) {
            return std::unexpected(X.error());
        }
        auto k_arg = ctx.parse_scalar_arg(assign.args[1], "ml_spectral_clustering");
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
            auto sigma_arg = ctx.parse_scalar_arg(assign.args[2], "ml_spectral_clustering");
            if (!sigma_arg) {
                return std::unexpected(sigma_arg.error());
            }
            sigma = *sigma_arg;
        }
        if (assign.args.size() >= 4) {
            auto nn_arg = ctx.parse_scalar_arg(assign.args[3], "ml_spectral_clustering");
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
