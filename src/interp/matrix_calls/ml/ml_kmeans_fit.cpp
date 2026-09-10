#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_ml_kmeans_fit(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "ml_kmeans_fit" && assign.args.size() == 2) {
        auto X = ctx.resolve_operand(assign.args[0]);
        if (!X) {
            return std::unexpected(X.error());
        }
        auto k_arg = ctx.parse_scalar_arg(assign.args[1], "ml_kmeans_fit");
        if (!k_arg) {
            return std::unexpected(k_arg.error());
        }
        const int k = static_cast<int>(*k_arg);
        if (*k_arg != k) {
            return std::unexpected(DomainError{"ml_kmeans_fit", "expected integer k"});
        }
        auto fitted = eval_ml_kmeans_fit(*X, k);
        if (!fitted) {
            return std::unexpected(fitted.error());
        }
        result = *fitted;
    }

    return result;
}

void ms_register_matrix_call_ml_kmeans_fit() {
    register_matrix_call("ml_kmeans_fit", &handle_ml_kmeans_fit);
}

} // namespace ms::interp
