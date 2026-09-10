#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_ml_knn_fit(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "ml_knn_fit" && assign.args.size() == 3) {
        auto X = ctx.resolve_operand(assign.args[0]);
        if (!X) {
            return std::unexpected(X.error());
        }
        auto y = ctx.resolve_operand(assign.args[1]);
        if (!y) {
            return std::unexpected(y.error());
        }
        auto k_val = ctx.parse_scalar_arg(assign.args[2], "ml_knn_fit");
        if (!k_val) {
            return std::unexpected(k_val.error());
        }
        const int k = static_cast<int>(*k_val);
        if (k < 1 || *k_val != k) {
            return std::unexpected(DomainError{"ml_knn_fit", "expected positive integer k"});
        }
        auto fitted = eval_ml_knn_fit(*X, *y, k);
        if (!fitted) {
            return std::unexpected(fitted.error());
        }
        result = *fitted;
    }

    return result;
}

void ms_register_matrix_call_ml_knn_fit() {
    register_matrix_call("ml_knn_fit", &handle_ml_knn_fit);
}

} // namespace ms::interp
