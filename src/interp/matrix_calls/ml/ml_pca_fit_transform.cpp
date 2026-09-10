#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_ml_pca_fit_transform(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "ml_pca_fit_transform" && assign.args.size() == 2) {
        auto X = ctx.resolve_operand(assign.args[0]);
        if (!X) {
            return std::unexpected(X.error());
        }
        auto n_comp = ctx.parse_scalar_arg(assign.args[1], "ml_pca_fit_transform");
        if (!n_comp) {
            return std::unexpected(n_comp.error());
        }
        const int n_components = static_cast<int>(*n_comp);
        if (*n_comp != n_components) {
            return std::unexpected(
                DomainError{"ml_pca_fit_transform", "expected integer n_components"});
        }
        auto transformed = eval_ml_pca_fit_transform(*X, n_components);
        if (!transformed) {
            return std::unexpected(transformed.error());
        }
        result = *transformed;
    }

    return result;
}

void ms_register_matrix_call_ml_pca_fit_transform() {
    register_matrix_call("ml_pca_fit_transform", &handle_ml_pca_fit_transform);
}

} // namespace ms::interp
