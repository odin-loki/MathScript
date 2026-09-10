#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_ml_pca_transform(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "ml_pca_transform" && assign.args.size() == 2) {
        auto X = ctx.resolve_operand(assign.args[0]);
        if (!X) {
            return std::unexpected(X.error());
        }
        auto model = ctx.resolve_operand(assign.args[1]);
        if (!model) {
            return std::unexpected(model.error());
        }
        auto transformed = eval_ml_pca_transform(*X, *model);
        if (!transformed) {
            return std::unexpected(transformed.error());
        }
        result = *transformed;
    }

    return result;
}

void ms_register_matrix_call_ml_pca_transform() {
    register_matrix_call("ml_pca_transform", &handle_ml_pca_transform);
}

} // namespace ms::interp
