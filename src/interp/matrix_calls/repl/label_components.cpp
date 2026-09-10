#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_label_components(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "label_components" && assign.args.size() == 1) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto gray = matrix_to_gray_image(*matrix);
        if (!gray) {
            return std::unexpected(gray.error());
        }
        const auto labels = image::label_components(*gray);
        Matrix<double> out(labels.size(), labels.empty() ? 0u : labels[0].size());
        for (size_t r = 0; r < labels.size(); ++r) {
            for (size_t c = 0; c < labels[r].size(); ++c) {
                out(r, c) = static_cast<double>(labels[r][c]);
            }
        }
        result = out;
    }

    return result;
}

void ms_register_matrix_call_label_components() {
    register_matrix_call("label_components", &handle_label_components);
}

} // namespace ms::interp
