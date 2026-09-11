// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_ml_svm_fit(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "ml_svm_fit" &&
               (assign.args.size() == 2 || assign.args.size() == 3 || assign.args.size() == 4)) {
        auto X = ctx.resolve_operand(assign.args[0]);
        if (!X) {
            return std::unexpected(X.error());
        }
        auto y = ctx.resolve_operand(assign.args[1]);
        if (!y) {
            return std::unexpected(y.error());
        }
        double C = 1.0;
        double gamma = 0.1;
        bool use_rbf = false;
        if (assign.args.size() >= 3) {
            auto c_val = ctx.parse_scalar_arg(assign.args[2], "ml_svm_fit");
            if (!c_val) {
                return std::unexpected(c_val.error());
            }
            C = *c_val;
        }
        if (assign.args.size() >= 4) {
            auto gamma_val = ctx.parse_scalar_arg(assign.args[3], "ml_svm_fit");
            if (!gamma_val) {
                return std::unexpected(gamma_val.error());
            }
            gamma = *gamma_val;
            use_rbf = true;
        }
        auto fitted = eval_ml_svm_fit(*X, *y, C, gamma, use_rbf);
        if (!fitted) {
            return std::unexpected(fitted.error());
        }
        result = *fitted;
    }

    return result;
}

void ms_register_matrix_call_ml_svm_fit() {
    register_matrix_call("ml_svm_fit", &handle_ml_svm_fit);
}

} // namespace ms::interp
