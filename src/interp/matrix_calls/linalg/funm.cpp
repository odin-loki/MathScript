// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_funm(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "funm" && assign.args.size() == 2) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        std::string fn_name;
        if (!parse_quoted_string(assign.args[1], fn_name)) {
            return std::unexpected(
                DomainError{"funm", "expected funm(A, \"sin\"|\"cos\"|\"exp\"|\"sqrt\")"});
        }
        fn_name = lower(trim_copy(fn_name));
        if (fn_name == "sin") {
            result = sinm(*matrix);
        } else if (fn_name == "cos") {
            result = cosm(*matrix);
        } else if (fn_name == "exp") {
            result = expm(*matrix);
        } else if (fn_name == "sqrt") {
            result = sqrtm(*matrix);
        } else {
            return std::unexpected(
                DomainError{"funm", "expected \"sin\", \"cos\", \"exp\", or \"sqrt\""});
        }
    }

    return result;
}

void ms_register_matrix_call_funm() {
    register_matrix_call("funm", &handle_funm);
}

} // namespace ms::interp
