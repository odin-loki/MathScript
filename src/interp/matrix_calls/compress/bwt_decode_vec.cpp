// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_bwt_decode_vec(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "bwt_decode_vec" && assign.args.size() == 2) {
        auto l = ctx.resolve_operand(assign.args[0]);
        if (!l) {
            return std::unexpected(l.error());
        }
        auto pi = ctx.parse_scalar_arg(assign.args[1], "bwt_decode_vec");
        if (!pi) {
            return std::unexpected(pi.error());
        }
        result = eval_bwt_decode_vec(*l, *pi);
    }

    return result;
}

void ms_register_matrix_call_bwt_decode_vec() {
    register_matrix_call("bwt_decode_vec", &handle_bwt_decode_vec);
}

} // namespace ms::interp
