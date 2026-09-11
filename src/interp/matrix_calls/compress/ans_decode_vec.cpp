// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_ans_decode_vec(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "ans_decode_vec" && assign.args.size() == 2) {
        auto orig = ctx.resolve_operand(assign.args[0]);
        if (!orig) {
            return std::unexpected(orig.error());
        }
        auto encoded = ctx.resolve_operand(assign.args[1]);
        if (!encoded) {
            return std::unexpected(encoded.error());
        }
        result = eval_ans_decode_vec(*orig, *encoded);
    }

    return result;
}

void ms_register_matrix_call_ans_decode_vec() {
    register_matrix_call("ans_decode_vec", &handle_ans_decode_vec);
}

} // namespace ms::interp
