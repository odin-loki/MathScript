// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_izaac_encrypt(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "izaac_encrypt" && assign.args.size() == 2) {
        auto plain = ctx.resolve_operand(assign.args[0]);
        if (!plain) {
            return std::unexpected(plain.error());
        }
        auto key = ctx.resolve_operand(assign.args[1]);
        if (!key) {
            return std::unexpected(key.error());
        }
        result = eval_izaac_encrypt(*plain, *key);
    }

    return result;
}

void ms_register_matrix_call_izaac_encrypt() {
    register_matrix_call("izaac_encrypt", &handle_izaac_encrypt);
}

} // namespace ms::interp
