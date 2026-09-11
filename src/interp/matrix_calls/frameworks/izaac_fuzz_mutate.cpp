// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_izaac_fuzz_mutate(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "izaac_fuzz_mutate" &&
               (assign.args.size() == 1 || assign.args.size() == 2)) {
        auto input = ctx.resolve_operand(assign.args[0]);
        if (!input) {
            return std::unexpected(input.error());
        }
        size_t max_edits = 16;
        if (assign.args.size() == 2) {
            auto edits_val = ctx.parse_scalar_arg(assign.args[1], "izaac_fuzz_mutate");
            if (!edits_val) {
                return std::unexpected(edits_val.error());
            }
            const int edits_i = static_cast<int>(*edits_val);
            if (*edits_val != edits_i || edits_i < 0) {
                return std::unexpected(
                    DomainError{"izaac_fuzz_mutate", "expected non-negative integer max_edits"});
            }
            max_edits = static_cast<size_t>(edits_i);
        }
        result = eval_izaac_fuzz_mutate(*input, max_edits);
    }

    return result;
}

void ms_register_matrix_call_izaac_fuzz_mutate() {
    register_matrix_call("izaac_fuzz_mutate", &handle_izaac_fuzz_mutate);
}

} // namespace ms::interp
