// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_quantum_partial_trace(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "quantum_partial_trace" && assign.args.size() == 4) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto d1_val = ctx.parse_scalar_arg(assign.args[1], "quantum_partial_trace");
        if (!d1_val) {
            return std::unexpected(d1_val.error());
        }
        auto d2_val = ctx.parse_scalar_arg(assign.args[2], "quantum_partial_trace");
        if (!d2_val) {
            return std::unexpected(d2_val.error());
        }
        auto sub_val = ctx.parse_scalar_arg(assign.args[3], "quantum_partial_trace");
        if (!sub_val) {
            return std::unexpected(sub_val.error());
        }
        const int d1 = static_cast<int>(*d1_val);
        const int d2 = static_cast<int>(*d2_val);
        const int subsystem = static_cast<int>(*sub_val);
        if (d1 < 1 || d2 < 1 || *d1_val != d1 || *d2_val != d2 ||
            (subsystem != 0 && subsystem != 1) || *sub_val != subsystem) {
            return std::unexpected(DomainError{
                "quantum_partial_trace",
                "expected positive integer d1, d2 and subsystem 0 or 1"});
        }
        result = eval_quantum_partial_trace_matrix(*matrix, d1, d2, subsystem);
    }

    return result;
}

void ms_register_matrix_call_quantum_partial_trace() {
    register_matrix_call("quantum_partial_trace", &handle_quantum_partial_trace);
}

} // namespace ms::interp
