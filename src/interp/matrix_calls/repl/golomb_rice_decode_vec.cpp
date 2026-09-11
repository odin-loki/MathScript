// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_golomb_rice_decode_vec(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "golomb_rice_decode_vec" && assign.args.size() == 3) {
        auto encoded = ctx.resolve_operand(assign.args[0]);
        if (!encoded) {
            return std::unexpected(encoded.error());
        }
        auto m_bits_val = ctx.parse_scalar_arg(assign.args[1], "golomb_rice_decode_vec");
        if (!m_bits_val) {
            return std::unexpected(m_bits_val.error());
        }
        auto count_val = ctx.parse_scalar_arg(assign.args[2], "golomb_rice_decode_vec");
        if (!count_val) {
            return std::unexpected(count_val.error());
        }
        const int m_bits = static_cast<int>(*m_bits_val);
        if (*m_bits_val != m_bits || m_bits < 0) {
            return std::unexpected(
                DomainError{"golomb_rice_decode_vec", "expected non-negative integer m_bits"});
        }
        if (*count_val < 0.0 || std::floor(*count_val) != *count_val) {
            return std::unexpected(
                DomainError{"golomb_rice_decode_vec", "expected non-negative integer count"});
        }
        const size_t count = static_cast<size_t>(*count_val);
        auto decoded = eval_golomb_rice_decode_vec(*encoded, m_bits, count);
        if (!decoded) {
            return std::unexpected(decoded.error());
        }
        result = *decoded;
    }

    return result;
}

void ms_register_matrix_call_golomb_rice_decode_vec() {
    register_matrix_call("golomb_rice_decode_vec", &handle_golomb_rice_decode_vec);
}

} // namespace ms::interp
