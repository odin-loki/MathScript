#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_golomb_rice_encode_vec(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "golomb_rice_encode_vec" && assign.args.size() == 2) {
        auto values = ctx.resolve_operand(assign.args[0]);
        if (!values) {
            return std::unexpected(values.error());
        }
        auto m_bits_val = ctx.parse_scalar_arg(assign.args[1], "golomb_rice_encode_vec");
        if (!m_bits_val) {
            return std::unexpected(m_bits_val.error());
        }
        const int m_bits = static_cast<int>(*m_bits_val);
        if (*m_bits_val != m_bits || m_bits < 0) {
            return std::unexpected(
                DomainError{"golomb_rice_encode_vec", "expected non-negative integer m_bits"});
        }
        auto encoded = eval_golomb_rice_encode_vec(*values, m_bits);
        if (!encoded) {
            return std::unexpected(encoded.error());
        }
        result = *encoded;
    }

    return result;
}

void ms_register_matrix_call_golomb_rice_encode_vec() {
    register_matrix_call("golomb_rice_encode_vec", &handle_golomb_rice_encode_vec);
}

} // namespace ms::interp
