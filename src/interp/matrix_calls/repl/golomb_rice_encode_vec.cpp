#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_golomb_rice_encode_vec(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);
    auto resolve_operand = [&ctx](const std::string& text) { return ctx.resolve_operand(text); };
    auto parse_scalar_arg = [&ctx](const std::string& arg_text, const char* fn) -> Result<double> {
        double value = 0.0;
        if (parse_number(arg_text, value)) return value;
        auto expr = eval_scalar_expr(ctx.state(), arg_text);
        if (!expr) {
            return std::unexpected(DomainError{fn, "expected numeric scalar argument"});
        }
        return *expr;
    };
    auto parse_positive_size_arg = [](double value, const char* fn, const char* label) -> Result<std::size_t> {
        const int i = static_cast<int>(value);
        if (i < 1 || value != static_cast<double>(i)) {
            return std::unexpected(DomainError{fn, label});
        }
        return static_cast<std::size_t>(i);
    };
    auto parse_uint64_arg = [](double value, const char* fn, const char* label) -> Result<uint64_t> {
        if (value < 0.0 || value != std::floor(value)) {
            return std::unexpected(DomainError{fn, label});
        }
        return static_cast<uint64_t>(value);
    };

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "golomb_rice_encode_vec" && assign.args.size() == 2) {
        auto values = resolve_operand(assign.args[0]);
        if (!values) {
            return std::unexpected(values.error());
        }
        auto m_bits_val = parse_scalar_arg(assign.args[1], "golomb_rice_encode_vec");
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
