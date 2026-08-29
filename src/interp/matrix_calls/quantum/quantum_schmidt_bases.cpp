#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_quantum_schmidt_bases(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "quantum_schmidt_bases" && assign.args.size() == 3) {
        auto psi = resolve_operand(assign.args[0]);
        if (!psi) {
            return std::unexpected(psi.error());
        }
        auto dim_a_val = parse_scalar_arg(assign.args[1], "quantum_schmidt_bases");
        if (!dim_a_val) {
            return std::unexpected(dim_a_val.error());
        }
        auto dim_b_val = parse_scalar_arg(assign.args[2], "quantum_schmidt_bases");
        if (!dim_b_val) {
            return std::unexpected(dim_b_val.error());
        }
        const int dim_a = static_cast<int>(*dim_a_val);
        const int dim_b = static_cast<int>(*dim_b_val);
        if (dim_a < 1 || dim_b < 1 || *dim_a_val != dim_a || *dim_b_val != dim_b) {
            return std::unexpected(DomainError{
                "quantum_schmidt_bases", "expected positive integer dim_a and dim_b"});
        }
        result = eval_quantum_schmidt_bases(*psi, dim_a, dim_b);
    }

    return result;
}

void ms_register_matrix_call_quantum_schmidt_bases() {
    register_matrix_call("quantum_schmidt_bases", &handle_quantum_schmidt_bases);
}

} // namespace ms::interp
