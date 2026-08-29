#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_quantum_ket_basis(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "quantum_ket_basis" && assign.args.size() == 2) {
        double dim_d = 0.0;
        double index_d = 0.0;
        if (!parse_number(assign.args[0], dim_d)) {
            auto it = ctx.state().scalars.find(assign.args[0]);
            if (it != ctx.state().scalars.end()) {
                dim_d = it->second;
            } else {
                return std::unexpected(
                    DomainError{assign.callee, "expected numeric dim argument"});
            }
        }
        if (!parse_number(assign.args[1], index_d)) {
            auto it = ctx.state().scalars.find(assign.args[1]);
            if (it != ctx.state().scalars.end()) {
                index_d = it->second;
            } else {
                return std::unexpected(
                    DomainError{assign.callee, "expected numeric index argument"});
            }
        }
        const int dim = static_cast<int>(dim_d);
        const int index = static_cast<int>(index_d);
        if (dim < 1 || dim_d != dim || index_d != index) {
            return std::unexpected(DomainError{
                assign.callee, "expected positive integer dim and integer index"});
        }
        result = eval_quantum_ket_basis(dim, index);
    }

    return result;
}

void ms_register_matrix_call_quantum_ket_basis() {
    register_matrix_call("quantum_ket_basis", &handle_quantum_ket_basis);
}

} // namespace ms::interp
