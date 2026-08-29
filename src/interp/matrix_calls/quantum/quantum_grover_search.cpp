#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_quantum_grover_search(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "quantum_grover_search" &&
               (assign.args.size() == 2 || assign.args.size() == 3)) {
        auto n_qubits_val = parse_scalar_arg(assign.args[0], "quantum_grover_search");
        if (!n_qubits_val) {
            return std::unexpected(n_qubits_val.error());
        }
        const int n_qubits = static_cast<int>(*n_qubits_val);
        if (n_qubits < 1 || *n_qubits_val != n_qubits) {
            return std::unexpected(DomainError{
                "quantum_grover_search", "expected positive integer n_qubits"});
        }
        auto marked_m = resolve_operand(assign.args[1]);
        if (!marked_m) {
            return std::unexpected(marked_m.error());
        }
        auto marked_indices = matrix_to_int_coeff_vector(*marked_m, "quantum_grover_search");
        if (!marked_indices) {
            return std::unexpected(marked_indices.error());
        }
        int n_iterations = quantum::grover_optimal_iterations(
            n_qubits, static_cast<int>(marked_indices->size()));
        if (assign.args.size() == 3) {
            auto iter_val = parse_scalar_arg(assign.args[2], "quantum_grover_search");
            if (!iter_val) {
                return std::unexpected(iter_val.error());
            }
            n_iterations = static_cast<int>(*iter_val);
            if (n_iterations < 0 || *iter_val != n_iterations) {
                return std::unexpected(DomainError{
                    "quantum_grover_search", "expected non-negative integer n_iterations"});
            }
        }
        result = eval_quantum_grover_search(n_qubits, *marked_m, n_iterations);
    }

    return result;
}

void ms_register_matrix_call_quantum_grover_search() {
    register_matrix_call("quantum_grover_search", &handle_quantum_grover_search);
}

} // namespace ms::interp
