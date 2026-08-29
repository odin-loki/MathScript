#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_graph_min_arborescence(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "graph_min_arborescence" && assign.args.size() == 2) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        double root_d = 0.0;
        if (!parse_number(assign.args[1], root_d)) {
            auto root_expr = eval_scalar_expr(ctx.state(), assign.args[1]);
            if (!root_expr) {
                return std::unexpected(DomainError{
                    "graph_min_arborescence", "expected graph_min_arborescence(A, root)"});
            }
            root_d = *root_expr;
        }
        const int root = static_cast<int>(root_d);
        if (root < 0 || root_d != root) {
            return std::unexpected(
                DomainError{"graph_min_arborescence", "expected non-negative integer root"});
        }
        auto arb = eval_graph_min_arborescence(*matrix, root);
        if (!arb) {
            return std::unexpected(arb.error());
        }
        result = *arb;
    }

    return result;
}

void ms_register_matrix_call_graph_min_arborescence() {
    register_matrix_call("graph_min_arborescence", &handle_graph_min_arborescence);
}

} // namespace ms::interp
