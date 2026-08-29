#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_graph_bellman_ford(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "graph_bellman_ford" && assign.args.size() == 2) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        double source_d = 0.0;
        if (!parse_number(assign.args[1], source_d)) {
            auto source_expr = eval_scalar_expr(ctx.state(), assign.args[1]);
            if (!source_expr) {
                return std::unexpected(DomainError{
                    "graph_bellman_ford", "expected graph_bellman_ford(A, source)"});
            }
            source_d = *source_expr;
        }
        const int source = static_cast<int>(source_d);
        if (source < 0 || source_d != source) {
            return std::unexpected(DomainError{
                "graph_bellman_ford", "expected non-negative integer source"});
        }
        auto sp = eval_graph_bellman_ford(*matrix, source);
        if (!sp) {
            return std::unexpected(sp.error());
        }
        result = *sp;
    }

    return result;
}

void ms_register_matrix_call_graph_bellman_ford() {
    register_matrix_call("graph_bellman_ford", &handle_graph_bellman_ford);
}

} // namespace ms::interp
