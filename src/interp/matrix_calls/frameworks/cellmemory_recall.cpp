#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_cellmemory_recall(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "cellmemory_recall" && assign.args.size() == 2) {
        std::string handle = trim_copy(assign.args[0]);
        if (!is_identifier(handle)) {
            return std::unexpected(DomainError{
                "cellmemory_recall", "expected CellMemory handle identifier"});
        }
        const auto it = ctx.session_objects().find(handle);
        if (it == ctx.session_objects().end()) {
            return std::unexpected(DomainError{
                "cellmemory_recall", "session object not found: " + handle});
        }
        if (!std::holds_alternative<cellai::CellMemory>(it->second)) {
            return std::unexpected(DomainError{
                "cellmemory_recall",
                std::string("session object '") + handle + "' is not a CellMemory"});
        }
        auto time_scale = parse_scalar_arg(assign.args[1], "cellmemory_recall");
        if (!time_scale) {
            return std::unexpected(time_scale.error());
        }
        auto recalled = std::get<cellai::CellMemory>(it->second).recall(*time_scale);
        if (!recalled) {
            return std::unexpected(recalled.error());
        }
        result = *recalled;
    }

    return result;
}

void ms_register_matrix_call_cellmemory_recall() {
    register_matrix_call("cellmemory_recall", &handle_cellmemory_recall);
}

} // namespace ms::interp
