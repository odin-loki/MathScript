#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_cellai_hebbian_update(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "cellai_hebbian_update" && assign.args.size() == 4) {
        auto w = resolve_operand(assign.args[0]);
        if (!w) {
            return std::unexpected(w.error());
        }
        auto x = resolve_operand(assign.args[1]);
        if (!x) {
            return std::unexpected(x.error());
        }
        auto y = resolve_operand(assign.args[2]);
        if (!y) {
            return std::unexpected(y.error());
        }
        auto lr = parse_scalar_arg(assign.args[3], "cellai_hebbian_update");
        if (!lr) {
            return std::unexpected(lr.error());
        }
        result = eval_cellai_hebbian_update(*w, *x, *y, *lr);
    }

    return result;
}

void ms_register_matrix_call_cellai_hebbian_update() {
    register_matrix_call("cellai_hebbian_update", &handle_cellai_hebbian_update);
}

} // namespace ms::interp
