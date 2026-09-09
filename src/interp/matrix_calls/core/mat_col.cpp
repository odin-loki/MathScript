#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_mat_col(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);
    auto parse_scalar_arg = [&ctx](const std::string& arg_text, const char* fn) -> Result<double> {
        double value = 0.0;
        if (parse_number(arg_text, value)) return value;
        auto expr = eval_scalar_expr(ctx.state(), arg_text);
        if (!expr) {
            return std::unexpected(DomainError{fn, "expected numeric scalar argument"});
        }
        return *expr;
    };

    if (assign.callee != "mat_col" || assign.args.size() != 2) {
        return std::unexpected(DomainError{"mat_col", "expected mat_col(A, j)"});
    }
    auto A = ctx.resolve_operand(assign.args[0]);
    if (!A) {
        return std::unexpected(A.error());
    }
    auto index = parse_scalar_arg(assign.args[1], "mat_col");
    if (!index) {
        return std::unexpected(index.error());
    }
    return eval_mat_col(*A, *index);
}

void ms_register_matrix_call_mat_col() {
    register_matrix_call("mat_col", &handle_mat_col);
}

} // namespace ms::interp
