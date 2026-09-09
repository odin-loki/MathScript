#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_mat_reshape(Interpreter& interp, const MatrixCallAssign& assign) {
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

    if (assign.callee != "mat_reshape" || assign.args.size() != 3) {
        return std::unexpected(DomainError{"mat_reshape", "expected mat_reshape(A, rows, cols)"});
    }
    auto A = ctx.resolve_operand(assign.args[0]);
    if (!A) {
        return std::unexpected(A.error());
    }
    auto rows = parse_scalar_arg(assign.args[1], "mat_reshape");
    if (!rows) {
        return std::unexpected(rows.error());
    }
    auto cols = parse_scalar_arg(assign.args[2], "mat_reshape");
    if (!cols) {
        return std::unexpected(cols.error());
    }
    return eval_mat_reshape(*A, *rows, *cols);
}

void ms_register_matrix_call_mat_reshape() {
    register_matrix_call("mat_reshape", &handle_mat_reshape);
}

} // namespace ms::interp
