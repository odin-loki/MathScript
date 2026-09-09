#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_mat_submatrix(Interpreter& interp, const MatrixCallAssign& assign) {
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

    if (assign.callee != "mat_submatrix" || assign.args.size() != 5) {
        return std::unexpected(
            DomainError{"mat_submatrix", "expected mat_submatrix(A, r0, c0, rows, cols)"});
    }
    auto A = ctx.resolve_operand(assign.args[0]);
    if (!A) {
        return std::unexpected(A.error());
    }
    double values[4] = {0.0, 0.0, 0.0, 0.0};
    for (size_t i = 0; i < 4; ++i) {
        auto parsed = parse_scalar_arg(assign.args[i + 1], "mat_submatrix");
        if (!parsed) {
            return std::unexpected(parsed.error());
        }
        values[i] = *parsed;
    }
    return eval_mat_submatrix(*A, values[0], values[1], values[2], values[3]);
}

void ms_register_matrix_call_mat_submatrix() {
    register_matrix_call("mat_submatrix", &handle_mat_submatrix);
}

} // namespace ms::interp
