#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_numthy_cornacchia(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "numthy_cornacchia" && assign.args.size() == 2) {
        double d_d = 0.0;
        double p_d = 0.0;
        if (!parse_number(assign.args[0], d_d)) {
            auto d_expr = eval_scalar_expr(ctx.state(), assign.args[0]);
            if (!d_expr) {
                return std::unexpected(
                    DomainError{"numthy_cornacchia", "expected numthy_cornacchia(d,p)"});
            }
            d_d = *d_expr;
        }
        if (!parse_number(assign.args[1], p_d)) {
            auto p_expr = eval_scalar_expr(ctx.state(), assign.args[1]);
            if (!p_expr) {
                return std::unexpected(
                    DomainError{"numthy_cornacchia", "expected numthy_cornacchia(d,p)"});
            }
            p_d = *p_expr;
        }
        if (d_d < 1.0 || p_d < 2.0 || d_d != static_cast<int>(d_d) ||
            p_d != static_cast<int>(p_d) || d_d >= p_d) {
            return std::unexpected(DomainError{
                "numthy_cornacchia", "expected positive integers d,p with 0 < d < p"});
        }
        auto sol = eval_numthy_cornacchia(static_cast<uint64_t>(d_d),
                                          static_cast<uint64_t>(p_d));
        if (!sol) {
            return std::unexpected(sol.error());
        }
        result = *sol;
    }

    return result;
}

void ms_register_matrix_call_numthy_cornacchia() {
    register_matrix_call("numthy_cornacchia", &handle_numthy_cornacchia);
}

} // namespace ms::interp
