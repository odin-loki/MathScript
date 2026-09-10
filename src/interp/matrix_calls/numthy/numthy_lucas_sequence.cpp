#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_numthy_lucas_sequence(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "numthy_lucas_sequence" && assign.args.size() == 3) {
        double k_d = 0.0;
        double P_d = 0.0;
        double Q_d = 0.0;
        if (!parse_number(assign.args[0], k_d)) {
            auto k_expr = eval_scalar_expr(ctx.state(), assign.args[0]);
            if (!k_expr) {
                return std::unexpected(DomainError{
                    "numthy_lucas_sequence", "expected numthy_lucas_sequence(k,P,Q)"});
            }
            k_d = *k_expr;
        }
        if (!parse_number(assign.args[1], P_d)) {
            auto P_expr = eval_scalar_expr(ctx.state(), assign.args[1]);
            if (!P_expr) {
                return std::unexpected(DomainError{
                    "numthy_lucas_sequence", "expected numthy_lucas_sequence(k,P,Q)"});
            }
            P_d = *P_expr;
        }
        if (!parse_number(assign.args[2], Q_d)) {
            auto Q_expr = eval_scalar_expr(ctx.state(), assign.args[2]);
            if (!Q_expr) {
                return std::unexpected(DomainError{
                    "numthy_lucas_sequence", "expected numthy_lucas_sequence(k,P,Q)"});
            }
            Q_d = *Q_expr;
        }
        if (std::floor(k_d) != k_d || std::floor(P_d) != P_d || std::floor(Q_d) != Q_d) {
            return std::unexpected(
                DomainError{"numthy_lucas_sequence", "expected integer arguments"});
        }
        result = eval_numthy_lucas_sequence(static_cast<int64_t>(k_d),
                                           static_cast<int64_t>(P_d),
                                           static_cast<int64_t>(Q_d));
    }

    return result;
}

void ms_register_matrix_call_numthy_lucas_sequence() {
    register_matrix_call("numthy_lucas_sequence", &handle_numthy_lucas_sequence);
}

} // namespace ms::interp
