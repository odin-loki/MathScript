#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_zeros(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if ((assign.callee == "zeros" || assign.callee == "eye" || assign.callee == "ones") &&
               (assign.args.size() == 1 || assign.args.size() == 2)) {
        double m_d = 0.0, n_d = 0.0;
        if (!parse_number(assign.args[0], m_d)) {
            // try resolving as scalar variable
            auto it = ctx.state().scalars.find(assign.args[0]);
            if (it != ctx.state().scalars.end()) {
                m_d = it->second;
            } else {
                return std::unexpected(DomainError{assign.callee, "expected numeric size argument"});
            }
        }
        if (assign.args.size() == 2) {
            if (!parse_number(assign.args[1], n_d)) {
                auto it = ctx.state().scalars.find(assign.args[1]);
                if (it != ctx.state().scalars.end()) {
                    n_d = it->second;
                } else {
                    return std::unexpected(DomainError{assign.callee, "expected numeric size argument"});
                }
            }
        } else {
            n_d = m_d;
        }
        size_t rows = 0;
        size_t cols = 0;
        if (!repl_dims_allowed(m_d, n_d, rows, cols)) {
            return std::unexpected(
                DomainError{assign.callee, kReplMatrixTooLarge});
        }
        if (assign.callee == "zeros") {
            result = zeros<double>(rows, cols);
        } else if (assign.callee == "eye") {
            auto I = eye<double>(rows);
            result = I;
        } else {
            result = ones<double>(rows, cols);
        }
    }

    return result;
}

void ms_register_matrix_call_zeros() {
    register_matrix_call("zeros", &handle_zeros);
}

} // namespace ms::interp
