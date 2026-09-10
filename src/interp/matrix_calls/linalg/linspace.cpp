#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_linspace(Interpreter& /*interp*/, const MatrixCallAssign& assign) {
    using namespace detail;

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "linspace" && assign.args.size() == 3) {
        double a = 0.0, b = 0.0, n_d = 0.0;
        if (!parse_number(assign.args[0], a) || !parse_number(assign.args[1], b) ||
            !parse_number(assign.args[2], n_d)) {
            return std::unexpected(DomainError{"linspace", "expected linspace(a, b, n)"});
        }
        size_t n = 0;
        size_t one = 0;
        if (!repl_dims_allowed(1.0, n_d, one, n)) {
            return std::unexpected(DomainError{"linspace", kReplMatrixTooLarge});
        }
        const auto vec = linspace(a, b, n);
        Matrix<double> col(vec.size(), 1);
        for (size_t i = 0; i < vec.size(); ++i) {
            col(i, 0) = vec[i];
        }
        result = col;
    }

    return result;
}

void ms_register_matrix_call_linspace() {
    register_matrix_call("linspace", &handle_linspace);
}

} // namespace ms::interp
