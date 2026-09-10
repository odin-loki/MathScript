#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_crypto_from_hex(Interpreter& /*interp*/, const MatrixCallAssign& assign) {
    using namespace detail;

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "crypto_from_hex" && assign.args.size() == 1) {
        std::string hex;
        if (!parse_quoted_string(assign.args[0], hex)) {
            hex = trim_copy(assign.args[0]);
        }
        result = eval_crypto_from_hex(hex);
    }

    return result;
}

void ms_register_matrix_call_crypto_from_hex() {
    register_matrix_call("crypto_from_hex", &handle_crypto_from_hex);
}

} // namespace ms::interp
