#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_crypto_bytes_to_hex(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "crypto_bytes_to_hex" && assign.args.size() == 1) {
        auto bytes = ctx.resolve_operand(assign.args[0]);
        if (!bytes) {
            return std::unexpected(bytes.error());
        }
        result = eval_crypto_bytes_to_hex_vec(*bytes);
    }

    return result;
}

void ms_register_matrix_call_crypto_bytes_to_hex() {
    register_matrix_call("crypto_bytes_to_hex", &handle_crypto_bytes_to_hex);
}

} // namespace ms::interp
