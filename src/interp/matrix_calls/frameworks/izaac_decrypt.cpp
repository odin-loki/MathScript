#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_izaac_decrypt(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "izaac_decrypt" && assign.args.size() == 2) {
        auto ct = ctx.resolve_operand(assign.args[0]);
        if (!ct) {
            return std::unexpected(ct.error());
        }
        auto key = ctx.resolve_operand(assign.args[1]);
        if (!key) {
            return std::unexpected(key.error());
        }
        result = eval_izaac_decrypt(*ct, *key);
    }

    return result;
}

void ms_register_matrix_call_izaac_decrypt() {
    register_matrix_call("izaac_decrypt", &handle_izaac_decrypt);
}

} // namespace ms::interp
