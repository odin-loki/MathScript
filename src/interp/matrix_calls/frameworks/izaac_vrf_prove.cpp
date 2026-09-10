#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_izaac_vrf_prove(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "izaac_vrf_prove" && assign.args.size() == 2) {
        auto key = ctx.resolve_operand(assign.args[0]);
        if (!key) {
            return std::unexpected(key.error());
        }
        auto msg = ctx.resolve_operand(assign.args[1]);
        if (!msg) {
            return std::unexpected(msg.error());
        }
        result = eval_izaac_vrf_prove(*key, *msg);
    }

    return result;
}

void ms_register_matrix_call_izaac_vrf_prove() {
    register_matrix_call("izaac_vrf_prove", &handle_izaac_vrf_prove);
}

} // namespace ms::interp
