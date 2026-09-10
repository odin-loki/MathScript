#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_izaac_vrf_keygen(Interpreter& /*interp*/, const MatrixCallAssign& assign) {
    using namespace detail;

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "izaac_vrf_keygen" && assign.args.empty()) {
        result = eval_izaac_vrf_keygen();
    }

    return result;
}

void ms_register_matrix_call_izaac_vrf_keygen() {
    register_matrix_call("izaac_vrf_keygen", &handle_izaac_vrf_keygen);
}

} // namespace ms::interp
