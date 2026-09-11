// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_izaac_vrf_keygen(Interpreter& /*interp*/, const MatrixCallAssign& assign) {
    using namespace detail;

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    // Spelled as an explicit count, not assign.args.empty(), to match the other
    // 484 handlers. The two are identical to the compiler and not to
    // scripts/extract_manifest.py, which reads every guard as a predicate over
    // the argument count so that gen_matrix_call_tests.py can emit an arity test
    // for each handler. This one parsed as "depends on something other than the
    // count" and dropped out of the manifest -- and out of the generated tests --
    // without anything failing. Keep the form.
    if (assign.callee == "izaac_vrf_keygen" && assign.args.size() == 0) {
        result = eval_izaac_vrf_keygen();
    }

    return result;
}

void ms_register_matrix_call_izaac_vrf_keygen() {
    register_matrix_call("izaac_vrf_keygen", &handle_izaac_vrf_keygen);
}

} // namespace ms::interp
