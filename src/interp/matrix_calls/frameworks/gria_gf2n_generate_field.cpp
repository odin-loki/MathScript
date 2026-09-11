// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_gria_gf2n_generate_field(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "gria_gf2n_generate_field" && assign.args.size() == 1) {
        auto n_val = ctx.parse_scalar_arg(assign.args[0], "gria_gf2n_generate_field");
        if (!n_val) {
            return std::unexpected(n_val.error());
        }
        const int n = static_cast<int>(*n_val);
        if (*n_val != n || n < 1 || n > 16) {
            return std::unexpected(
                DomainError{"gria_gf2n_generate_field", "expected integer n in [1,16]"});
        }
        result = eval_gria_gf2n_generate_field(n);
    }

    return result;
}

void ms_register_matrix_call_gria_gf2n_generate_field() {
    register_matrix_call("gria_gf2n_generate_field", &handle_gria_gf2n_generate_field);
}

} // namespace ms::interp
