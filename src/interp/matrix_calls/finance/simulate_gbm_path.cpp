// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_simulate_gbm_path(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "simulate_gbm_path" && assign.args.size() == 5) {
        const char* fn = "simulate_gbm_path";
        std::array<Result<double>, 5> scalars{};
        for (size_t i = 0; i < 5; ++i) {
            scalars[i] = ctx.parse_scalar_arg(assign.args[i], fn);
            if (!scalars[i]) {
                return std::unexpected(scalars[i].error());
            }
        }
        auto steps_i = ctx.parse_positive_size_arg(*scalars[4], fn, "expected positive integer steps");
        if (!steps_i) {
            return std::unexpected(steps_i.error());
        }
        result = eval_simulate_gbm_path(*scalars[0], *scalars[1], *scalars[2], *scalars[3], *steps_i);
    }

    return result;
}

void ms_register_matrix_call_simulate_gbm_path() {
    register_matrix_call("simulate_gbm_path", &handle_simulate_gbm_path);
}

} // namespace ms::interp
