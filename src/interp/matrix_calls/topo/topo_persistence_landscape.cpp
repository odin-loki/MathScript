// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_topo_persistence_landscape(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "topo_persistence_landscape" &&
               (assign.args.size() == 3 || assign.args.size() == 5)) {
        auto dgm = ctx.resolve_operand(assign.args[0]);
        if (!dgm) {
            return std::unexpected(dgm.error());
        }
        auto layers_arg = ctx.parse_scalar_arg(assign.args[1], "topo_persistence_landscape");
        if (!layers_arg) {
            return std::unexpected(layers_arg.error());
        }
        auto samples_arg = ctx.parse_scalar_arg(assign.args[2], "topo_persistence_landscape");
        if (!samples_arg) {
            return std::unexpected(samples_arg.error());
        }
        // The result is n_layers by n_samples, so the two have to be charged against one
        // budget: each is unremarkable at 200000 and together they are 4e10 values,
        // measured aborting the process. The casts also used to come BEFORE the range
        // checks, which is undefined rather than wrapped for a double outside int.
        if (!(*layers_arg >= 1.0) || *layers_arg != std::floor(*layers_arg)) {
            return std::unexpected(
                DomainError{"topo_persistence_landscape", "expected integer n_layers >= 1"});
        }
        if (!(*samples_arg >= 2.0) || *samples_arg != std::floor(*samples_arg)) {
            return std::unexpected(
                DomainError{"topo_persistence_landscape", "expected integer n_samples >= 2"});
        }
        ExtentBudget budget("topo_persistence_landscape");
        auto layers_bounded = budget.take("n_layers", *layers_arg);
        if (!layers_bounded) {
            return std::unexpected(layers_bounded.error());
        }
        auto samples_bounded = budget.take("n_samples", *samples_arg);
        if (!samples_bounded) {
            return std::unexpected(samples_bounded.error());
        }
        const int n_layers = static_cast<int>(*layers_bounded);
        const int n_samples = static_cast<int>(*samples_bounded);
        double t_min = 0.0;
        double t_max = 0.0;
        if (assign.args.size() == 5) {
            auto tmin = ctx.parse_scalar_arg(assign.args[3], "topo_persistence_landscape");
            if (!tmin) {
                return std::unexpected(tmin.error());
            }
            auto tmax = ctx.parse_scalar_arg(assign.args[4], "topo_persistence_landscape");
            if (!tmax) {
                return std::unexpected(tmax.error());
            }
            t_min = *tmin;
            t_max = *tmax;
        }
        result = eval_topo_persistence_landscape(*dgm, n_layers, n_samples, t_min, t_max);
    }

    return result;
}

void ms_register_matrix_call_topo_persistence_landscape() {
    register_matrix_call("topo_persistence_landscape", &handle_topo_persistence_landscape);
}

} // namespace ms::interp
