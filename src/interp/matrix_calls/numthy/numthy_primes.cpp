// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_numthy_primes(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "numthy_primes" && assign.args.size() == 2) {
        double lo_d = 0.0;
        double hi_d = 0.0;
        if (!parse_number(assign.args[0], lo_d)) {
            auto it = ctx.state().scalars.find(assign.args[0]);
            if (it != ctx.state().scalars.end()) {
                lo_d = it->second;
            } else {
                return std::unexpected(
                    DomainError{assign.callee, "expected numeric lo argument"});
            }
        }
        if (!parse_number(assign.args[1], hi_d)) {
            auto it = ctx.state().scalars.find(assign.args[1]);
            if (it != ctx.state().scalars.end()) {
                hi_d = it->second;
            } else {
                return std::unexpected(
                    DomainError{assign.callee, "expected numeric hi argument"});
            }
        }
        if (lo_d < 0.0 || hi_d < 0.0 || std::floor(lo_d) != lo_d || std::floor(hi_d) != hi_d) {
            return std::unexpected(
                DomainError{assign.callee, "expected non-negative integer bounds"});
        }
        auto primes = eval_numthy_primes(static_cast<uint64_t>(lo_d), static_cast<uint64_t>(hi_d));
        if (!primes) {
            return std::unexpected(primes.error());
        }
        result = *primes;
    }

    return result;
}

void ms_register_matrix_call_numthy_primes() {
    register_matrix_call("numthy_primes", &handle_numthy_primes);
}

} // namespace ms::interp
