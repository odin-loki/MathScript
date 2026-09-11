// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_quantum_ket_basis(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "quantum_ket_basis" && assign.args.size() == 2) {
        double dim_d = 0.0;
        double index_d = 0.0;
        if (!parse_number(assign.args[0], dim_d)) {
            auto it = ctx.state().scalars.find(assign.args[0]);
            if (it != ctx.state().scalars.end()) {
                dim_d = it->second;
            } else {
                return std::unexpected(
                    DomainError{assign.callee, "expected numeric dim argument"});
            }
        }
        if (!parse_number(assign.args[1], index_d)) {
            auto it = ctx.state().scalars.find(assign.args[1]);
            if (it != ctx.state().scalars.end()) {
                index_d = it->second;
            } else {
                return std::unexpected(
                    DomainError{assign.callee, "expected numeric index argument"});
            }
        }
        const int dim = static_cast<int>(dim_d);
        const int index = static_cast<int>(index_d);
        if (dim < 1 || dim_d != dim || index_d != index) {
            return std::unexpected(DomainError{
                assign.callee, "expected positive integer dim and integer index"});
        }
        result = eval_quantum_ket_basis(dim, index);
    }

    return result;
}

void ms_register_matrix_call_quantum_ket_basis() {
    register_matrix_call("quantum_ket_basis", &handle_quantum_ket_basis);
}

} // namespace ms::interp
