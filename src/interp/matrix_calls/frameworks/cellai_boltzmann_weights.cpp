// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_cellai_boltzmann_weights(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "cellai_boltzmann_weights" &&
               (assign.args.size() == 1 || assign.args.size() == 2)) {
        auto resolve_coeff_vector = [&](const std::string& text,
                                        const char* fn) -> Result<std::vector<double>> {
            auto parsed = parse_bracket_vector_literal(trim_copy(text), fn);
            if (parsed) {
                return *parsed;
            }
            auto matrix = ctx.resolve_operand(text);
            if (!matrix) {
                return std::unexpected(matrix.error());
            }
            return matrix_to_coeff_vector(*matrix, fn);
        };
        auto energies = resolve_coeff_vector(assign.args[0], "cellai_boltzmann_weights");
        if (!energies) {
            return std::unexpected(energies.error());
        }
        double temperature = 1.0;
        if (assign.args.size() == 2) {
            auto temp = ctx.parse_scalar_arg(assign.args[1], "cellai_boltzmann_weights");
            if (!temp) {
                return std::unexpected(temp.error());
            }
            temperature = *temp;
        }
        result = eval_cellai_boltzmann_weights(*energies, temperature);
    }

    return result;
}

void ms_register_matrix_call_cellai_boltzmann_weights() {
    register_matrix_call("cellai_boltzmann_weights", &handle_cellai_boltzmann_weights);
}

} // namespace ms::interp
