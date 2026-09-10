#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_graph_max_weight_matching(Interpreter& interp,
                                                        const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "graph_max_weight_matching" &&
        (assign.args.size() == 1 || assign.args.size() == 2)) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        bool maxcardinality = false;
        if (assign.args.size() == 2) {
            auto flag = ctx.parse_scalar_arg(assign.args[1], "graph_max_weight_matching");
            if (!flag) {
                return std::unexpected(flag.error());
            }
            maxcardinality = (*flag != 0.0);
        }
        auto mm = eval_graph_max_weight_matching(*matrix, maxcardinality);
        if (!mm) {
            return std::unexpected(mm.error());
        }
        result = *mm;
    }

    return result;
}

void ms_register_matrix_call_graph_max_weight_matching() {
    register_matrix_call("graph_max_weight_matching", &handle_graph_max_weight_matching);
}

} // namespace ms::interp
