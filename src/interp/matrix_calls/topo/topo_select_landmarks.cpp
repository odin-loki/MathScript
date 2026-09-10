#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_topo_select_landmarks(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "topo_select_landmarks" &&
               (assign.args.size() == 2 || assign.args.size() == 3)) {
        auto P = ctx.resolve_operand(assign.args[0]);
        if (!P) {
            return std::unexpected(P.error());
        }
        auto n_arg = ctx.parse_scalar_arg(assign.args[1], "topo_select_landmarks");
        if (!n_arg) {
            return std::unexpected(n_arg.error());
        }
        const int n_landmarks = static_cast<int>(*n_arg);
        if (n_landmarks < 1 || *n_arg != n_landmarks) {
            return std::unexpected(
                DomainError{"topo_select_landmarks", "expected positive integer n"});
        }
        int seed_index = 0;
        if (assign.args.size() == 3) {
            auto seed = ctx.parse_scalar_arg(assign.args[2], "topo_select_landmarks");
            if (!seed) {
                return std::unexpected(seed.error());
            }
            seed_index = static_cast<int>(*seed);
            if (*seed != seed_index) {
                return std::unexpected(
                    DomainError{"topo_select_landmarks", "expected integer seed_index"});
            }
        }
        result = eval_topo_select_landmarks(*P, n_landmarks, seed_index);
    }

    return result;
}

void ms_register_matrix_call_topo_select_landmarks() {
    register_matrix_call("topo_select_landmarks", &handle_topo_select_landmarks);
}

} // namespace ms::interp
