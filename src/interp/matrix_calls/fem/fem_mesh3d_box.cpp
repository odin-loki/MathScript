#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_fem_mesh3d_box(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if ((assign.callee == "fem_mesh3d_box" || assign.callee == "fem_mesh3d") &&
               assign.args.size() == 9) {
        const char* fn = assign.callee.c_str();
        std::array<Result<double>, 9> scalars{};
        for (size_t i = 0; i < 9; ++i) {
            scalars[i] = ctx.parse_scalar_arg(assign.args[i], fn);
            if (!scalars[i]) {
                return std::unexpected(scalars[i].error());
            }
        }
        auto nx_i = ctx.parse_positive_size_arg(*scalars[6], fn, "expected positive integer nx");
        if (!nx_i) {
            return std::unexpected(nx_i.error());
        }
        auto ny_i = ctx.parse_positive_size_arg(*scalars[7], fn, "expected positive integer ny");
        if (!ny_i) {
            return std::unexpected(ny_i.error());
        }
        auto nz_i = ctx.parse_positive_size_arg(*scalars[8], fn, "expected positive integer nz");
        if (!nz_i) {
            return std::unexpected(nz_i.error());
        }
        result = eval_fem_mesh3d_box(*scalars[0], *scalars[1], *scalars[2], *scalars[3],
                                     *scalars[4], *scalars[5], *nx_i, *ny_i, *nz_i);
    }

    return result;
}

void ms_register_matrix_call_fem_mesh3d_box() {
    register_matrix_call("fem_mesh3d_box", &handle_fem_mesh3d_box);
}

} // namespace ms::interp
