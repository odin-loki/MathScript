#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_iradon(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "iradon" && assign.args.size() == 2) {
        auto sino_m = ctx.resolve_operand(assign.args[0]);
        if (!sino_m) {
            return std::unexpected(sino_m.error());
        }
        auto theta_m = ctx.resolve_operand(assign.args[1]);
        if (!theta_m) {
            return std::unexpected(theta_m.error());
        }
        auto theta = matrix_to_coeff_vector(*theta_m, "iradon");
        if (!theta) {
            return std::unexpected(theta.error());
        }
        if (theta->empty()) {
            return std::unexpected(DomainError{"iradon", "expected non-empty theta_deg vector"});
        }
        if (sino_m->cols() != theta->size()) {
            return std::unexpected(
                DomainError{"iradon", "sinogram column count must match theta length"});
        }
        std::vector<float> theta_f(theta->size());
        for (size_t i = 0; i < theta->size(); ++i) {
            theta_f[i] = static_cast<float>((*theta)[i]);
        }
        std::vector<std::vector<float>> sino(sino_m->cols(),
                                             std::vector<float>(sino_m->rows(), 0.f));
        for (size_t ti = 0; ti < sino_m->cols(); ++ti) {
            for (size_t ri = 0; ri < sino_m->rows(); ++ri) {
                sino[ti][ri] = static_cast<float>((*sino_m)(ri, ti));
            }
        }
        result = gray_image_to_matrix(image::iradon(sino, theta_f));
    }

    return result;
}

void ms_register_matrix_call_iradon() {
    register_matrix_call("iradon", &handle_iradon);
}

} // namespace ms::interp
