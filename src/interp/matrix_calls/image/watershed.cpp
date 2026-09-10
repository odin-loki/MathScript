#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_watershed(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "watershed" && assign.args.size() == 2) {
        auto gray_m = ctx.resolve_operand(assign.args[0]);
        if (!gray_m) {
            return std::unexpected(gray_m.error());
        }
        auto markers_m = ctx.resolve_operand(assign.args[1]);
        if (!markers_m) {
            return std::unexpected(markers_m.error());
        }
        auto gray = matrix_to_gray_image(*gray_m);
        if (!gray) {
            return std::unexpected(gray.error());
        }
        // Marker labels are integer IDs, not intensities Ã¢â‚¬â€� do not apply 0..255 scaling.
        if (markers_m->rows() == 0 || markers_m->cols() == 0) {
            return std::unexpected(DomainError{"watershed", "empty markers matrix"});
        }
        image::Image markers(static_cast<int>(markers_m->rows()),
                             static_cast<int>(markers_m->cols()), 1);
        for (size_t r = 0; r < markers_m->rows(); ++r) {
            for (size_t c = 0; c < markers_m->cols(); ++c) {
                markers.at(static_cast<int>(r), static_cast<int>(c), 0) =
                    static_cast<float>((*markers_m)(r, c));
            }
        }
        result = gray_image_to_matrix(image::watershed(*gray, markers));
    }

    return result;
}

void ms_register_matrix_call_watershed() {
    register_matrix_call("watershed", &handle_watershed);
}

} // namespace ms::interp
