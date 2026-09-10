#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_imhist(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "imhist" &&
               (assign.args.size() == 1 || assign.args.size() == 2)) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        int nbins = 256;
        if (assign.args.size() == 2) {
            double nbins_d = 0.0;
            if (!parse_number(assign.args[1], nbins_d)) {
                return std::unexpected(DomainError{"imhist", "expected imhist(M[, nbins])"});
            }
            nbins = static_cast<int>(nbins_d);
            if (nbins < 1 || nbins_d != nbins) {
                return std::unexpected(
                    DomainError{"imhist", "expected positive integer nbins"});
            }
        }
        auto gray = matrix_to_gray_image(*matrix);
        if (!gray) {
            return std::unexpected(gray.error());
        }
        const auto hist = image::imhist(*gray, nbins);
        Matrix<double> out(hist.size(), 1);
        for (size_t i = 0; i < hist.size(); ++i) {
            out(i, 0) = static_cast<double>(hist[i]);
        }
        result = out;
    }

    return result;
}

void ms_register_matrix_call_imhist() {
    register_matrix_call("imhist", &handle_imhist);
}

} // namespace ms::interp
