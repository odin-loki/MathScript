#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_radon(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);
    auto resolve_operand = [&ctx](const std::string& text) { return ctx.resolve_operand(text); };
    auto parse_scalar_arg = [&ctx](const std::string& arg_text, const char* fn) -> Result<double> {
        double value = 0.0;
        if (parse_number(arg_text, value)) return value;
        auto expr = eval_scalar_expr(ctx.state(), arg_text);
        if (!expr) {
            return std::unexpected(DomainError{fn, "expected numeric scalar argument"});
        }
        return *expr;
    };
    auto parse_positive_size_arg = [](double value, const char* fn, const char* label) -> Result<std::size_t> {
        const int i = static_cast<int>(value);
        if (i < 1 || value != static_cast<double>(i)) {
            return std::unexpected(DomainError{fn, label});
        }
        return static_cast<std::size_t>(i);
    };
    auto parse_uint64_arg = [](double value, const char* fn, const char* label) -> Result<uint64_t> {
        if (value < 0.0 || value != std::floor(value)) {
            return std::unexpected(DomainError{fn, label});
        }
        return static_cast<uint64_t>(value);
    };

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "radon" && assign.args.size() == 2) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto theta_m = resolve_operand(assign.args[1]);
        if (!theta_m) {
            return std::unexpected(theta_m.error());
        }
        auto theta = matrix_to_coeff_vector(*theta_m, "radon");
        if (!theta) {
            return std::unexpected(theta.error());
        }
        if (theta->empty()) {
            return std::unexpected(DomainError{"radon", "expected non-empty theta_deg vector"});
        }
        auto gray = matrix_to_gray_image(*matrix);
        if (!gray) {
            return std::unexpected(gray.error());
        }
        std::vector<float> theta_f(theta->size());
        for (size_t i = 0; i < theta->size(); ++i) {
            theta_f[i] = static_cast<float>((*theta)[i]);
        }
        const auto sino = image::radon(*gray, theta_f);
        if (sino.empty() || sino[0].empty()) {
            result = Matrix<double>(0, 0);
        } else {
            // Sinogram as rows (projection bins) Ãƒâ€” angles (columns).
            Matrix<double> out(sino[0].size(), sino.size());
            for (size_t ti = 0; ti < sino.size(); ++ti) {
                for (size_t ri = 0; ri < sino[ti].size(); ++ri) {
                    out(ri, ti) = sino[ti][ri];
                }
            }
            result = out;
        }
    }

    return result;
}

void ms_register_matrix_call_radon() {
    register_matrix_call("radon", &handle_radon);
}

} // namespace ms::interp
