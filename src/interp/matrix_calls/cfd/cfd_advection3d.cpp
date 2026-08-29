#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_cfd_advection3d(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "cfd_advection3d" && assign.args.size() == 8) {
        double nx_d = 0.0;
        double ny_d = 0.0;
        double nz_d = 0.0;
        double vx = 0.0;
        double vy = 0.0;
        double vz = 0.0;
        double t_end = 0.0;
        double dt = 0.0;
        if (!parse_number(assign.args[0], nx_d) || !parse_number(assign.args[1], ny_d) ||
            !parse_number(assign.args[2], nz_d) || !parse_number(assign.args[3], vx) ||
            !parse_number(assign.args[4], vy) || !parse_number(assign.args[5], vz) ||
            !parse_number(assign.args[6], t_end) || !parse_number(assign.args[7], dt)) {
            return std::unexpected(DomainError{
                "cfd_advection3d",
                "expected cfd_advection3d(nx, ny, nz, vx, vy, vz, t_end, dt)"});
        }
        const int nx_i = static_cast<int>(nx_d);
        const int ny_i = static_cast<int>(ny_d);
        const int nz_i = static_cast<int>(nz_d);
        if (nx_i < 0 || ny_i < 0 || nz_i < 0 || nx_d != nx_i || ny_d != ny_i || nz_d != nz_i) {
            return std::unexpected(
                DomainError{"cfd_advection3d", "expected non-negative integer nx, ny, and nz"});
        }
        result = eval_cfd_advection3d(static_cast<std::size_t>(nx_i), static_cast<std::size_t>(ny_i),
                                        static_cast<std::size_t>(nz_i), vx, vy, vz, t_end, dt);
    }

    return result;
}

void ms_register_matrix_call_cfd_advection3d() {
    register_matrix_call("cfd_advection3d", &handle_cfd_advection3d);
}

} // namespace ms::interp
