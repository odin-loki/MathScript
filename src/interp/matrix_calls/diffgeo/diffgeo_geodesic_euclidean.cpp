#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_diffgeo_geodesic_euclidean(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "diffgeo_geodesic_euclidean" && assign.args.size() == 5) {
        double x0 = 0.0;
        double y0 = 0.0;
        double vx = 0.0;
        double vy = 0.0;
        double s_end = 0.0;
        if (!parse_number(assign.args[0], x0) || !parse_number(assign.args[1], y0) ||
            !parse_number(assign.args[2], vx) || !parse_number(assign.args[3], vy) ||
            !parse_number(assign.args[4], s_end)) {
            return std::unexpected(DomainError{
                "diffgeo_geodesic_euclidean",
                "expected diffgeo_geodesic_euclidean(x0,y0,vx,vy,s_end)"});
        }
        auto traj = eval_diffgeo_geodesic_euclidean(x0, y0, vx, vy, s_end);
        if (!traj) {
            return std::unexpected(traj.error());
        }
        result = *traj;
    }

    return result;
}

void ms_register_matrix_call_diffgeo_geodesic_euclidean() {
    register_matrix_call("diffgeo_geodesic_euclidean", &handle_diffgeo_geodesic_euclidean);
}

} // namespace ms::interp
