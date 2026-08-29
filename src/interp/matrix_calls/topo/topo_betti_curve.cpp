#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_topo_betti_curve(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "topo_betti_curve" && assign.args.size() == 3) {
        auto dist_m = resolve_operand(assign.args[0]);
        if (!dist_m) {
            return std::unexpected(dist_m.error());
        }
        auto thresholds_m = resolve_operand(assign.args[1]);
        if (!thresholds_m) {
            return std::unexpected(thresholds_m.error());
        }
        double max_dim_d = 0.0;
        if (!parse_number(assign.args[2], max_dim_d)) {
            auto md_expr = eval_scalar_expr(ctx.state(), assign.args[2]);
            if (!md_expr) {
                return std::unexpected(md_expr.error());
            }
            max_dim_d = *md_expr;
        }
        const int max_dim = static_cast<int>(max_dim_d);
        if (max_dim < 0 || max_dim_d != max_dim) {
            return std::unexpected(
                DomainError{"topo_betti_curve", "expected non-negative integer max_dim"});
        }
        auto curve = eval_topo_betti_curve(*dist_m, *thresholds_m, max_dim);
        if (!curve) {
            return std::unexpected(curve.error());
        }
        result = *curve;
    }

    return result;
}

void ms_register_matrix_call_topo_betti_curve() {
    register_matrix_call("topo_betti_curve", &handle_topo_betti_curve);
}

} // namespace ms::interp
