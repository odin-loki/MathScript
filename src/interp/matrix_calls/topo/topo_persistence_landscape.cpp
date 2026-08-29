#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_topo_persistence_landscape(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "topo_persistence_landscape" &&
               (assign.args.size() == 3 || assign.args.size() == 5)) {
        auto dgm = resolve_operand(assign.args[0]);
        if (!dgm) {
            return std::unexpected(dgm.error());
        }
        auto layers_arg = parse_scalar_arg(assign.args[1], "topo_persistence_landscape");
        if (!layers_arg) {
            return std::unexpected(layers_arg.error());
        }
        auto samples_arg = parse_scalar_arg(assign.args[2], "topo_persistence_landscape");
        if (!samples_arg) {
            return std::unexpected(samples_arg.error());
        }
        const int n_layers = static_cast<int>(*layers_arg);
        const int n_samples = static_cast<int>(*samples_arg);
        if (n_layers < 1 || *layers_arg != n_layers) {
            return std::unexpected(
                DomainError{"topo_persistence_landscape", "expected integer n_layers >= 1"});
        }
        if (n_samples < 2 || *samples_arg != n_samples) {
            return std::unexpected(
                DomainError{"topo_persistence_landscape", "expected integer n_samples >= 2"});
        }
        double t_min = 0.0;
        double t_max = 0.0;
        if (assign.args.size() == 5) {
            auto tmin = parse_scalar_arg(assign.args[3], "topo_persistence_landscape");
            if (!tmin) {
                return std::unexpected(tmin.error());
            }
            auto tmax = parse_scalar_arg(assign.args[4], "topo_persistence_landscape");
            if (!tmax) {
                return std::unexpected(tmax.error());
            }
            t_min = *tmin;
            t_max = *tmax;
        }
        result = eval_topo_persistence_landscape(*dgm, n_layers, n_samples, t_min, t_max);
    }

    return result;
}

void ms_register_matrix_call_topo_persistence_landscape() {
    register_matrix_call("topo_persistence_landscape", &handle_topo_persistence_landscape);
}

} // namespace ms::interp
