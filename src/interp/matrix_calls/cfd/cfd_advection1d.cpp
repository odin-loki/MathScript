#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_cfd_advection1d(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "cfd_advection1d" && assign.args.size() == 4) {
        auto nx_val = parse_scalar_arg(assign.args[0], "cfd_advection1d");
        if (!nx_val) {
            return std::unexpected(nx_val.error());
        }
        auto vx_val = parse_scalar_arg(assign.args[1], "cfd_advection1d");
        if (!vx_val) {
            return std::unexpected(vx_val.error());
        }
        auto t_end_val = parse_scalar_arg(assign.args[2], "cfd_advection1d");
        if (!t_end_val) {
            return std::unexpected(t_end_val.error());
        }
        auto dt_val = parse_scalar_arg(assign.args[3], "cfd_advection1d");
        if (!dt_val) {
            return std::unexpected(dt_val.error());
        }
        const int nx_i = static_cast<int>(*nx_val);
        if (nx_i < 0 || *nx_val != nx_i) {
            return std::unexpected(
                DomainError{"cfd_advection1d", "expected non-negative integer nx"});
        }
        result = eval_cfd_advection1d(static_cast<std::size_t>(nx_i), *vx_val, *t_end_val,
                                      *dt_val);
    }

    return result;
}

void ms_register_matrix_call_cfd_advection1d() {
    register_matrix_call("cfd_advection1d", &handle_cfd_advection1d);
}

} // namespace ms::interp
