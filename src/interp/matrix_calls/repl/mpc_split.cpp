#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_mpc_split(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "mpc_split" && assign.args.size() == 3) {
        auto secret_val = parse_scalar_arg(assign.args[0], "mpc_split");
        if (!secret_val) {
            return std::unexpected(secret_val.error());
        }
        auto secret = parse_uint64_arg(*secret_val, "mpc_split", "expected unsigned integer secret");
        if (!secret) {
            return std::unexpected(secret.error());
        }
        auto n_val = parse_scalar_arg(assign.args[1], "mpc_split");
        if (!n_val) {
            return std::unexpected(n_val.error());
        }
        auto k_val = parse_scalar_arg(assign.args[2], "mpc_split");
        if (!k_val) {
            return std::unexpected(k_val.error());
        }
        const int n = static_cast<int>(*n_val);
        const int k = static_cast<int>(*k_val);
        if (*n_val != n || *k_val != k) {
            return std::unexpected(DomainError{"mpc_split", "expected integer n and k"});
        }
        result = eval_mpc_split(*secret, n, k);
    }

    return result;
}

void ms_register_matrix_call_mpc_split() {
    register_matrix_call("mpc_split", &handle_mpc_split);
}

} // namespace ms::interp
