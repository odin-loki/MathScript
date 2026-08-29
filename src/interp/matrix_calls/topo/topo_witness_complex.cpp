#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_topo_witness_complex(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "topo_witness_complex" &&
               (assign.args.size() == 3 || assign.args.size() == 4)) {
        auto P = resolve_operand(assign.args[0]);
        if (!P) {
            return std::unexpected(P.error());
        }
        auto landmarks = resolve_operand(assign.args[1]);
        if (!landmarks) {
            return std::unexpected(landmarks.error());
        }
        auto eps = parse_scalar_arg(assign.args[2], "topo_witness_complex");
        if (!eps) {
            return std::unexpected(eps.error());
        }
        int max_dim = 2;
        if (assign.args.size() == 4) {
            auto md = parse_scalar_arg(assign.args[3], "topo_witness_complex");
            if (!md) {
                return std::unexpected(md.error());
            }
            max_dim = static_cast<int>(*md);
            if (max_dim < 0 || *md != max_dim) {
                return std::unexpected(
                    DomainError{"topo_witness_complex", "expected non-negative integer max_dim"});
            }
        }
        result = eval_topo_witness_complex(*P, *landmarks, *eps, max_dim);
    }

    return result;
}

void ms_register_matrix_call_topo_witness_complex() {
    register_matrix_call("topo_witness_complex", &handle_topo_witness_complex);
}

} // namespace ms::interp
